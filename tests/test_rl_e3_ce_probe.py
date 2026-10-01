"""CPU test of the fused-cross-entropy probe in rc_trace (a8-rootcause plan.md section 12).

Runs in a subprocess with TORCHDYNAMO_DISABLE=1 (no compiler needed); skipped when megatron.core is absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
PROBES = str(Path(__file__).resolve().parents[1] / "tools" / "probes" / "e3_reshard")

SCRIPT = r"""
import sys, os, tempfile
sys.path.insert(0, %r)
import torch, torch.distributed as dist
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:%d", rank=0, world_size=1)
from types import SimpleNamespace
import rc_trace
from megatron.core.fusions import fused_cross_entropy as fce

G = lambda: dist.group.WORLD
def run(probe):
    rc_trace.reset_for_tests()
    actor = SimpleNamespace(model=[torch.nn.Linear(2, 2)])
    if probe:
        rc_trace.install_trace(actor, module_hooks=False, loss_level=False, save_wgrads=False, ce_probe=True, ce_save=1)
        rc_trace._ST["mb"] = 0
    outs = []
    torch.manual_seed(0)
    for k in range(3):
        logits = (torch.randn(1, 5, 64) * 3).bfloat16().requires_grad_(True)
        target = torch.randint(0, 64, (1, 5))
        loss = fce.fused_vocab_parallel_cross_entropy(logits, target, G())
        w = torch.tensor([[0.0, 1.0, 0.5, 0.0, 2.0]]) if k != 1 else torch.zeros(1, 5)
        (loss * w).sum().backward()
        outs.append((loss.detach().clone(), logits.grad.clone()))
    return actor, outs
_, base = run(False)
actor, probed = run(True)
for (l0, g0), (l1, g1) in zip(base, probed):
    assert torch.equal(l0, l1) and torch.equal(g0, g1), "the probe changed the computation"
r = rc_trace.dump_trace(actor, directory=tempfile.mkdtemp(), tag="t", coord={"dp": 0})
p = torch.load(r["path"], weights_only=False)
calls = p["ce"]
bwd = [c["bwd"] for c in calls if "bwd" in c]
fwd = [c for c in calls if "grad" in c]
assert len(fwd) == 3 and len(bwd) == 3, (len(fwd), len(bwd))
assert all(b["fwd_ptr_match"] for b in bwd)
assert [b["grad_output_nonzero"] for b in bwd] == [True, False, True]
assert all(b["compiled_equal"] for b in bwd)  # same input, same function: bitwise repeat
assert all(b["eager_equal"] for b in bwd)     # compile disabled here: eager is the same code
assert sorted(k for k in p["big"] if k.endswith("softmax_in")) == ["ce0|softmax_in"]  # ce_save=1: first non-zero only
for key in ("logits_in", "logits_max", "exp_logits", "softmax", "loss"):
    assert key in fwd[0], key
assert p["ce"][-1]["errors"] == []
print("OK")
"""


def test_ce_probe_is_read_only_and_records():
    port = 20000 + os.getpid() % 20000
    env = {**os.environ, "TORCHDYNAMO_DISABLE": "1"}
    probe = subprocess.run([sys.executable, "-c", "import megatron.core"], capture_output=True, env=env)
    if probe.returncode != 0:
        pytest.skip("megatron.core not importable")
    res = subprocess.run([sys.executable, "-c", SCRIPT % (PROBES, port)], capture_output=True, text=True, env=env)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout[-2000:] + res.stderr[-3000:]
