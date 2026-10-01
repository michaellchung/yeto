"""CPU test of the A8 root-cause trace plugin (tools/probes/e3_reshard/rc_trace.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "probes" / "e3_reshard"))
import rc_trace  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


def _run(seed_shift=0.0, mbs=(0, 1)):
    rc_trace.reset_for_tests()
    torch.manual_seed(0)
    net = Net().to(torch.bfloat16)
    actor = SimpleNamespace(model=[net])
    rc_trace.install_trace(actor, save_mb=(0,), name_filter=r"layers\.\d+$")
    # forward-only pass (no_grad) must not count as a micro batch
    with torch.no_grad():
        net(torch.ones(2, 8, dtype=torch.bfloat16))
    for k in mbs:
        x = (torch.arange(16, dtype=torch.float32).reshape(2, 8) * 0.1 + k + seed_shift).to(torch.bfloat16)
        net(x).backward()
    return rc_trace.dump_trace(actor, directory=str(Path(rc_trace_dir)), tag="t", coord={"dp": 0})


rc_trace_dir = ""


def test_trace_is_deterministic_and_sensitive(tmp_path):
    global rc_trace_dir
    rc_trace_dir = str(tmp_path / "a")
    r1 = _run()
    a = torch.load(Path(r1["path"]), weights_only=False)
    rc_trace_dir = str(tmp_path / "b")
    r2 = _run()
    b = torch.load(Path(r2["path"]), weights_only=False)
    assert a["n_microbatches"] == 2 and r1["microbatches"] == 2  # no_grad pass not counted
    assert a["records"] == b["records"]  # identical computation -> identical bits
    keys = set(a["records"])
    assert any("|c0.layers.2|bwd_out|" in k for k in keys) and any("|c0.layers.1.lin|bwd_in|" in k for k in keys)
    assert any("|<root-input>|in|" in k for k in keys)
    assert set(a["wgrad"]) == {0, 1} and "c0.layers.0.lin.weight" in a["wgrad"][0]
    assert a["wgrad"][0]["c0.layers.0.lin.weight"].dtype == torch.bfloat16
    assert any(k.startswith("0|") and "c0.layers.1|bwd_out" in k for k in a["tensors"])
    assert not any(k.startswith("1|") for k in a["tensors"])  # save_mb=(0,)
    # a one-ulp change of the input changes some checksum, a repeat does not
    rc_trace_dir = str(tmp_path / "c")
    c = torch.load(Path(_run(seed_shift=0.01)["path"]), weights_only=False)
    assert c["records"] != a["records"]


def test_compare_rc_finds_the_first_differing_record_and_wgrad(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "probes" / "e3_reshard"))
    import compare_rc

    def rec(bits, l2=1.0):
        return {"bits": bits, "l2": l2, "shape": (2,), "dtype": "bf16"}

    base = {"0|m1|bwd_out|0": rec([1, 2]), "0|m2|bwd_out|0": rec([3, 4]), "0|m3|bwd_out|0": rec([5, 6]),
            "0|m1|fwd|0": rec([7, 8])}
    other = dict(base)
    other["0|m2|bwd_out|0"] = rec([3, 5], 1.0001)
    other["0|m3|bwd_out|0"] = rec([5, 7], 1.0001)
    w = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    pa = {"records": base, "wgrad": {0: {"p": w}}}
    pb = {"records": other, "wgrad": {0: {"p": w * 2}}}  # DP2 arm: exactly 2x
    a = compare_rc.collect({0: pa}, {0: [11]})
    b = compare_rc.collect({0: pb}, {0: [11]})
    rep = compare_rc.trace_report(a, b)["11"]
    assert rep["bwd_out"]["first"]["key"] == "m2|bwd_out|0" and rep["bwd_out"]["differ"] == 2
    assert rep["fwd"]["differ"] == 0
    assert compare_rc.wgrad_report(a, b, 1.0, 0.5)["per_sample"]["11"]["differ"] == 0
    assert compare_rc.wgrad_report(a, b)["per_sample"]["11"]["differ"] == 1
