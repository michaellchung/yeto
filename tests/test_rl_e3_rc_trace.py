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


def test_bwd_scale_makes_dp2_gradients_bit_comparable(tmp_path):
    """DP2 divides the loss by 8, DP1 by 16: with bwd_scale=1/dp the recorded gradients, wgrads and saved tensors of the
    two arms are bit-identical for the same micro batch (power-of-two scaling is exact)."""
    def run(scale, loss_mult, sub):
        rc_trace.reset_for_tests()
        torch.manual_seed(0)
        net = Net().to(torch.bfloat16)
        rc_trace.install_trace(SimpleNamespace(model=[net]), bwd_scale=scale, save_mb=(0,),
                               name_filter=r"layers\.\d+$")
        x = (torch.arange(16, dtype=torch.float32).reshape(2, 8) * 0.1).to(torch.bfloat16)
        (net(x) * loss_mult).backward()
        info = rc_trace.dump_trace(SimpleNamespace(), directory=str(tmp_path / sub), tag="t", coord={"dp": 0})
        return torch.load(Path(info["path"]), weights_only=False)

    a = run(1.0, 1.0, "dp1")
    b = run(0.5, 2.0, "dp2")
    assert {k: v["bits"] for k, v in a["records"].items() if "|bwd_" in k} == \
        {k: v["bits"] for k, v in b["records"].items() if "|bwd_" in k}
    assert torch.equal(a["wgrad"][0]["c0.layers.0.lin.weight"], b["wgrad"][0]["c0.layers.0.lin.weight"])
    assert b["bwd_scale"] == 0.5 and "torch" in b["device"]


def test_kernel_profile_never_breaks_the_step(tmp_path):
    rc_trace.reset_for_tests()
    torch.manual_seed(0)
    net = Net().to(torch.bfloat16)
    rc_trace.install_trace(SimpleNamespace(model=[net]), profile_kernels=True, name_filter=r"layers\.\d+$")
    for k in (0, 1, 2):
        net(torch.ones(2, 8, dtype=torch.bfloat16) * k).backward()
    info = rc_trace.dump_trace(SimpleNamespace(), directory=str(tmp_path), tag="t", coord={"dp": 0})
    payload = torch.load(Path(info["path"]), weights_only=False)
    assert payload["n_microbatches"] == 3 and isinstance(payload["kernels"], dict)  # kernels, or {"__error__": ...} on CPU
    assert "env" in payload["device"] and "transformer_engine" in payload["device"]


def test_forward_only_pass_and_loss_level_capture(tmp_path, monkeypatch):
    import types

    fake_pkg = {}
    for name in ("miles", "miles.backends", "miles.backends.training_utils"):
        fake_pkg[name] = types.ModuleType(name)
    loss_mod = types.ModuleType("miles.backends.training_utils.loss")

    def get_loss_function(*a, **k):
        def func(args, batch, logits, sum_of_sample_mean, *rest, **kw):
            loss = logits.float().pow(2).sum() * batch["advantages"][0]
            return loss, {"pg_loss": loss.detach(), "ppo_kl": torch.zeros(())}
        return func

    loss_mod.get_loss_function = get_loss_function
    fake_pkg["miles.backends.training_utils.loss"] = loss_mod
    fake_pkg["miles.backends.training_utils"].loss = loss_mod
    for k, v in fake_pkg.items():
        monkeypatch.setitem(sys.modules, k, v)

    rc_trace.reset_for_tests()
    torch.manual_seed(0)
    net = Net().to(torch.bfloat16)
    rc_trace.install_trace(SimpleNamespace(model=[net]), forward_only=True, loss_level=True, logit_grad_mb=(-1,),
                           bwd_scale=0.5, name_filter=r"layers\.\d+$", fo_filter=r"layers\.\d+$")
    with torch.no_grad():
        net(torch.ones(2, 8, dtype=torch.bfloat16))  # old-policy log-prob pass
    x = torch.ones(4, 8, dtype=torch.bfloat16, requires_grad=True)
    logits = net.layers[0](x * 1.0)
    func = loss_mod.get_loss_function()
    loss, log = func(None, {"advantages": [torch.tensor(2.0)], "log_probs": [torch.zeros(3)]}, logits, None)
    loss.backward()
    info = rc_trace.dump_trace(SimpleNamespace(), directory=str(tmp_path), tag="t", coord={"dp": 0})
    payload = torch.load(Path(info["path"]), weights_only=False)
    assert any(k.startswith("fo0|") and "|<root-input>|in|" in k for k in payload["records"])
    assert any(k.startswith("fo0|c0.layers.2|fwd") for k in payload["records"])
    assert not any(k.startswith("0|") and "layers.2|fwd" in k for k in payload["records"])  # the training mb not run via root
    # loss-level capture is keyed by the (so far -1) training mb counter; here only check the wrapper ran
    small = payload["small"]
    assert small and any("loss_metrics" in v and "pg_loss" in v["loss_metrics"] for v in small.values())
    assert any("logit_grad_row_bits" in v for v in small.values())
    assert payload["big"] and next(iter(payload["big"].values())).shape == logits.shape
