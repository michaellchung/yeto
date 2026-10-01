"""CPU tests for the RC-6 kernel probe tooling (no torch/GPU needed)."""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "probes" / "e3_reshard"))
import kernel_probe_compare as kc  # noqa: E402


def _run(variant, tag, grad, cfg="w4"):
    s = {"T": 10, "loss": "l", "logp": "p", "grad_output": "g", "logit_grad": grad}
    return {"variant": variant, "tag": tag, "samples": {"0": s}, "inventory": {"cubin": {"k.cubin": {"sha": cfg, "num_warps": 4}}, "best_config": {}}}


def test_all_equal_and_config_identical():
    r = kc.summarize([_run("a", "a_r0_g0", "x"), _run("a", "a_r0_g1", "x")])
    assert r["a"]["all_equal"] and r["a"]["cfg_identical_across_workers"]


def test_difference_in_logit_grad_is_reported_with_worker():
    r = kc.summarize([_run("a", "a_r0_g0", "x"), _run("a", "a_r0_g1", "y", cfg="w8")])
    assert not r["a"]["all_equal"] and r["a"]["differing"] == ["a_r0_g1"]
    assert r["a"]["per_field"]["logit_grad"]["n_diff"] == 1 and not r["a"]["cfg_identical_across_workers"]


def test_probe_scripts_parse_and_variants_are_pinned():
    for n in ("kernel_probe.py", "kernel_probe_run.py"):
        ast.parse((Path(kc.__file__).parent / n).read_text())
    src = (Path(kc.__file__).parent / "kernel_probe.py").read_text()
    assert 'TS = [384, 347, 291, 402, 256, 331, 365, 318]' in src and 'PHASES = [("a", 3, "shared"), ("b", 1, "shared"), ("c1", 2, "shared"), ("c2", 1, "pinned")]' in src
