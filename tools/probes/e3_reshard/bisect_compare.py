"""Offline readout of an a8-rootcause RC-4 run (plan.md section 7): which DP2 arm is in the 'bad' numeric mode?

    python bisect_compare.py <run dir> [--bad <packed .pt of a known bad-mode DP2 step-3 state>]
                                       [--good <packed .pt of a known good-mode DP2 step-3 state>] [--out result.json]

Every DP2 arm restores C1 (eA1's step-2 state) and trains step 3; eA1 is the DP1 reference. For each DP2 arm the
step gradient (from exp_avg) relative L2 against eA1 is reported, and whether its step-3 state is bitwise equal to
the DP1 state, to the reference bad-mode state (A8's B1_s3 / RC-2's rcBc_s3) and to the reference good-mode state
(RC-3's dBc_s3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_rc import _load, bitwise_equal, state_metrics  # noqa: E402


def main(argv):
    run = Path(argv[0])
    packed = (run / "work" if (run / "work").is_dir() else run) / "packed"
    ref = {k: _load(Path(argv[argv.index(f"--{k}") + 1])) for k in ("bad", "good") if f"--{k}" in argv}
    c1 = _load(packed / "eA1_s2.pt")
    dp1 = _load(packed / "eA1_s3.pt")
    out = {}
    for p in sorted(packed.glob("e*_s3.pt")):
        arm = p.name[:-6]
        if arm in ("eA1", "eA2"):
            continue
        st = _load(p)
        m = state_metrics(c1, dp1, c1, st)
        out[arm] = {"grad_rel_l2_vs_DP1": m["grad_rel_l2"], "update_rel_l2_vs_DP1": m["update_rel_l2"],
                    "sign_agreement": m["sign_agreement"], "equal_DP1": bitwise_equal(st, dp1)["equal"],
                    **{f"equal_{k}_mode_ref": bitwise_equal(st, v)["equal"] for k, v in ref.items()}}
        out[arm]["mode"] = "bad" if m["grad_rel_l2"] >= 3e-3 else ("good" if m["grad_rel_l2"] <= 1e-6 else "other")
    text = json.dumps(out, indent=1, sort_keys=True)
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
