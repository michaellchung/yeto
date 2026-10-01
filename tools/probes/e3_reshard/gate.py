"""RC-5 gate: is this container's DP2 in the 'bad' numeric mode?  (a8-rootcause plan.md section 9)

    python gate.py <work> <arm> <tag> <reference packed state> <threshold>

Packs the arm's dumps (pack_states), compares the arm's exp_avg with the reference (the DP1 state of A8, which is
bitwise identical in every container seen so far) and exits 0 when the relative L2 difference exceeds the threshold
(bad mode: continue with the deep arms), 3 when it does not (good host: stop), 4 on any error (continue).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv):
    work, arm, tag, ref, threshold = Path(argv[0]), argv[1], argv[2], argv[3], float(argv[4])
    try:
        import torch
        from compare_rc import _load, flat, rel
        from miles.backends.megatron_utils.lora.dp_invariant_state import merge_named_optimizer_states
        import pack_states
        from compare import _load as load_state

        pack_states.pack(work, merge_named_optimizer_states, load_state)
        mine = _load(work / "packed" / f"{arm}_{tag}.pt")
        other = _load(Path(ref))
        d = rel(flat(other, "exp_avg"), flat(mine, "exp_avg"))
        print(f"GATE exp_avg_rel_l2={d:.3e} threshold={threshold:.1e}", flush=True)
        return 0 if d > threshold else 3
    except Exception as exc:  # noqa: BLE001
        print(f"GATE error {type(exc).__name__}: {exc}", flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
