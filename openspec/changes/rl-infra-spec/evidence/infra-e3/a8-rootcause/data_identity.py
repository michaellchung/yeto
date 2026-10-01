"""Is the RC frozen data the A8 frozen data? Compare per-sample step-1 losses (hex) of A8's A1 with RC's rcA1.

    python data_identity.py <A8 A1.events.jsonl.gz> <RC rcA1/events.jsonl>

Same data AND same kernels (both on H100, same code path for DP1 step 1) => bitwise-equal losses for every
step (1, 2, 3 are compared as far as both runs have them). Anything else => different data (or kernels).
"""
import gzip
import json
import sys


def losses(path):
    op = gzip.open if path.endswith(".gz") else open
    out = {}
    with op(path, "rt") as fh:
        for line in fh:
            e = json.loads(line)
            if e["kind"] == "train":
                out[e["step"]] = [float.fromhex(r["loss_hex"]) for rank in e["probe"] for r in rank if r["kind"] == "loss"]
    return out


a, b = losses(sys.argv[1]), losses(sys.argv[2])
for step in sorted(set(a) & set(b)):
    same = a[step] == b[step]
    print(f"step {step}: A8 n={len(a[step])} RC n={len(b[step])} bitwise-equal={same}"
          + ("" if same else f" max rel diff {max(abs(x - y) / max(abs(x), 1e-30) for x, y in zip(a[step], b[step])):.3e}"))
