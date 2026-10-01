"""Offline (CPU) re-analysis of the A8 run-2 packed states (a8-rootcause plan.md §0).

    python offline_g4_layers.py <packed dir>

1. A1_s2 vs A2_s2 (DP1 vs DP2, trained from the same init/seed/data, NO restore anywhere): per-layer
   relative L2 and bitwise-equal fraction of exp_avg / exp_avg_sq / master param.
2. A1_s3 vs B1_s3 (both from C1): effective gradient g=(m3-0.9*m2)/0.1 per layer, plus the bitwise-equal
   fraction of exp_avg (fp32 absorption of tiny differences into 0.9*m2 means equality does NOT imply equal g).
"""
import collections
import re
import sys
from pathlib import Path

import torch

D = Path(sys.argv[1])
L = lambda n: torch.load(D / f"{n}.pt", weights_only=False)["optimizer"]  # noqa: E731
T = lambda o, k, n: o[k]["tensors"][n]  # noqa: E731
layer = lambda k: int(re.search(r"layers\.(\d+)", k).group(1))  # noqa: E731
SHOW = (0, 5, 13, 20, 24, 25, 26, 27)


def per_layer(a, b, fn):
    by = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])
    for k in a:
        x, y = fn(a, k), fn(b, k)
        v = by[layer(k)]
        v[0] += ((x - y) ** 2).sum().item()
        v[1] += (x ** 2).sum().item()
        v[2] += (x == y).sum().item()
        v[3] += x.numel()
    return by


print("== 1. A1_s2 vs A2_s2 (no restore) ==")
a, b = L("A1_s2"), L("A2_s2")
for name in ("exp_avg", "exp_avg_sq", "param"):
    by = per_layer(a, b, lambda o, k, name=name: T(o, k, name).double())
    tot = (sum(v[0] for v in by.values()) / sum(v[1] for v in by.values())) ** 0.5
    print(f"{name}: total rel L2 {tot:.3e}; layer: rel / bitwise-equal fraction")
    print("  " + "  ".join(f"{l}: {(v[0] / v[1]) ** .5:.2e}/{v[2] / v[3]:.3f}" for l, v in sorted(by.items()) if l in SHOW))

print("== 2. A1_s3 vs B1_s3 from C1 (effective gradient) ==")
c1, a3, b3 = L("A1_s2"), L("A1_s3"), L("B1_s3")
g = lambda o, k: (T(o, k, "exp_avg").double() - 0.9 * T(c1, k, "exp_avg").double()) / 0.1  # noqa: E731
by = per_layer(a3, b3, g)
tot = (sum(v[0] for v in by.values()) / sum(v[1] for v in by.values())) ** 0.5
print(f"gradient total rel L2 {tot:.3e}")
print("  " + "  ".join(f"{l}: {(v[0] / v[1]) ** .5:.2e}" for l, v in sorted(by.items()) if l in SHOW))
eq = per_layer(a3, b3, lambda o, k: T(o, k, "exp_avg"))
print("exp_avg bitwise-equal fraction by layer: " + "  ".join(f"{l}: {v[2] / v[3]:.3f}" for l, v in sorted(eq.items()) if l in SHOW))
