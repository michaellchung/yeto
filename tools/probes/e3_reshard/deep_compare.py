"""Offline readout of an a8-rootcause RC-3 run (plan.md section 6; arms dA1, dBc, dSaX, dSbX).

    python deep_compare.py <run dir containing work/packed and work/arms> [--out result.json]

For every sample with a non-zero loss and for the pairs

    dA1 vs dBc    (DP1 vs DP2 from the same C1; the pair that fails G4)
    dA1 vs dSaX   (DP1 on physical GPU 0 vs on physical GPU 1)
    dBc vs dSbX   (DP2 as before vs DP2 with the rank/GPU assignment swapped)
    dSaX vs dSbX  (DP1 vs DP2 after the swap)

it reports (all bitwise): the loss inputs (old log-probs, advantages, masks, lengths, rewards), the loss and its
metrics, the logits forward rows, the rows of the logits gradient, the forward-only (old log-prob) pass records and
the per-parameter micro-batch wgrads; and, for the micro batch whose full logits gradient was saved, an
element-level description of where the two gradients differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compare_rc import _load, _records, collect, first_difference, read_events, sample_order  # noqa: E402

PAIRS = (("dA1", "dBc"), ("dA1", "dSaX"), ("dBc", "dSbX"), ("dSaX", "dSbX"))
SMALL_KEYS = ("log_probs", "advantages", "loss_masks", "response_lengths", "total_lengths", "rewards")


def _eq(a: Any, b: Any) -> bool:
    import torch

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and torch.equal(a, b)
    return a == b


def _maxdiff(a: Any, b: Any) -> float | None:
    import torch

    if isinstance(a, (list, tuple)):
        vals = [_maxdiff(x, y) for x, y in zip(a, b)]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.is_floating_point():
        return float((a.double() - b.double()).abs().max())
    return None


def sample_report(px: dict, kx: int, py: dict, ky: int) -> dict[str, Any]:
    import torch

    sx, sy = px["small"].get(kx, {}), py["small"].get(ky, {})
    out: dict[str, Any] = {"inputs_equal": {}, "inputs_maxdiff": {}}
    for key in SMALL_KEYS:
        if key in sx and key in sy:
            out["inputs_equal"][key] = _eq(sx[key], sy[key])
            if not out["inputs_equal"][key]:
                out["inputs_maxdiff"][key] = _maxdiff(sx[key], sy[key])
    out["loss_equal"] = sx.get("loss") == sy.get("loss")
    out["metrics_equal"] = sx.get("loss_metrics") == sy.get("loss_metrics")
    if not out["metrics_equal"]:
        out["metrics"] = {k: (sx.get("loss_metrics", {}).get(k), sy.get("loss_metrics", {}).get(k))
                          for k in set(sx.get("loss_metrics", {})) | set(sy.get("loss_metrics", {}))
                          if sx.get("loss_metrics", {}).get(k) != sy.get("loss_metrics", {}).get(k)}
    for key in ("logits_row_bits", "logit_grad_row_bits"):
        if key in sx and key in sy:
            d = (sx[key] != sy[key]).any(0)
            rows = [int(i) for i in torch.nonzero(d).reshape(-1)]
            out[key] = {"rows": int(d.numel()), "rows_differ": len(rows), "first_rows": rows[:10]}
    if "logit_grad_row_abs" in sx and "logit_grad_row_abs" in sy:
        a, b = sx["logit_grad_row_abs"], sy["logit_grad_row_abs"]
        nz = a.abs() > 0
        out["logit_grad_row_abs_max_rel"] = float(((a - b).abs()[nz] / a.abs()[nz]).max()) if bool(nz.any()) else 0.0
    ra, rb = _records(px, f"fo{kx}"), _records(py, f"fo{ky}")
    if ra and rb:
        out["forward_only"] = first_difference(ra, rb, ("fwd", "in"))
    ta, tb = _records(px, kx), _records(py, ky)
    out["training_fwd"] = first_difference(ta, tb, ("fwd", "in"))
    out["training_bwd_out"] = first_difference(ta, tb, ("bwd_out",))
    out["training_bwd_in"] = first_difference(ta, tb, ("bwd_in",))
    ga, gb = px["wgrad"].get(kx, {}), py["wgrad"].get(ky, {})
    out["wgrad_params_differ"] = sum(1 for n in ga if not torch.equal(ga[n], gb[n])) if ga and gb else None
    return out


def big_gradient_report(px: dict, kx: int, py: dict, ky: int) -> dict[str, Any] | None:
    """Element-level comparison of the full logits gradient (saved for one micro batch per arm)."""
    import torch

    a, b = px["big"].get(f"{kx}|logit_grad"), py["big"].get(f"{ky}|logit_grad")
    if a is None or b is None:
        return None
    d = a.double() - b.double()
    nz = d != 0
    n = int(nz.sum())
    out: dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype), "elements_differ": n, "elements": a.numel()}
    if not n:
        return out
    idx = torch.nonzero(nz)
    rows = sorted({int(i[-2]) for i in idx})
    cols = sorted({int(i[-1]) for i in idx})
    av = a.double()[nz]
    bv = b.double()[nz]
    out.update({"rows_differ": len(rows), "rows": rows[:40], "cols_differ": len(cols), "cols": cols[:40],
                "max_abs_diff": float(d.abs().max()), "max_rel_diff_at_differing": float((d[nz].abs() / av.abs().clamp_min(1e-300)).max()),
                "median_rel_diff_at_differing": float((d[nz].abs() / av.abs().clamp_min(1e-300)).median()),
                "abs_value_quantiles_at_differing": [float(q) for q in torch.quantile(av.abs().float(), torch.tensor([0.0, 0.5, 1.0]))],
                "per_row_count_top": sorted(((int(nz.reshape(-1, nz.shape[-1])[r].sum()), r) for r in rows), reverse=True)[:8]})
    # is the difference concentrated in the largest-magnitude entries of its row (target-token column) or spread?
    a2 = a.double().reshape(-1, a.shape[-1])
    big_col = a2.abs().argmax(-1)
    hit = sum(1 for r in rows if bool(nz.reshape(-1, nz.shape[-1])[r, int(big_col[r])]))
    out["rows_where_the_largest_entry_differs"] = hit
    return out


def main(argv: list[str]) -> int:
    run = Path(argv[0])
    work = run / "work" if (run / "work").is_dir() else run
    packed = work / "packed"
    arms = sorted({p.name.split("_")[0] for p in packed.glob("d*_trace_s3_dp*.pt")})
    res: dict[str, Any] = {"arms": arms, "devices": {}, "pairs": {}}
    traces: dict[str, dict[int, dict]] = {}
    for arm in arms:
        traces[arm] = {int(p.stem.rsplit("dp", 1)[1]): _load(p) for p in sorted(packed.glob(f"{arm}_trace_s3_dp*.pt"))}
        res["devices"][arm] = {str(dp): {"coord": t["coord"], "uuid": t["device"].get("uuid"),
                                         "index": t["device"].get("device_index"),
                                         "CUDA_VISIBLE_DEVICES": (t["device"].get("env") or {}).get("CUDA_VISIBLE_DEVICES")}
                               for dp, t in traces[arm].items()}
    ev = {arm: read_events(work / "arms" / arm) for arm in arms}
    orders = {arm: sample_order(ev[arm], 3) for arm in arms}
    cols = {arm: collect(traces[arm], orders[arm]) for arm in arms}
    # samples with a non-zero loss are the ones with a gradient
    train = next(e for e in ev[arms[0]] if e["kind"] == "train" and e["step"] == 3)
    losses = {i: float.fromhex(r["loss_hex"]) for rank in train["probe"] for r in rank if r["kind"] == "loss" for i in r["sample_indices"]}
    active = sorted(i for i, v in losses.items() if v != 0.0)
    res["active_samples"] = active
    for x, y in PAIRS:
        if x not in cols or y not in cols:
            continue
        rep = {}
        for sid in active:
            (px, kx), (py, ky) = cols[x][sid], cols[y][sid]
            rep[str(sid)] = {"rank_x": px["coord"]["dp"], "rank_y": py["coord"]["dp"], **sample_report(px, kx, py, ky)}
            big = big_gradient_report(px, kx, py, ky)
            if big is not None:
                rep[str(sid)]["logit_grad_elements"] = big
        res["pairs"][f"{x}_vs_{y}"] = rep
    state = {}
    try:
        from compare_rc import bitwise_equal

        for x, y in PAIRS:
            if (packed / f"{x}_s3.pt").is_file() and (packed / f"{y}_s3.pt").is_file():
                state[f"{x}_vs_{y}"] = bitwise_equal(_load(packed / f"{x}_s3.pt"), _load(packed / f"{y}_s3.pt"))["equal"]
    except Exception as exc:  # noqa: BLE001
        state["error"] = repr(exc)
    res["step3_state_bitwise_equal"] = state
    text = json.dumps(res, indent=1, sort_keys=True, default=repr)
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
