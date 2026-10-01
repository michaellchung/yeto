"""Offline analysis of an a8-rootcause run (plan.md §2 readout; reads the retrieved ``packed/`` dir + events).

    python compare_rc.py <run dir containing work/packed and work/arms> [--out result.json]

Three parts, all decided by the rules written in plan.md before the run:

* ``states``  bitwise / relative comparisons of the step-3 states of the five arms (R1);
* ``trace``   per-sample alignment of the DP1 and DP2 micro-batch traces and the first differing record in
  execution order for the forward and the backward (R2);
* ``wgrad``   per-parameter per-micro-batch weight-gradient equality and a fp32 re-accumulation in the DP1 and
  DP2 orders compared with the real step gradient (R2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load(path: Path) -> Any:
    import torch

    return torch.load(path, weights_only=False)


# ----------------------------------------------------------------------------- states
def flat(state: dict, key: str):
    import torch

    return torch.cat([state["optimizer"][n]["tensors"][key].double().reshape(-1) for n in sorted(state["optimizer"])])


def rel(a, b) -> float:
    return float((a - b).norm() / a.norm()) if float(a.norm()) else float((a - b).norm())


def state_metrics(before_ref: dict, after_ref: dict, before_new: dict, after_new: dict) -> dict[str, Any]:
    """Same definitions as compare.numeric (update = master after - master before) plus the step gradient."""
    import torch

    da = flat(after_ref, "param") - flat(before_ref, "param")
    db = flat(after_new, "param") - flat(before_new, "param")
    mask = da.abs() > 1e-12
    agree = float((torch.sign(da[mask]) == torch.sign(db[mask])).double().mean()) if int(mask.sum()) else 1.0
    m0a, m0b = flat(before_ref, "exp_avg"), flat(before_new, "exp_avg")
    ga = (flat(after_ref, "exp_avg") - 0.9 * m0a) / 0.1
    gb = (flat(after_new, "exp_avg") - 0.9 * m0b) / 0.1
    return {"update_rel_l2": rel(da, db), "sign_agreement": agree, "grad_rel_l2": rel(ga, gb),
            "exp_avg_rel_l2": rel(flat(after_ref, "exp_avg"), flat(after_new, "exp_avg")),
            "exp_avg_sq_rel_l2": rel(flat(after_ref, "exp_avg_sq"), flat(after_new, "exp_avg_sq"))}


def bitwise_equal(a: dict, b: dict) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {"equal": True, "differing": {}}
    for n in sorted(a["optimizer"]):
        for k in ("param", "exp_avg", "exp_avg_sq"):
            x, y = a["optimizer"][n]["tensors"][k], b["optimizer"][n]["tensors"][k]
            if not torch.equal(x, y):
                out["equal"] = False
                out["differing"].setdefault(k, 0)
                out["differing"][k] += int((x != y).sum())
    if a.get("adapter") is not None and b.get("adapter") is not None:
        bad = [n for n in a["adapter"] if not torch.equal(a["adapter"][n], b["adapter"][n])]
        if bad:
            out["equal"] = False
            out["differing"]["adapter_tensors"] = len(bad)
    return out


# ----------------------------------------------------------------------------- trace
def sample_order(events: list[dict], step: int = 3) -> dict[int, list[int]]:
    """rank (position in the probe list) -> sample ids in micro-batch (loss call) order for ``step``."""
    t = next(e for e in events if e["kind"] == "train" and e["step"] == step)
    out = {}
    for r, rank in enumerate(t["probe"]):
        out[r] = [i for rec in rank if rec["kind"] == "loss" for i in rec["sample_indices"]]
    return out


def collect(traces: dict[int, dict], order: dict[int, list[int]]) -> dict[int, tuple[dict, int]]:
    """sample id -> (trace payload of its rank, local micro-batch index)."""
    out = {}
    for r, ids in order.items():
        for k, sid in enumerate(ids):
            out[sid] = (traces[r], k)
    return out


def _records(payload: dict, mb: int) -> dict[str, dict]:
    prefix = f"{mb}|"
    return {k[len(prefix):]: v for k, v in payload["records"].items() if k.startswith(prefix)}


def first_difference(ra: dict[str, dict], rb: dict[str, dict], kinds: tuple[str, ...]) -> dict[str, Any]:
    """First key (in ra's insertion = execution order) of the given kinds whose bits differ."""
    n_equal = n_diff = 0
    first = None
    for key, va in ra.items():
        if key.split("|")[1] not in kinds:
            continue
        vb = rb.get(key)
        if vb is None:
            n_diff += 1
            first = first or {"key": key, "reason": "missing in B"}
        elif va["bits"] != vb["bits"] or va["shape"] != vb["shape"]:
            n_diff += 1
            first = first or {"key": key, "rel_norm_diff": abs(va["l2"] - vb["l2"]) / max(va["l2"], 1e-300),
                              "shape": [list(va["shape"]), list(vb["shape"])]}
        else:
            n_equal += 1
    return {"equal": n_equal, "differ": n_diff, "first": first}


def trace_report(a: dict[int, tuple[dict, int]], b: dict[int, tuple[dict, int]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sid in sorted(set(a) & set(b)):
        (pa, ka), (pb, kb) = a[sid], b[sid]
        ra, rb = _records(pa, ka), _records(pb, kb)
        out[str(sid)] = {kind: first_difference(ra, rb, (kind,)) for kind in ("in", "fwd", "bwd_out", "bwd_in")}
    return out


# ----------------------------------------------------------------------------- wgrad
def wgrad_report(a: dict[int, tuple[dict, int]], b: dict[int, tuple[dict, int]], scale_a: float = 1.0,
                 scale_b: float = 1.0) -> dict[str, Any]:
    """Bit comparison of the per-micro-batch wgrads after undoing each arm's loss scale (DP2 grads are exactly 2x)."""
    import torch

    out: dict[str, Any] = {"per_sample": {}}
    for sid in sorted(set(a) & set(b)):
        (pa, ka), (pb, kb) = a[sid], b[sid]
        ga, gb = pa["wgrad"].get(ka, {}), pb["wgrad"].get(kb, {})
        differ = []
        for n in ga:
            x, y = ga[n].float() * scale_a, gb[n].float() * scale_b
            if not torch.equal(x, y):
                differ.append((n, float((x - y).norm() / max(float(x.norm()), 1e-300))))
        out["per_sample"][str(sid)] = {"params": len(ga), "differ": len(differ),
                                       "worst": sorted(differ, key=lambda t: -t[1])[:5]}
    return out


def reaccumulate(a_dp1: dict[int, tuple[dict, int]], order1: list[int], order2: dict[int, list[int]],
                 b_dp2: dict[int, tuple[dict, int]]) -> dict[str, Any]:
    """fp32 sums of the DP1 micro-batch wgrads in DP1 order and in the DP2 order (rank sums, then x0.5 and add)
    compared with each other: the size of the pure summation-order effect."""
    import torch

    names = sorted(next(iter(a_dp1.values()))[0]["wgrad"][0])
    num = den = 0.0
    worst = 0.0
    for n in names:
        seq = [a_dp1[s][0]["wgrad"][a_dp1[s][1]][n].float() for s in order1]
        s1 = torch.zeros_like(seq[0])
        for g in seq:
            s1 = s1 + g
        by = {s: g for s, g in zip(order1, seq)}
        ranks = []
        for r in sorted(order2):
            acc = torch.zeros_like(seq[0])
            for s in order2[r]:
                acc = acc + by[s]
            ranks.append(acc)
        s2 = ranks[0]
        for x in ranks[1:]:
            s2 = s2 + x
        d = float((s1.double() - s2.double()).norm())
        num += d * d
        den += float(s1.double().norm()) ** 2
        worst = max(worst, d / max(float(s1.double().norm()), 1e-300))
    return {"total_rel_l2": (num / den) ** 0.5 if den else 0.0, "worst_param_rel_l2": worst}


# ----------------------------------------------------------------------------- driver
def read_events(arm_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (arm_dir / "events.jsonl").read_text().splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    run = Path(argv[0])
    work = run / "work" if (run / "work").is_dir() else run
    packed = work / "packed"
    present = sorted({p.name.split("_")[0] for p in packed.glob("rc*_s3.pt")})
    st = lambda arm, tag: _load(packed / f"{arm}_{tag}.pt")  # noqa: E731
    res: dict[str, Any] = {"arms": present, "states": {}, "trace": {}, "wgrad": {}}
    c1 = st("rcA1", "s2")
    s3 = {arm: st(arm, "s3") for arm in present}
    restored = {arm: st(arm, "restored") for arm in present if (packed / f"{arm}_restored.pt").is_file()}
    res["states"]["restored_equals_C1"] = {a: bitwise_equal(c1, s) for a, s in restored.items()}
    pairs = [("rcBc", "rcSb"), ("rcDd", "rcSa"), ("rcSa", "rcA1"), ("rcSb", "rcSa"), ("rcBc", "rcA1"), ("rcBc", "rcDd"),
             ("rcA2", "rcA1")]
    for x, y in pairs:
        if x in s3 and y in s3:
            res["states"][f"{x}_vs_{y}"] = {"bitwise": bitwise_equal(s3[x], s3[y]),
                                            "metrics": state_metrics(c1, s3[y], c1, s3[x])}
    if "rcA2" in s3 and (packed / "rcA2_s2.pt").is_file():  # no restore anywhere: from-scratch DP1 vs DP2, step 2
        a2s2 = st("rcA2", "s2")
        res["states"]["rcA2_s2_vs_rcA1_s2"] = {"bitwise": bitwise_equal(a2s2, c1)}
        res["states"]["rcA2_s2_vs_rcA1_s2"]["exp_avg_rel_l2"] = rel(flat(c1, "exp_avg"), flat(a2s2, "exp_avg"))
        res["states"]["rcA2_s2_vs_rcA1_s2"]["exp_avg_sq_rel_l2"] = rel(flat(c1, "exp_avg_sq"), flat(a2s2, "exp_avg_sq"))
    raw_scale = False
    traces: dict[str, dict[str, dict[int, dict]]] = {}
    for arm in present:
        traces[arm] = {}
        for f in sorted(packed.glob(f"{arm}_trace_s*_dp*.pt")):
            tag, dp = f.stem.split("_trace_")[1].rsplit("_dp", 1)
            payload = _load(f)
            raw_scale = raw_scale or "bwd_scale" not in payload
            traces[arm].setdefault(tag, {})[int(dp)] = payload
    res["gradient_scale_stored_raw"] = raw_scale
    ev = {arm: read_events(work / "arms" / arm) for arm in present}
    cmp_pairs = [("rcSa", "rcSb", "s3"), ("rcA1", "rcSa", "s3"), ("rcBc", "rcSb", "s3"), ("rcDd", "rcSa", "s3"),
                 ("rcA1", "rcA2", "s1"), ("rcA1", "rcA2", "s2"), ("rcA1", "rcA2", "s3")]
    for x, y, tag in cmp_pairs:
        if x not in traces or y not in traces or tag not in traces[x] or tag not in traces[y]:
            continue
        step = int(tag[1:])
        cx = collect(traces[x][tag], sample_order(ev[x], step))
        cy = collect(traces[y][tag], sample_order(ev[y], step))
        name = f"{x}_vs_{y}_{tag}"
        res["trace"][name] = trace_report(cx, cy)
        if tag == "s3" and any(traces[x][tag][d]["wgrad"] for d in traces[x][tag]):
            unit = {"rcSb": 0.5, "rcBc": 0.5, "rcA2": 0.5} if raw_scale else {}
            res["wgrad"][name] = wgrad_report(cx, cy, unit.get(x, 1.0), unit.get(y, 1.0))
    if "rcSa" in traces and "rcSb" in traces and traces["rcSa"].get("s3") and traces["rcSb"].get("s3"):
        a = collect(traces["rcSa"]["s3"], sample_order(ev["rcSa"], 3))
        b = collect(traces["rcSb"]["s3"], sample_order(ev["rcSb"], 3))
        res["wgrad"]["reaccumulate_DP1_vs_DP2_order"] = reaccumulate(a, sample_order(ev["rcSa"], 3)[0],
                                                                      sample_order(ev["rcSb"], 3), b)
    text = json.dumps(res, indent=1, sort_keys=True, default=repr)
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
