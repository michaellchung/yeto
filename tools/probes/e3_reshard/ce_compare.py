"""RC-7 offline analysis (a8-rootcause plan.md section 12): compare the fused-CE probe records of several hosts.

    python ce_compare.py <label>=<packed dir>[@arm] [...] [--out result.json]

<packed dir> holds gA2e_trace_s2_dp0.pt / dp1.pt (and gA2e_s2.pt). For every pair of hosts, every rank and every
call, the stages are compared in pipeline order (first fields are the earliest) and the FIRST differing stage is
reported; in-process recompute flags (``compiled_equal``/``eager_equal``) are listed per host. Saved full tensors
(``big``) are compared elementwise when both hosts saved the same key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FWD_STAGES = ("logits_in", "logits_max", "logits_shifted_in", "target", "target_mask", "masked_target_1d",
              "pred_sumexp", "exp_logits", "softmax", "loss")
BWD_STAGES = ("softmax_in", "grad_output", "target_mask_in", "masked_target_in", "out")


def load_trace(d: Path, dp: int, arm: str = "gA2e"):
    import torch

    p = d / f"{arm}_trace_s2_dp{dp}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def split(ce):
    fwd = [c for c in ce if "seq" in c]
    bwd = [c["bwd"] for c in ce if "bwd" in c]
    return fwd, bwd


def same(a, b, key):
    x, y = a.get(key), b.get(key)
    if x is None or y is None:
        return None
    return x["bits"] == y["bits"] and x["shape"] == y["shape"]


def first_diff(a, b, stages):
    for s in stages:
        r = same(a, b, s)
        if r is False:
            return s
    return None


def compare_pair(ta, tb):
    out = {}
    for dp in (0, 1):
        pa, pb = ta.get(dp), tb.get(dp)
        if pa is None or pb is None:
            continue
        fa, ba = split(pa["ce"])
        fb, bb = split(pb["ce"])
        rows = {"n_fwd": [len(fa), len(fb)], "n_bwd": [len(ba), len(bb)], "fwd_first_diff": {}, "bwd_first_diff": {}}
        for i, (x, y) in enumerate(zip(fa, fb)):
            d = first_diff(x, y, FWD_STAGES)
            if d:
                rows["fwd_first_diff"][i] = d
        for i, (x, y) in enumerate(zip(ba, bb)):
            d = first_diff(x, y, BWD_STAGES)
            if d:
                rows["bwd_first_diff"][i] = {"stage": d, "sample_ids": x.get("sample_ids"),
                                             "nonzero": [x.get("grad_output_nonzero"), y.get("grad_output_nonzero")]}
        sa, sb = pa["small"], pb["small"]
        rows["logit_grad_row_bits_diff_mbs"] = [mb for mb in sa if mb in sb and "logit_grad_row_bits" in sa[mb]
                                                and not (sa[mb]["logit_grad_row_bits"] == sb[mb]["logit_grad_row_bits"]).all()]
        rows["logits_row_bits_diff_mbs"] = [mb for mb in sa if mb in sb and "logits_row_bits" in sa[mb]
                                            and not (sa[mb]["logits_row_bits"] == sb[mb]["logits_row_bits"]).all()]
        big = {}
        for k in sorted(set(pa["big"]) & set(pb["big"])):
            x, y = pa["big"][k], pb["big"][k]
            if x.shape != y.shape:
                big[k] = "shape differs"
                continue
            if x.dtype.is_floating_point:
                ne = (x.view(-1).view({2: __import__("torch").int16, 4: __import__("torch").int32}[x.element_size()])
                      != y.view(-1).view({2: __import__("torch").int16, 4: __import__("torch").int32}[y.element_size()]))
            else:
                ne = x != y
            big[k] = {"differing_elements": int(ne.sum()), "numel": int(x.numel())}
        rows["big"] = big
        out[f"dp{dp}"] = rows
    return out


def host_summary(t):
    s = {}
    for dp, p in t.items():
        fwd, bwd = split(p["ce"])
        errs = [c for c in p["ce"] if "errors" in c]
        s[f"dp{dp}"] = {
            "n_fwd": len(fwd), "n_bwd": len(bwd), "ptr_matched_bwd": sum(1 for b in bwd if b.get("fwd_ptr_match")),
            "nonzero_bwd": sum(1 for b in bwd if b.get("grad_output_nonzero")),
            "compiled_recompute_equal": [b.get("compiled_equal") for b in bwd],
            "eager_recompute_equal": [b.get("eager_equal") for b in bwd],
            "eager_vs_out_max_abs": [b.get("eager_vs_out_max_abs") for b in bwd],
            "probe_errors": [e["errors"] for e in errs if e["errors"]],
            "device": {k: p["device"].get(k) for k in ("gpu", "uuid", "sm_count", "capability", "torch", "cuda")}}
    return s


def main(argv):
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    hosts = {}
    for a in argv:
        label, path = a.split("=", 1)
        path, _, arm = path.partition("@")  # <dir>[@arm], default arm gA2e
        hosts[label] = {dp: load_trace(Path(path), dp, arm or "gA2e") for dp in (0, 1)}
        hosts[label] = {dp: t for dp, t in hosts[label].items() if t is not None}
    res = {"hosts": {k: host_summary(v) for k, v in hosts.items()}, "pairs": {}}
    labels = list(hosts)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            res["pairs"][f"{a}__{b}"] = compare_pair(hosts[a], hosts[b])
    text = json.dumps(res, indent=1, default=str)
    if out:
        Path(out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
