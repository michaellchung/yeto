"""Summarise kernel_probe worker JSONs: for each variant, are the per-sample hashes equal across ranks and cold-compile repeats?

    python kernel_probe_compare.py <dir with *.json>   -> prints a table and writes RESULT_kprobe.json
"""
import glob
import json
import os
import sys

FIELDS = ("loss", "logp", "grad_output", "logit_grad")


def load(d):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "*.json"))) if not p.endswith("RESULT_kprobe.json")]
    return [r for r in runs if "samples" in r]


def summarize(runs):
    out = {}
    for v in sorted({r["variant"] for r in runs}):
        rs = [r for r in runs if r["variant"] == v]
        ref = rs[0]
        rep = {"workers": [r["tag"] for r in rs], "all_equal": True, "per_field": {}, "differing": []}
        for f in FIELDS:
            bad = []
            for r in rs[1:]:
                for i, s in ref["samples"].items():
                    if r["samples"][i][f] != s[f]:
                        bad.append((r["tag"], int(i)))
            rep["per_field"][f] = {"n_diff": len(bad), "first": bad[:3]}
            if bad:
                rep["all_equal"] = False
        # which workers differ from the first one (any field), to separate ranks from repeats
        rep["differing"] = sorted({t for f in FIELDS for t, _ in rep["per_field"][f]["first"]})
        cfgs = {}
        for r in rs:
            inv = r.get("inventory", {})
            cfgs[r["tag"]] = {"n_cubin": len(inv.get("cubin", {})),
                              "best_config": {k: x for k, x in sorted(inv.get("best_config", {}).items())},
                              "cubin": {k: (x.get("sha"), x.get("num_warps")) for k, x in sorted(inv.get("cubin", {}).items())}}
        rep["cfg_identical_across_workers"] = all(json.dumps(c, sort_keys=True) == json.dumps(cfgs[rs[0]["tag"]], sort_keys=True)
                                                   for c in cfgs.values())
        rep["configs"] = cfgs
        out[v] = rep
    # compiled variants vs eager: do they agree at all (informative only)
    return out


def main(d):
    res = summarize(load(d))
    for v, r in res.items():
        print(v, "workers", len(r["workers"]), "bitwise-equal across workers:", r["all_equal"],
              "| triton cubin/config sets identical:", r["cfg_identical_across_workers"],
              {f: x["n_diff"] for f, x in r["per_field"].items()})
    json.dump(res, open(os.path.join(d, "RESULT_kprobe.json"), "w"), indent=1, sort_keys=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1])
