"""Kernel-selection probe for the A8 G4 root cause (a8-rootcause RC-6; runs INSIDE the Miles image, one GPU per process).

    python kernel_probe.py worker <variant> <tag> <rot> <out.json>     # one process = one rank, pinned by CUDA_VISIBLE_DEVICES
    python kernel_probe.py driver <outdir>                              # orchestrates the phases (see PHASES)

Chain per sample (what a training microbatch does with the logits): bf16 logits -> fp32 copy -> Megatron
fused_vocab_parallel_cross_entropy (jit_fuser = torch.compile) -> Miles compute_policy_loss (torch.compile dynamic) ->
backward. Per sample the worker records sha256 of loss, log-probs, the gradient arriving at the log-probs (grad_output of
the fused CE) and the bf16 gradient of the logits. After the run it hashes every Triton artifact in the compile caches
(kernel name, num_warps, num_stages, cubin sha256, and the Inductor .best_config files = the autotune choices).
Sample shapes follow the real data (token counts of the A8 microbatches); `rot` rotates which shape a process sees first
(torch.compile specialises on the first shape, then recompiles dynamic), like DP2 rank0/rank1 vs DP1.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

TS = [384, 347, 291, 402, 256, 331, 365, 318]  # token counts per sample (fixed for the probe; not tuned afterwards)
V = 151936
VARIANTS = ("a", "b", "c1", "c2")  # a: default; b: all torch.compile off; c1: autotune off; c2: pinned (pre-populated) cache


def sha(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:20]


def cache_inventory(dirs):
    inv = {"best_config": {}, "cubin": {}, "triton_meta": {}}
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in sorted(files):
                p = os.path.join(root, f)
                try:
                    if f.endswith(".best_config"):
                        inv["best_config"][f] = open(p).read()[:400]
                    elif f.endswith(".cubin"):
                        meta = {}
                        jp = p[:-6] + ".json"
                        if os.path.exists(jp):
                            m = json.load(open(jp))
                            meta = {k: m.get(k) for k in ("name", "num_warps", "num_stages", "shared", "num_ctas")}
                        inv["cubin"][f] = {"sha": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], **meta}
                except Exception as exc:  # noqa: BLE001
                    inv.setdefault("errors", []).append(f"{f}: {exc!r}")
    return inv


def worker(variant, tag, rot, out):
    import torch
    t_start = time.time()
    if variant in ("c1", "c2"):
        import torch._inductor.config as ic
        ic.max_autotune = False
        ic.max_autotune_pointwise = False
        ic.coordinate_descent_tuning = False
        ic.triton.autotune_pointwise = False
    import torch.distributed as dist
    cpu = os.environ.get("KPROBE_CPU") == "1"  # local smoke test only
    dist.init_process_group("gloo" if cpu else "nccl", init_method=f"file:///tmp/pg_{tag}", world_size=1, rank=0)
    grp = dist.group.WORLD
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
    dev = torch.device("cpu" if cpu else "cuda", 0 if not cpu else None)
    props = torch.cuda.get_device_properties(0) if not cpu else type("P", (), {"name": "cpu", "multi_processor_count": 0})()
    import triton
    import torch._inductor.config as ic
    res = {"variant": variant, "tag": tag, "rot": rot, "gpu": props.name, "uuid": str(getattr(props, "uuid", "")),
           "torch": torch.__version__, "triton": triton.__version__, "sm_count": props.multi_processor_count,
           "inductor": {k: getattr(ic, k) for k in ("max_autotune", "max_autotune_pointwise", "coordinate_descent_tuning",
                                                      "fx_graph_cache", "freezing")},
           "autotune_pointwise": ic.triton.autotune_pointwise, "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
           "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"), "order": [], "samples": {}}
    order = list(range(len(TS)))
    order = order[rot:] + order[:rot]
    for i in order:
        T = TS[i]
        g = torch.Generator().manual_seed(1000 + i)
        x = (torch.randn(T, V, generator=g) * 2.5).to(torch.bfloat16)
        tok = torch.randint(0, V, (T,), generator=g)
        # a few confident positions, like a trained policy
        for j in range(0, T, 17):
            x[j, tok[j]] = 14.0
        adv = torch.randn(T, generator=g)
        adv[adv.abs() < 0.3] = 0.0
        old = (-torch.rand(T, generator=g) * 0.5 - 0.3).float()
        xd = x.to(dev).requires_grad_(True)
        logits = xd.float()
        lp = -fused_vocab_parallel_cross_entropy(logits.unsqueeze(1), tok.to(dev).unsqueeze(1), grp)
        lp = lp.squeeze(1)
        gout = {}
        lp.register_hook(lambda g_, gout=gout: gout.__setitem__("g", g_.detach().clone()))
        ppo_kl = old.to(dev) - lp
        loss, _ = compute_policy_loss(ppo_kl, adv.to(dev), 0.2, 0.2)
        loss.sum().backward()
        if not cpu:
            torch.cuda.synchronize()
        res["order"].append(i)
        res["samples"][str(i)] = {"T": T, "loss": sha(loss), "logp": sha(lp), "grad_output": sha(gout["g"]),
                                  "logit_grad": sha(xd.grad), "grad_abs_sum": float(xd.grad.double().abs().sum())}
    res["wall_s"] = round(time.time() - t_start, 1)
    dirs = [d for d in (os.environ.get("TORCHINDUCTOR_CACHE_DIR"), os.environ.get("TRITON_CACHE_DIR")) if d]
    res["inventory"] = cache_inventory(dirs)
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print("WORKER", variant, tag, "done", res["wall_s"], "s", flush=True)


# (variant, reps, how the processes share caches)
PHASES = [("a", 3, "shared"), ("b", 1, "shared"), ("c1", 2, "shared"), ("c2", 1, "pinned")]


def run_pair(variant, rep, outdir, cache, populate_only=False):
    """Start GPU0 (rot 0) and GPU1 (rot 1) concurrently with the given cache dir (cold unless 'pinned')."""
    procs = []
    gpus = (0,) if populate_only else (0, 1)
    for gpu in gpus:
        tag = f"{variant}_r{rep}_g{gpu}" + ("_pop" if populate_only else "")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TORCHINDUCTOR_CACHE_DIR=f"{cache}/ind", TRITON_CACHE_DIR=f"{cache}/tri",
                   PYTHONPATH="/root/miles:/sgl-workspace/sglang/python:/yeto:/yeto/tools/probes/e3_reshard:" + os.environ.get("PYTHONPATH", ""),
                   NCCL_ALGO="Ring", CUBLAS_WORKSPACE_CONFIG=":4096:8", NVIDIA_TF32_OVERRIDE="0", NVTE_ALLOW_NONDETERMINISTIC_ALGO="0")
        if variant == "b":
            env["TORCHDYNAMO_DISABLE"] = "1"
        if variant in ("c1", "c2"):
            env.update(TORCHINDUCTOR_MAX_AUTOTUNE="0", TORCHINDUCTOR_COORDINATE_DESCENT_TUNING="0", TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE="0")
        log = open(f"{outdir}/{tag}.log", "w")
        procs.append(subprocess.Popen([sys.executable, __file__, "worker", variant, tag, str(gpu), f"{outdir}/{tag}.json"],
                                      env=env, stdout=log, stderr=subprocess.STDOUT))
    rcs = [p.wait(timeout=900) for p in procs]
    print("PAIR", variant, rep, "rc", rcs, flush=True)
    return rcs


def driver(outdir):
    import shutil
    os.makedirs(outdir, exist_ok=True)
    for variant, reps, mode in PHASES:
        if mode == "pinned":
            cache = f"/tmp/cache_{variant}_pin"
            shutil.rmtree(cache, ignore_errors=True)
            run_pair(variant, 0, outdir, cache, populate_only=True)  # one process compiles + autotunes alone; both ranks then only read
        for rep in range(reps):
            cache = f"/tmp/cache_{variant}_{rep}" if mode == "shared" else f"/tmp/cache_{variant}_pin"
            if mode == "shared":
                shutil.rmtree(cache, ignore_errors=True)
            run_pair(variant, rep, outdir, cache)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        driver(sys.argv[2])
