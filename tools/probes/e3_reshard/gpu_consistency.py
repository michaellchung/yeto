"""GPU0-vs-GPU1 and run-to-run consistency of the ops the training step uses (a8-rootcause RC-5; container, before Ray).

    python gpu_consistency.py <out dir>

Identical inputs (seeded on the CPU) go to each device; every op runs 3 times per device; the outputs are compared
bitwise between devices and between repeats. Includes Megatron's torch.compile'd fused cross-entropy pieces (the
op whose backward produces the logits gradient, the first place where the DP2 rank-1 difference appeared).
Also writes nvidia-smi / lscpu host information.
"""
import json
import os
import subprocess
import sys

import torch

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for name, cmd in (("nvidia_smi_q.txt", "nvidia-smi -q"), ("nvidia_smi_topo.txt", "nvidia-smi topo -m"),
                  ("lscpu.txt", "lscpu"), ("nvidia_smi_L.txt", "nvidia-smi -L")):
    with open(os.path.join(out, name), "w") as fh:
        subprocess.run(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT, check=False)

n = torch.cuda.device_count()
res = {"devices": [torch.cuda.get_device_properties(i).name for i in range(n)], "ops": {}}
g = torch.Generator().manual_seed(0)
T, H, V, F = 384, 1024, 151936, 6144


def r(*shape, dtype=torch.bfloat16, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).to(dtype)


x, w_head, gl = r(T, H), r(V, H, scale=0.05), r(T, V, scale=1e-3)
w_up, lora_a, lora_b = r(H, F, scale=0.05), r(H, 16, scale=0.05), r(16, F, scale=0.05)
q, k, v = r(16, T, 128), r(16, T, 128), r(16, T, 128)
logits = r(T, 1, V, scale=3.0)
target = torch.randint(0, V, (T, 1), generator=g)
gout = (torch.randn(T, 1, generator=g) * 1e-2).float()
big = torch.randn(64_000_000, generator=g)


def ce(lg, tg, go, dev):
    from megatron.core.fusions import fused_cross_entropy as fce

    lg = lg.detach()
    lg, m = fce.calculate_logits_max(lg)
    tm, mt, packed, expl = fce.calculate_predicted_logits(lg, tg, m, 0, V)
    expl, loss = fce.calculate_cross_entropy_loss(expl, packed)
    grad = fce.calculate_gradients(expl, go, tm, mt)
    return loss, grad


ops = {
    "lm_head_fwd": lambda d: torch.nn.functional.linear(x.to(d), w_head.to(d)),
    "lm_head_dgrad_K151936": lambda d: gl.to(d) @ w_head.to(d),
    "lm_head_wgrad": lambda d: gl.to(d).t() @ x.to(d),
    "mlp_up_fwd": lambda d: x.to(d) @ w_up.to(d),
    "lora_in_out": lambda d: (x.to(d) @ lora_a.to(d)) @ lora_b.to(d),
    "bmm_qk": lambda d: torch.bmm(q.to(d), k.to(d).transpose(1, 2)),
    "softmax_fp32_attn": lambda d: torch.softmax(torch.bmm(q.to(d), k.to(d).transpose(1, 2)).float(), -1),
    "softmax_vocab_fp32": lambda d: torch.softmax(logits.to(d).float(), -1),
    "log_softmax_vocab_fp32": lambda d: torch.log_softmax(logits.to(d).float(), -1),
    "sum_64M_fp32": lambda d: big.to(d).sum().reshape(1),
    "fused_ce_compiled_fwd_loss": lambda d: ce(logits.to(d), target.to(d), gout.to(d), d)[0],
    "fused_ce_compiled_bwd_grad": lambda d: ce(logits.to(d), target.to(d), gout.to(d), d)[1],
}
for name, fn in ops.items():
    try:
        outs = {}
        for i in range(n):
            d = torch.device("cuda", i)
            torch.cuda.set_device(i)
            runs = []
            for _ in range(3):
                y = fn(d)
                torch.cuda.synchronize(d)
                runs.append(y.cpu())
            outs[i] = runs
        res["ops"][name] = {"repeat_equal": {i: all(torch.equal(outs[i][0], t) for t in outs[i][1:]) for i in outs},
                            "device_equal": {f"0_vs_{i}": torch.equal(outs[0][0], outs[i][0]) for i in range(1, n)}}
        if n > 1 and not res["ops"][name]["device_equal"]["0_vs_1"]:
            d = (outs[0][0].double() - outs[1][0].double())
            res["ops"][name]["elements_differ"] = int((d != 0).sum())
            res["ops"][name]["max_abs_diff"] = float(d.abs().max())
    except Exception as exc:  # noqa: BLE001
        res["ops"][name] = {"error": f"{type(exc).__name__}: {exc}"}
res["any_device_difference"] = any(not all(v.get("device_equal", {}).values()) for v in res["ops"].values() if "device_equal" in v)
res["any_repeat_difference"] = any(not all(v.get("repeat_equal", {}).values()) for v in res["ops"].values() if "repeat_equal" in v)
json.dump(res, open(os.path.join(out, "gpu_consistency.json"), "w"), indent=1)
print("BATTERY", json.dumps({k: res[k] for k in ("devices", "any_device_difference", "any_repeat_difference")}), flush=True)
