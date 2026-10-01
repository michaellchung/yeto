"""A8 root-cause trace plugin (rl-infra-spec 4.6, evidence/infra-e3/a8-rootcause/plan.md).

Read-only probe loaded through ``TrainGroup.run_plugin`` (``rc_trace.install_trace`` /
``rc_trace.dump_trace``; the container puts this directory on PYTHONPATH). It changes
no computation: forward hooks and tensor hooks only observe.

Per TRAINING micro batch (forward passes with grad enabled; the forward-only log-prob passes
run under ``no_grad`` and are skipped) it records, for every submodule of the model chunk:

* ``fwd``   a bit checksum + L2 norm of every output tensor,
* ``bwd_out`` the same for the gradient arriving at that output (the backward of the consumers),
* ``bwd_in``  the same for the gradient produced for each input tensor (the module's own backward),

plus, for every trainable parameter, the per-micro-batch weight gradient (bf16 as autograd
produced it, before the DDP hook adds it to ``main_grad``). Full tensors of the layer-level
outputs/gradients are kept for the micro batches in ``save_mb``.

The checksum is two int64 sums over the raw bits (plain and index-weighted; wrap-around is
fine, it only has to be identical for identical bits) so that "bit-identical" is decidable
without copying every tensor to the host.
"""

from __future__ import annotations

import os
import re
from typing import Any

_MODULE = "rc_trace"
INSTALL_TRACE = f"{_MODULE}.install_trace"
DUMP_TRACE = f"{_MODULE}.dump_trace"

_ST: dict[str, Any] = {}


def _checksum_bits(t: Any) -> Any:
    """Exact variant: int64 sums returned as an int64 tensor [sum, weighted] and a float64 norm."""
    import torch

    x = t.detach().contiguous()
    if x.dtype in (torch.bfloat16, torch.float16):
        v = x.view(torch.int16).to(torch.int64)
    elif x.dtype == torch.float32:
        v = x.view(torch.int32).to(torch.int64)
    elif x.dtype == torch.float64:
        v = x.view(torch.int64)
    else:
        v = x.to(torch.int64)
    v = v.reshape(-1)
    w = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 1000003) + 1
    norm = x.double().norm() if x.is_floating_point() else torch.zeros((), dtype=torch.float64, device=v.device)
    return torch.stack([v.sum(), (v * w).sum()]), norm


def _tensors(obj: Any) -> list[Any]:
    import torch

    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in _tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _tensors(o)]
    return []


def _unscale(key: str, t: Any) -> Any:
    """Undo the arm's loss scale on gradients: DP2 divides the loss by 8 instead of 16 (exact factor 2), so a
    DP2 gradient times 0.5 is bit-comparable with the DP1 one (``bwd_scale`` = 1/dp, a power of two)."""
    scale = _ST.get("bwd_scale", 1.0)
    if scale != 1.0 and ("|bwd_" in key) and t.is_floating_point():
        return t * scale
    return t


def _put(key: str, t: Any) -> None:
    """Record now, on the host: the trainer offloads its GPU memory after the step (run RC-1: a deferred
    ``.cpu()`` on checksums allocated during the step failed with CUDA 'invalid argument' at dump time)."""
    bits, norm = _checksum_bits(_unscale(key, t))
    _ST["records"][key] = {"bits": [int(b) for b in bits.cpu().tolist()], "l2": float(norm.cpu()),
                           "shape": tuple(t.shape), "dtype": str(t.dtype)}


def _flush() -> dict[str, Any]:
    out = dict(_ST["records"])
    _ST["records"].clear()
    return out


def install_trace(actor: Any, *, bwd_scale: float = 1.0, save_mb: tuple = (0, 1), name_filter: str = r"decoder\.layers\.\d+$|output_layer$|final_layernorm$|embedding$",
                  save_wgrads: bool = True) -> dict[str, Any]:
    """Register the hooks on every model chunk (idempotent per process)."""
    import torch

    if _ST.get("handles"):
        return {"installed": False, "reason": "already installed"}
    _ST.update({"records": {}, "mb": -1, "save_mb": set(int(i) for i in save_mb), "tensors": {}, "wgrad": {},
                "filter": re.compile(name_filter), "handles": [], "bwd_scale": float(bwd_scale), "save_wgrads": bool(save_wgrads), "names": []})
    handles = _ST["handles"]
    mod_count = 0

    def keep(mb: int, name: str) -> bool:
        return mb in _ST["save_mb"] and _ST["filter"].search(name) is not None

    def make_pre_root():
        def pre(mod, args, kwargs=None):
            if not torch.is_grad_enabled():
                return
            _ST["mb"] += 1
            mb = _ST["mb"]
            for i, t in enumerate(_tensors(list(args)) + _tensors(kwargs or {})):
                if t.numel():
                    _put(f"{mb}|<root-input>|in|{i}", t)
        return pre

    def make_fwd(name: str):
        def pre(mod, args):
            if not torch.is_grad_enabled() or _ST["mb"] < 0:
                return
            mb = _ST["mb"]
            for i, t in enumerate(_tensors(list(args))):
                if t.requires_grad and t.is_floating_point():
                    t.register_hook(lambda g, mb=mb, name=name, i=i: _put(f"{mb}|{name}|bwd_in|{i}", g))

        def hook(mod, args, out):
            if not torch.is_grad_enabled() or _ST["mb"] < 0:
                return
            mb = _ST["mb"]
            for i, t in enumerate(_tensors(out)):
                _put(f"{mb}|{name}|fwd|{i}", t)
                if keep(mb, name):
                    _ST["tensors"][f"{mb}|{name}|fwd|{i}"] = t.detach().to("cpu", copy=True)
                if t.requires_grad and t.is_floating_point():
                    def on_grad(g, mb=mb, name=name, i=i):
                        _put(f"{mb}|{name}|bwd_out|{i}", g)
                        if keep(mb, name):
                            _ST["tensors"][f"{mb}|{name}|bwd_out|{i}"] = _unscale("|bwd_", g).detach().to("cpu", copy=True)
                    t.register_hook(on_grad)
        return pre, hook

    chunks = list(getattr(actor, "model", None) or [])
    if not chunks:
        raise RuntimeError("actor has no model chunks")
    for ci, chunk in enumerate(chunks):
        handles.append(chunk.register_forward_pre_hook(make_pre_root(), with_kwargs=True))
        for name, m in chunk.named_modules():
            if not name:
                continue
            pre, hook = make_fwd(f"c{ci}.{name}")
            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(hook))
            mod_count += 1
            _ST["names"].append(f"c{ci}.{name}")
        if _ST["save_wgrads"]:
            for pname, p in chunk.named_parameters():
                if p.requires_grad:
                    def on_wgrad(g, pname=f"c{ci}.{pname}"):
                        _ST["wgrad"].setdefault(_ST["mb"], {})[pname] = _unscale("|bwd_", g).detach().to("cpu", copy=True)
                    handles.append(p.register_hook(on_wgrad))
    return {"installed": True, "modules": mod_count}


def dump_trace(actor: Any, *, directory: str, tag: str, coord: dict | None = None, save: bool = True,
               heavy: bool = True) -> dict[str, Any]:
    """Write this rank's trace and clear the buffers (the hooks stay installed); ``save=False`` only clears."""
    import torch

    if not save:
        _ST["records"].clear()
        _ST["tensors"].clear()
        _ST["wgrad"].clear()
        _ST["mb"] = -1
        return {"saved": False}

    if coord is None:
        from yeto.rl.engine.miles_adapter.cut_plugin import _backend

        coord = dict(_backend(actor).coord())
    rec = _flush()
    payload = {"coord": dict(coord), "records": rec, "tensors": dict(_ST["tensors"]) if heavy else {},
               "wgrad": dict(_ST["wgrad"]) if heavy else {}, "bwd_scale": _ST.get("bwd_scale", 1.0),
               "device": _device_info(),
               "n_microbatches": _ST["mb"] + 1, "env": {k: os.environ.get(k) for k in (
                   "CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO", "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                   "NVIDIA_TF32_OVERRIDE", "NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")}}
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"trace_{tag}_dp{coord.get('dp', 0)}.pt")
    torch.save(payload, path)
    _ST["tensors"].clear()
    _ST["wgrad"].clear()
    _ST["mb"] = -1
    return {"path": path, "records": len(rec), "microbatches": payload["n_microbatches"], "coord": dict(coord)}


def _device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                            "bf16_reduced_precision_reduction":
                                bool(getattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", None))}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        info.update({"gpu": props.name, "sm_count": props.multi_processor_count,
                     "capability": list(torch.cuda.get_device_capability())})
    try:
        import transformer_engine as te

        info["transformer_engine"] = getattr(te, "__version__", "?")
    except Exception:  # noqa: BLE001 - absent on CPU
        pass
    return info


def reset_for_tests() -> None:
    for h in _ST.get("handles", []):
        h.remove()
    _ST.clear()
