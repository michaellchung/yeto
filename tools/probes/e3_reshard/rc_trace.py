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


def _put(key: str, t: Any) -> None:
    bits, norm = _checksum_bits(t)
    _ST["pending"].append((key, bits, norm, tuple(t.shape), str(t.dtype)))


def _flush() -> dict[str, Any]:
    out = {}
    for key, bits, norm, shape, dtype in _ST["pending"]:
        out[key] = {"bits": [int(b) for b in bits.cpu().tolist()], "l2": float(norm.cpu()), "shape": shape,
                    "dtype": dtype}
    _ST["pending"].clear()
    return out


def install_trace(actor: Any, *, save_mb: tuple = (0, 1), name_filter: str = r"decoder\.layers\.\d+$|output_layer$|final_layernorm$|embedding$",
                  save_wgrads: bool = True) -> dict[str, Any]:
    """Register the hooks on every model chunk (idempotent per process)."""
    import torch

    if _ST.get("handles"):
        return {"installed": False, "reason": "already installed"}
    _ST.update({"pending": [], "mb": -1, "save_mb": set(int(i) for i in save_mb), "tensors": {}, "wgrad": {},
                "filter": re.compile(name_filter), "handles": [], "save_wgrads": bool(save_wgrads), "names": []})
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
                            _ST["tensors"][f"{mb}|{name}|bwd_out|{i}"] = g.detach().to("cpu", copy=True)
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
                        _ST["wgrad"].setdefault(_ST["mb"], {})[pname] = g.detach().to("cpu", copy=True)
                    handles.append(p.register_hook(on_wgrad))
    return {"installed": True, "modules": mod_count}


def dump_trace(actor: Any, *, directory: str, tag: str, coord: dict | None = None, save: bool = True) -> dict[str, Any]:
    """Write this rank's trace and clear the buffers (the hooks stay installed); ``save=False`` only clears."""
    import torch

    if not save:
        _ST["pending"].clear()
        _ST["tensors"].clear()
        _ST["wgrad"].clear()
        _ST["mb"] = -1
        return {"saved": False}

    if coord is None:
        from yeto.rl.engine.miles_adapter.cut_plugin import _backend

        coord = dict(_backend(actor).coord())
    rec = _flush()
    payload = {"coord": dict(coord), "records": rec, "tensors": dict(_ST["tensors"]), "wgrad": dict(_ST["wgrad"]),
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


def reset_for_tests() -> None:
    for h in _ST.get("handles", []):
        h.remove()
    _ST.clear()
