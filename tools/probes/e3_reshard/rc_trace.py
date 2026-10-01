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


FO_FILTER = r"decoder\.layers\.\d+$|output_layer$|final_layernorm$|module\.module$"


def install_trace(actor: Any, *, bwd_scale: float = 1.0, profile_kernels: bool = False, save_mb: tuple = (0, 1),
                  name_filter: str = r"decoder\.layers\.\d+$|output_layer$|final_layernorm$|embedding$",
                  save_wgrads: bool = True, forward_only: bool = False, loss_level: bool = False,
                  logit_grad_mb: tuple = (), fo_filter: str = FO_FILTER,
                  module_hooks: bool = True, logit_grad_samples: tuple = (), ce_probe: bool = False,
                  ce_save: int = 1) -> dict[str, Any]:
    """Register the hooks on every model chunk (idempotent per process)."""
    import torch

    if _ST.get("handles"):
        return {"installed": False, "reason": "already installed"}
    _ST.update({"records": {}, "mb": -1, "save_mb": set(int(i) for i in save_mb), "tensors": {}, "wgrad": {},
                "filter": re.compile(name_filter), "handles": [], "bwd_scale": float(bwd_scale),
                "profile": bool(profile_kernels), "prof": None, "kernels": None, "save_wgrads": bool(save_wgrads), "names": [],
                "fo": -1, "module_hooks": bool(module_hooks), "forward_only": bool(forward_only), "fo_filter": re.compile(fo_filter), "small": {},
                "logit_grad_mb": set(int(i) for i in logit_grad_mb),
                "logit_grad_samples": set(int(i) for i in logit_grad_samples), "big": {}})
    handles = _ST["handles"]
    mod_count = 0

    def keep(mb: int, name: str) -> bool:
        return mb in _ST["save_mb"] and _ST["filter"].search(name) is not None

    def make_pre_root():
        def pre(mod, args, kwargs=None):
            if not torch.is_grad_enabled():
                if _ST["forward_only"]:  # the old-policy log-prob pass (no_grad): its own counter, outputs only
                    _ST["fo"] += 1
                    for i, t in enumerate(_tensors(list(args)) + _tensors(kwargs or {})):
                        if t.numel():
                            _put(f"fo{_ST['fo']}|<root-input>|in|{i}", t)
                return
            _ST["mb"] += 1
            mb = _ST["mb"]
            _profile_step(mb)
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
            if not torch.is_grad_enabled():
                if _ST["forward_only"] and _ST["fo"] >= 0 and _ST["fo_filter"].search(name):
                    for i, t in enumerate(_tensors(out)):
                        _put(f"fo{_ST['fo']}|{name}|fwd|{i}", t)
                return
            if _ST["mb"] < 0:
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
    if not module_hooks:  # loss-level probe only (RC-4 bisection): no hooks on the model, no root counter
        chunks = []
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
    if loss_level:
        _install_loss_level()
    info = {"installed": True, "modules": mod_count, "loss_level": bool(loss_level)}
    if ce_probe:
        info["ce_probe"] = _install_ce_probe(int(ce_save))
    return info


def _cs(t: Any) -> dict[str, Any]:
    """Host-side checksum record of one tensor (see ``_checksum_bits``); taken immediately, never deferred."""
    bits, norm = _checksum_bits(t)
    return {"bits": [int(b) for b in bits.cpu().tolist()], "l2": float(norm.cpu()), "shape": tuple(t.shape),
            "dtype": str(t.dtype)}


CE_NAMES = ("calculate_logits_max", "calculate_predicted_logits", "calculate_cross_entropy_loss", "calculate_gradients")


def _eager_gradients(softmax: Any, grad_output: Any, target_mask: Any, masked_target_1d: Any) -> Any:
    """The body of Megatron's ``fused_cross_entropy.calculate_gradients`` without ``torch.compile``."""
    import torch
    from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy as V

    grad_2d, arange_1d, softmax_update, grad_input = V.prepare_gradient_calculation_operands(softmax, target_mask)
    grad_input = V.calculate_gradients(grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output)
    return grad_input.to(torch.bfloat16)


def _install_ce_probe(save_n: int) -> dict[str, Any]:
    """Observe Megatron's fused cross entropy (a8-rootcause plan.md section 12).

    Every call of the four ``jit_fuser`` functions is recorded (input and output checksums). In the backward the
    inputs are cloned BEFORE the original call (it works in place on the softmax) and the same call is repeated on
    the clones: once through the same compiled function and once eagerly; both outputs are checksummed and compared
    with the original output. The wrapper always returns the original call's result. The first ``save_n`` backward
    calls with a non-zero ``grad_output`` of a step keep their full inputs and output (host copies) for offline use.
    """
    import torch
    from megatron.core.fusions import fused_cross_entropy as fce

    ce: dict[str, Any] = {"calls": [], "cur": None, "train_fwd": [], "n_bwd": 0, "saved": 0, "save_n": save_n,
                          "orig": {n: getattr(fce, n) for n in CE_NAMES}, "fce": fce, "errors": []}
    _ST["ce"] = ce
    orig = ce["orig"]

    def guard(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # noqa: BLE001 - the probe must never break the step
                ce["errors"].append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
                return None
        return run

    def w_logits_max(x):
        # (grad mode is always off inside an autograd.Function.forward: forward-only passes are told apart later,
        # they are the forward records that no backward call claims; mb = the loss-level counter at this moment)
        rec = {"grad": bool(torch.is_grad_enabled()), "mb": _ST.get("mb", -1), "seq": len(ce["calls"])}
        rec["sample_ids"] = list(_ST.get("small", {}).get(rec["mb"], {}).get("sample_indices", [])) or None
        guard(lambda: rec.__setitem__("logits_in", _cs(x)))()
        ce["calls"].append(rec)
        ce["cur"] = rec
        out = orig["calculate_logits_max"](x)
        guard(lambda: rec.__setitem__("logits_max", _cs(out[1])))()
        ce["train_fwd"].append(rec)
        return out

    def w_predicted(vpl, target, logits_max, vs, ve):
        rec = ce["cur"]
        if rec is not None:
            guard(lambda: (rec.__setitem__("logits_shifted_in", _cs(vpl)), rec.__setitem__("target", _cs(target))))()
        out = orig["calculate_predicted_logits"](vpl, target, logits_max, vs, ve)
        if rec is not None:
            def f():
                rec["target_mask"] = _cs(out[0])
                rec["masked_target_1d"] = _cs(out[1])
                rec["pred_sumexp"] = _cs(out[2])
                rec["exp_logits"] = _cs(out[3])
            guard(f)()
        return out

    def w_loss(exp_logits, pred_sumexp):
        out = orig["calculate_cross_entropy_loss"](exp_logits, pred_sumexp)
        rec = ce["cur"]
        if rec is not None:
            def f():
                rec["softmax"] = _cs(out[0])
                rec["loss"] = _cs(out[1])
                rec["softmax_ptr"] = int(out[0].data_ptr())
            guard(f)()
        return out

    def w_grad(softmax, grad_output, target_mask, masked_target_1d):
        n = ce["n_bwd"]
        ce["n_bwd"] += 1
        rec: dict[str, Any] = {"n": n}
        ptr = int(softmax.data_ptr())
        fwd = next((r for r in reversed(ce["train_fwd"]) if r.get("softmax_ptr") == ptr and not r.get("matched")), None)
        if fwd is not None:
            fwd["matched"] = True
        rec["fwd_ptr_match"] = fwd is not None
        rec["fwd_seq"] = fwd["seq"] if fwd is not None else None
        rec["mb"] = fwd["mb"] if fwd else None
        rec["sample_ids"] = fwd["sample_ids"] if fwd else None
        clones: tuple = ()
        saved_in = None

        def pre():
            nonlocal clones, saved_in
            rec["softmax_in"] = _cs(softmax)
            rec["grad_output"] = _cs(grad_output)
            rec["target_mask_in"] = _cs(target_mask)
            rec["masked_target_in"] = _cs(masked_target_1d)
            rec["grad_output_nonzero"] = bool(grad_output.ne(0).any().item())
            clones = (softmax.clone(), softmax.clone())
            if rec["grad_output_nonzero"] and ce["saved"] < ce["save_n"]:
                ce["saved"] += 1
                saved_in = {f"ce{n}|softmax_in": softmax.detach().to("cpu", copy=True),
                            f"ce{n}|grad_output": grad_output.detach().to("cpu", copy=True),
                            f"ce{n}|target_mask": target_mask.detach().to("cpu", copy=True),
                            f"ce{n}|masked_target_1d": masked_target_1d.detach().to("cpu", copy=True)}
        guard(pre)()
        out = orig["calculate_gradients"](softmax, grad_output, target_mask, masked_target_1d)

        def post():
            rec["out"] = _cs(out)
            if clones:
                r1 = orig["calculate_gradients"](clones[0], grad_output, target_mask, masked_target_1d)
                r2 = _eager_gradients(clones[1], grad_output, target_mask, masked_target_1d)
                rec["recompute_compiled"] = _cs(r1)
                rec["recompute_eager"] = _cs(r2)
                rec["compiled_equal"] = bool(torch.equal(r1, out))
                rec["eager_equal"] = bool(torch.equal(r2, out))
                rec["eager_vs_out_max_abs"] = float((r2.float() - out.float()).abs().max().item())
            if saved_in is not None:
                saved_in[f"ce{n}|out"] = out.detach().to("cpu", copy=True)
                _ST["big"].update(saved_in)
        guard(post)()
        ce["calls"].append({"bwd": rec})
        return out

    fce.calculate_logits_max = w_logits_max
    fce.calculate_predicted_logits = w_predicted
    fce.calculate_cross_entropy_loss = w_loss
    fce.calculate_gradients = w_grad
    import hashlib
    import inspect

    return {"source_sha256": hashlib.sha256(inspect.getsource(fce).encode()).hexdigest(), "file": fce.__file__}


def _ce_flush() -> list[Any]:
    ce = _ST.get("ce")
    if not ce:
        return []
    out = list(ce["calls"]) + [{"errors": list(ce["errors"])}]
    ce["calls"].clear()
    ce["train_fwd"].clear()
    ce["errors"].clear()
    ce["n_bwd"] = 0
    ce["saved"] = 0
    ce["cur"] = None
    return out


def _install_loss_level() -> None:
    """Wrap ``get_loss_function`` (outside the e3 probe's wrapper): record the loss inputs and metrics of every
    training micro batch and the per-token-row checksums of the gradient arriving at the logits."""
    import torch
    from miles.backends.training_utils import loss as loss_mod

    orig = loss_mod.get_loss_function

    def wrapped(*a, **k):
        func = orig(*a, **k)

        def recorded(args, batch, logits, sum_of_sample_mean, *rest, **kw):
            if not _ST["module_hooks"]:
                _ST["mb"] += 1
            mb = _ST["mb"]
            small = _ST["small"].setdefault(mb, {})
            ids = batch.get("sample_indices") if isinstance(batch, dict) else None
            small["sample_indices"] = [int(i) for i in (ids or [])]
            for key in ("log_probs", "advantages", "loss_masks", "response_lengths", "total_lengths", "rewards"):
                v = batch.get(key) if isinstance(batch, dict) else None
                if v is None:
                    continue
                items = v if isinstance(v, (list, tuple)) else [v]
                small[key] = [x.detach().to("cpu", copy=True) if hasattr(x, "detach") else x for x in items]
            if torch.is_tensor(logits) and logits.requires_grad:
                _put(f"{mb}|<logits>|fwd|0", logits)
                small["logits_row_bits"] = _row_bits(logits)

                def on_logit_grad(g, mb=mb):
                    gg = _unscale("|bwd_", g)
                    _ST["small"].setdefault(mb, {})["logit_grad_row_bits"] = _row_bits(gg)
                    _ST["small"][mb]["logit_grad_row_abs"] = gg.double().abs().sum(-1).reshape(-1).cpu()
                    if mb in _ST["logit_grad_mb"] or _ST["logit_grad_samples"] & set(_ST["small"][mb].get("sample_indices", [])):
                        _ST["big"][f"{mb}|logit_grad"] = gg.detach().to("cpu", copy=True)
                logits.register_hook(on_logit_grad)
            loss, log = func(args, batch, logits, sum_of_sample_mean, *rest, **kw)
            small["loss_metrics"] = {str(n): float(v.detach().float().cpu()) if hasattr(v, "detach") else float(v)
                                     for n, v in (log.items() if isinstance(log, dict) else ())}
            small["loss"] = float(loss.detach().float().cpu())
            return loss, log
        return recorded

    loss_mod.get_loss_function = wrapped
    _ST["loss_orig"] = orig
    _ST["loss_mod"] = loss_mod


def _row_bits(t: Any) -> Any:
    """[rows] int64 checksum of the raw bits of every row of the last dim (+ rows' L1)."""
    import torch

    x = t.detach().reshape(-1, t.shape[-1]).contiguous()
    v = x.view(torch.int32).to(torch.int64) if x.dtype == torch.float32 else x.view(torch.int16).to(torch.int64)
    w = (torch.arange(v.shape[-1], device=v.device, dtype=torch.int64) % 1000003) + 1
    return torch.stack([v.sum(-1), (v * w).sum(-1)]).cpu()


def dump_trace(actor: Any, *, directory: str, tag: str, coord: dict | None = None, save: bool = True,
               heavy: bool = True) -> dict[str, Any]:
    """Write this rank's trace and clear the buffers (the hooks stay installed); ``save=False`` only clears."""
    import torch

    if not save:
        _ce_flush()
        _ST["records"].clear()
        _ST["tensors"].clear()
        _ST["wgrad"].clear()
        _ST["small"].clear()
        _ST["big"].clear()
        _ST["mb"] = -1
        _ST["fo"] = -1
        return {"saved": False}

    if coord is None:
        from yeto.rl.engine.miles_adapter.cut_plugin import _backend

        coord = dict(_backend(actor).coord())
    rec = _flush()
    payload = {"coord": dict(coord), "records": rec, "tensors": dict(_ST["tensors"]) if heavy else {},
               "wgrad": dict(_ST["wgrad"]) if heavy else {}, "bwd_scale": _ST.get("bwd_scale", 1.0),
               "kernels": _ST.get("kernels"), "small": dict(_ST.get("small", {})), "big": dict(_ST.get("big", {})) if heavy else {},
               "device": _device_info(), "ce": _ce_flush(),
               "n_microbatches": _ST["mb"] + 1, "env": {k: os.environ.get(k) for k in (
                   "CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO", "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                   "NVIDIA_TF32_OVERRIDE", "NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")}}
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"trace_{tag}_dp{coord.get('dp', 0)}.pt")
    torch.save(payload, path)
    _ST["tensors"].clear()
    _ST["wgrad"].clear()
    _ST["small"].clear()
    _ST["big"].clear()
    _ST["mb"] = -1
    _ST["fo"] = -1
    _ST["kernels"] = None if _ST.get("profile") else _ST.get("kernels")
    return {"path": path, "records": len(rec), "microbatches": payload["n_microbatches"], "coord": dict(coord)}


def _profile_step(mb: int) -> None:
    """Kernel names of micro batch 0 (forward+backward; stopped when micro batch 1 starts). Never raises."""
    if not _ST.get("profile"):
        return
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile

        if mb == 0 and _ST["prof"] is None and _ST["kernels"] is None:
            _ST["prof"] = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
            _ST["prof"].__enter__()
        elif mb == 1 and _ST["prof"] is not None:
            torch.cuda.synchronize()
            prof, _ST["prof"] = _ST["prof"], None
            prof.__exit__(None, None, None)
            counts: dict[str, int] = {}
            for ev in prof.events():
                if getattr(ev, "device_type", None) is not None and "CUDA" in str(ev.device_type):
                    counts[ev.name] = counts.get(ev.name, 0) + 1
            _ST["kernels"] = dict(sorted(counts.items()))
    except Exception as exc:  # noqa: BLE001 - the probe must not break the step
        _ST["kernels"] = {"__error__": f"{type(exc).__name__}: {exc}"}
        _ST["prof"] = None


def _device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                            "bf16_reduced_precision_reduction":
                                bool(getattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", None))}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        info.update({"gpu": props.name, "uuid": str(getattr(props, "uuid", "?")), "device_index": torch.cuda.current_device(),
                     "sm_count": props.multi_processor_count,
                     "capability": list(torch.cuda.get_device_capability())})
    for mod in ("transformer_engine", "flash_attn", "megatron.core", "flashinfer"):
        try:
            m = __import__(mod, fromlist=["x"])
            info[mod] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - absent on CPU / not installed
            info[mod] = f"unavailable: {type(exc).__name__}"
    try:
        info["cudnn"] = torch.backends.cudnn.version()
        info["cublas"] = ".".join(str(v) for v in torch.cuda.get_cublas_version()) if hasattr(torch.cuda, "get_cublas_version") else None
    except Exception as exc:  # noqa: BLE001
        info["cublas_error"] = f"{type(exc).__name__}: {exc}"
    info["env"] = {k: v for k, v in sorted(os.environ.items())
                   if k.startswith(("NVTE_", "CUBLAS", "CUDA_", "NCCL_", "TORCH_", "TE_", "NVIDIA_", "CUDNN", "PYTORCH_"))}
    return info


def reset_for_tests() -> None:
    ce = _ST.get("ce")
    if ce:
        for n, f in ce["orig"].items():
            setattr(ce["fce"], n, f)
    if _ST.get("loss_mod") is not None:
        _ST["loss_mod"].get_loss_function = _ST["loss_orig"]
    for h in _ST.get("handles", []):
        h.remove()
    _ST.clear()
