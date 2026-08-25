# Copyright (C) Tahoe Therapeutics 2025-2026. All rights reserved.
"""Compatibility shim for environments where only flash-attn-4 is installed.

flash-attn-4 (PyPI name ``flash-attn-4``, importable as ``flash_attn``) is Dao-AILab's
CUTE-DSL rewrite targeting newest-gen GPUs (e.g. Blackwell/B200). It ships only
``flash_attn.cute.*`` -- no top-level ``__version__`` and no ``flash_attn.bert_padding``
-- whereas llm-foundry's MPT implementation (which tahoe_x1 builds on) expects the
classic flash-attn v1/v2 API surface:

  - ``flash_attn.__version__`` is read unconditionally at import time
    (``is_flash_v2_installed()`` in llmfoundry's attention module), so without it,
    importing ``tahoe_x1.model``/``tahoe_x1.data`` raises AttributeError immediately.
  - ``flash_attn.bert_padding.unpad_input``/``pad_input`` are called unconditionally
    by tahoe_x1's own encoder (``model/blocks.py``) to compute padding info, regardless
    of which attention implementation (``attn_impl``) is actually selected.

This module patches only what's missing, so it's a no-op wherever a real classic
flash-attn (v1/v2) is already installed (e.g. the project's Docker image on
Ampere/Hopper GPUs). It also bridges llm-foundry's ``flash_attn_interface.flash_attn_varlen_func``
call onto flash-attn-4's ``flash_attn.cute.flash_attn_varlen_func`` kernel, whose
signature is a near-exact match, so ``attn_impl: "flash"`` gets real Blackwell-accelerated
attention instead of falling back to ``attn_impl: "torch"``.
"""
import importlib.machinery
import sys
import types

import torch
from einops import rearrange, repeat


def _has_working_bert_padding() -> bool:
    try:
        from flash_attn import bert_padding

        return hasattr(bert_padding, "unpad_input")
    except ImportError:
        return False


def _has_working_rotary() -> bool:
    try:
        from flash_attn.layers.rotary import RotaryEmbedding  # noqa: F401

        return True
    except ImportError:
        return False


def _has_working_varlen_func() -> bool:
    try:
        from flash_attn import flash_attn_interface

        return hasattr(flash_attn_interface, "flash_attn_varlen_func")
    except ImportError:
        return False


def _has_fa4_cute() -> bool:
    try:
        from flash_attn.cute import flash_attn_varlen_func  # noqa: F401

        return True
    except ImportError:
        return False


class _UnavailableRotaryEmbedding:
    """Stub for `flash_attn.layers.rotary.RotaryEmbedding`.

    llm-foundry's MPT module imports this unconditionally once
    `is_flash_v2_installed()` is True, but only instantiates it when a model
    config explicitly requests the "dail" rope implementation. tahoe_x1 never
    does (grep confirms no "rope"/"rotary" config anywhere in this repo), so
    this only needs to exist to satisfy the import -- it fails loudly instead of
    silently if that assumption is ever wrong.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "flash-attn-4 does not provide fused rotary embeddings "
            "(flash_attn.layers.rotary); this stub was only meant to satisfy an "
            "unconditional import, not to be instantiated. If you're seeing this, "
            "something now configures RoPE and needs a real implementation.",
        )


class _IndexFirstAxis(torch.autograd.Function):
    """Row-gather by index.

    Vendored from the classic flash-attn ``bert_padding``
    module (pure PyTorch, no CUDA kernel) since flash-attn-4 doesn't ship it.
    """

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(
            0,
            repeat(indices, "z -> z d", d=grad_output.shape[1]),
            grad_output,
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


def _index_first_axis(input, indices):
    return _IndexFirstAxis.apply(input, indices)


def _unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
        (1, 0),
    )
    return (
        _index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        seqlens_in_batch,
    )


def _pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1]
    output = torch.zeros(
        batch * seqlen,
        dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _fa4_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    softcap=0.0,
    **_unused,
):
    """Adapts llm-foundry's classic flash-attn-2 varlen call onto flash-attn-4's
    CuTeDSL kernel (``flash_attn.cute.flash_attn_varlen_func``).

    The two share
    the same packed ``(total, nheads, headdim)`` tensor convention and most
    keyword names, but differ on a few conventions handled below: FA4 has no
    dropout, takes ``None`` (not ``-1``) for "no window", and returns lse
    instead of the (unused, in this codebase) attention-probs tuple.
    """
    if dropout_p:
        raise NotImplementedError("flash-attn-4 does not support attention dropout.")
    if alibi_slopes is not None:
        raise NotImplementedError("flash-attn-4 does not support alibi_slopes.")
    if return_attn_probs:
        raise NotImplementedError(
            "flash-attn-4 adapter does not support return_attn_probs/needs_weights.",
        )

    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func

    left, right = window_size
    # FA4 always returns an (output, lse) pair (lse is None unless return_lse=True);
    # llm-foundry expects flash_attn_varlen_func to return just the output tensor.
    output, _lse = _fa4_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(None if left < 0 else left, None if right < 0 else right),
        softcap=softcap,
    )
    return output


def patch() -> None:
    """Idempotently patch ``flash_attn`` so llm-foundry's MPT import succeeds
    and ``bert_padding`` is available, when only flash-attn-4 (or nothing) is
    installed."""
    try:
        import flash_attn
    except ImportError:
        # No flash-attn of any kind (e.g. a CPU-only install). Stand up a stub package
        # so the unconditional reads below resolve. The stub must carry a real
        # ModuleSpec: `transformers` probes for installed packages via
        # importlib.util.find_spec, which rejects a spec-less module already present in
        # sys.modules ("flash_attn.__spec__ is None"). Without this, merely importing
        # tahoe_x1 failed outright with "RuntimeError: Failed to import
        # transformers.modeling_utils".
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn",
            loader=None,
            is_package=True,
        )
        flash_attn.__path__ = []
        sys.modules["flash_attn"] = flash_attn

    if not hasattr(flash_attn, "__version__"):
        # Unconditional on purpose, even when nothing is installed. llm-foundry's
        # is_flash_v2_installed() guards only the `import flash_attn` itself, then reads
        # `flash_attn.__version__` outside that try -- so a version-less module raises
        # AttributeError rather than degrading to False. The version claim is only ever
        # consulted for capability gating; if `attn_impl: "flash"` is then requested
        # with no real kernel behind it, llm-foundry's own lazy import of
        # flash_attn_interface raises a clear RuntimeError at the first forward pass.
        flash_attn.__version__ = "2.99.0"

    if not _has_working_bert_padding():
        bert_padding = types.ModuleType("flash_attn.bert_padding")
        bert_padding.index_first_axis = _index_first_axis
        bert_padding.unpad_input = _unpad_input
        bert_padding.pad_input = _pad_input
        flash_attn.bert_padding = bert_padding
        sys.modules["flash_attn.bert_padding"] = bert_padding

    if not _has_working_rotary():
        layers_pkg = sys.modules.get("flash_attn.layers")
        if layers_pkg is None:
            layers_pkg = types.ModuleType("flash_attn.layers")
            flash_attn.layers = layers_pkg
            sys.modules["flash_attn.layers"] = layers_pkg
        rotary_mod = types.ModuleType("flash_attn.layers.rotary")
        rotary_mod.RotaryEmbedding = _UnavailableRotaryEmbedding
        layers_pkg.rotary = rotary_mod
        sys.modules["flash_attn.layers.rotary"] = rotary_mod

    if not _has_working_varlen_func() and _has_fa4_cute():
        interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
        interface_mod.flash_attn_varlen_func = _fa4_flash_attn_varlen_func
        flash_attn.flash_attn_interface = interface_mod
        sys.modules["flash_attn.flash_attn_interface"] = interface_mod
