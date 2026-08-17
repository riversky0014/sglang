import torch
import triton
import triton.language as tl

from sglang.kernels.optional import get_optional_sgl_kernel
from sglang.srt.utils import is_cpu, next_power_of_2

_is_cpu = is_cpu()

_fill_accept_out_cache_loc_cpu = (
    get_optional_sgl_kernel("fill_accept_out_cache_loc_cpu") if _is_cpu else None
)
_fill_bonus_tokens_cpu = (
    get_optional_sgl_kernel("fill_bonus_tokens_cpu") if _is_cpu else None
)


@triton.jit
def fill_bonus_tokens(
    accept_tokens,
    accept_lens,
    bonus_tokens_ptr,
    accept_stride: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    # `accept_lens` includes the bonus token; the last accepted slot is at -1.
    accept_len = tl.load(accept_lens + pid)

    # accept_stride = per-req width of accept_tokens (= accept_index.shape[1]).
    bonus_token_idx = accept_stride * pid + accept_len - 1
    bonus_token = tl.load(accept_tokens + bonus_token_idx)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


def fill_bonus_tokens_func(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,  # mutable
    accept_stride: int,
    batch_size: int,
):
    if _fill_bonus_tokens_cpu is not None:
        _fill_bonus_tokens_cpu(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_stride,
        )
        return
    if _is_cpu:
        raise RuntimeError(
            "CPU speculative decoding requires the optional "
            "sgl_kernel.fill_bonus_tokens_cpu operation"
        )
    fill_bonus_tokens[(batch_size,)](
        accept_tokens,
        accept_lens,
        bonus_tokens,
        accept_stride,
    )


@triton.jit
def fill_accept_out_cache_loc(
    accept_index,
    out_cache_loc,
    accept_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accept_out_cache_loc + dst, value)


def fill_accept_out_cache_loc_func(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_out_cache_loc: torch.Tensor,  # mutable
    size: int,
):
    if _fill_accept_out_cache_loc_cpu is not None:
        _fill_accept_out_cache_loc_cpu(
            accept_index,
            out_cache_loc,
            accept_out_cache_loc,
        )
        return
    if _is_cpu:
        raise RuntimeError(
            "CPU speculative decoding requires the optional "
            "sgl_kernel.fill_accept_out_cache_loc_cpu operation"
        )
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )
