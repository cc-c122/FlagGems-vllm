# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton

from flaggems_vllm.ops.flash_mla import flash_mla_attn_kernel
from flaggems_vllm.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise NotImplementedError(f"MetaX flash_mla requires contiguous {name}")


def flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
):
    """Run the C550-compatible Triton FlashMLA decode path."""
    logger.debug("GEMS FLASH MLA")
    assert causal, "causal False not supported"
    assert d > dv, "mla with rope dim should be larger than no rope dim"

    _ = max_seqlen_pad
    _ = h_kv

    _require_contiguous("q", q)
    _require_contiguous("blocked_k", blocked_k)
    _require_contiguous("block_table", block_table)
    _require_contiguous("cache_seqlens", cache_seqlens)

    batch_size, query_length, head_num, head_dim = q.shape
    q = q.view(-1, head_num, head_dim)
    blocked_k = blocked_k.view(-1, head_dim)

    sm_scale = 1 / math.sqrt(d)
    output = torch.empty((b * s_q, h_q, dv), dtype=q.dtype, device=device)

    # C550 exposes 64 KiB shared memory per block. The generic major-8 launch
    # (BLOCK_H=32, BLOCK_N=64, num_stages=2) exceeds that limit.
    block_h = 16
    block_n = 16
    grid = (triton.cdiv(head_num, block_h), batch_size)

    with torch_device_fn.device(device):
        flash_mla_attn_kernel[grid](
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            output,
            sm_scale,
            head_num,
            q.stride(0),
            q.stride(1),
            blocked_k.stride(-2),
            block_table.stride(0),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            PAGE_SIZE=block_size,
            HEAD_DIM_V=dv,
            HEAD_DIM=d,
            num_warps=4,
            num_stages=1,
        )

    return output.view(b, query_length, h_q, dv)
