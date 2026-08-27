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

import math

import pytest
import torch

import flaggems_vllm
from benchmark.base import Benchmark
from flaggems_vllm.runtime.backend._metax.ops.minimax_sparse_attention import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
)


def _torch_sparse_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
):
    del output
    assert decode_query_len == 1
    batch, num_heads, head_dim = q.shape
    group_size = num_heads // num_kv_heads
    topk = topk_idx.shape[-1]
    offsets = torch.arange(SPARSE_BLOCK_SIZE, device=q.device)
    result = torch.empty_like(q)

    for kv_head in range(num_kv_heads):
        logical_blocks = topk_idx[kv_head].long()
        physical_pages = torch.gather(block_table.long(), 1, logical_blocks)
        pages = kv_cache[physical_pages, kv_head]
        k = pages[..., :head_dim].reshape(batch, topk * SPARSE_BLOCK_SIZE, head_dim)
        v = pages[..., head_dim:].reshape(
            batch, topk * SPARSE_BLOCK_SIZE, head_dim
        )
        positions = (
            logical_blocks[..., None] * SPARSE_BLOCK_SIZE + offsets[None, None, :]
        ).reshape(batch, topk * SPARSE_BLOCK_SIZE)
        valid = positions < seq_lens[:, None]

        head_start = kv_head * group_size
        head_end = head_start + group_size
        query = q[:, head_start:head_end].float()
        scores = torch.einsum("bhd,bkd->bhk", query, k.float()) * sm_scale
        scores.masked_fill_(~valid[:, None, :], -float("inf"))
        probs = torch.softmax(scores, dim=-1)
        result[:, head_start:head_end] = torch.einsum(
            "bhk,bkd->bhd", probs, v.float()
        ).to(q.dtype)
    return result


def _metax_sparse_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
):
    minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
    )
    return output


class MiniMaxM3SparseAttentionBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.float16, torch.bfloat16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        (1, 4096, 16, 32, 128),
        (4, 4096, 16, 32, 128),
        (8, 8192, 32, 32, 128),
        (16, 16384, 64, 32, 128),
        (32, 32768, 64, 32, 128),
    ]
    DEFAULT_SHAPE_DESC = "B, SEQ_LEN, TOPK, H, D"

    def init_user_config(self):
        super().init_user_config()
        if any(len(shape) != 5 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for batch, seq_len, topk, num_heads, head_dim in self.shapes:
            if seq_len < topk * SPARSE_BLOCK_SIZE:
                continue
            yield self._build_inputs(
                batch, seq_len, topk, num_heads, head_dim, dtype
            )

    def _build_inputs(self, batch, seq_len, topk, num_heads, head_dim, dtype):
        torch.manual_seed(batch + seq_len + topk)
        num_kv_heads = 2
        num_blocks = math.ceil(seq_len / SPARSE_BLOCK_SIZE)
        num_pages = batch * num_blocks
        q = torch.randn(
            batch,
            num_heads,
            head_dim,
            device=self.device,
            dtype=dtype,
        ) * 0.1
        kv_cache = torch.randn(
            num_pages,
            num_kv_heads,
            SPARSE_BLOCK_SIZE,
            2 * head_dim,
            device=self.device,
            dtype=dtype,
        ) * 0.1
        block_table = torch.randperm(
            num_pages, device=self.device, dtype=torch.int32
        ).reshape(batch, num_blocks)
        selected = torch.arange(
            num_blocks - topk,
            num_blocks,
            device=self.device,
            dtype=torch.int32,
        )
        topk_idx = selected[None, None, :].expand(
            num_kv_heads, batch, topk
        ).contiguous()
        seq_lens = torch.full(
            (batch,), seq_len, device=self.device, dtype=torch.int32
        )
        output = torch.empty_like(q)
        return (
            q,
            kv_cache,
            topk_idx,
            block_table,
            seq_lens,
            num_kv_heads,
            head_dim**-0.5,
            output,
            1,
        )


@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax",
    reason="MiniMax M3 specialization requires MetaX",
)
def test_minimax_m3_sparse_attention():
    bench = MiniMaxM3SparseAttentionBenchmark(
        op_name="minimax_m3_sparse_attention",
        torch_op=_torch_sparse_decode,
    )
    bench.set_gems(_metax_sparse_decode)
    bench.run()
