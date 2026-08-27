# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Correctness tests for the MetaX MiniMax M3 sparse-attention kernels."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._metax.ops.minimax_sparse_attention import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)


pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax",
    reason="MiniMax M3 specialization requires MetaX",
)

DEVICE = flaggems_vllm.device
HEAD_DIM = 128


def _assert_topk_equal_unordered(actual, expected):
    assert actual.shape == expected.shape
    actual_rows = actual.cpu().reshape(-1, actual.shape[-1]).tolist()
    expected_rows = expected.cpu().reshape(-1, expected.shape[-1]).tolist()
    for actual_row, expected_row in zip(actual_rows, expected_rows):
        assert set(actual_row) == set(expected_row)


def _reference_index_topk(
    idx_q,
    index_kv_cache,
    block_table,
    q_lens,
    seq_lens,
    prefix_lens,
    topk,
    init_blocks,
    local_blocks,
):
    total_q, num_idx_heads, _ = idx_q.shape
    out = torch.full(
        (num_idx_heads, total_q, topk),
        -1,
        device=idx_q.device,
        dtype=torch.int32,
    )
    q_start = 0
    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        q = idx_q[q_start : q_start + q_len]
        num_blocks = (seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        pages = block_table[req_id, :num_blocks]
        k = index_kv_cache[pages].reshape(num_blocks * SPARSE_BLOCK_SIZE, -1)
        score = torch.einsum("qhd,kd->hqk", q.float(), k.float())

        q_pos = prefix_len + torch.arange(q_len, device=idx_q.device)
        k_pos = torch.arange(k.shape[0], device=idx_q.device)
        score.masked_fill_(k_pos[None, :] > q_pos[:, None], -float("inf"))
        score = score.reshape(
            num_idx_heads, q_len, num_blocks, SPARSE_BLOCK_SIZE
        ).max(dim=3).values

        valid_blocks = (q_pos + SPARSE_BLOCK_SIZE) // SPARSE_BLOCK_SIZE
        for local_q, num_valid in enumerate(valid_blocks.tolist()):
            score[:, local_q, : min(init_blocks, num_valid)] = 1e30
            local_start = max(0, num_valid - local_blocks)
            score[:, local_q, local_start:num_valid] = 1e29
            selected = min(topk, num_valid)
            out[:, q_start + local_q, :selected] = score[
                :, local_q, :num_valid
            ].topk(selected, dim=1).indices
        q_start += q_len
    return out


def _reference_sparse_attention(
    q,
    kv_cache,
    topk_idx,
    block_table,
    q_lens,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    sm_scale,
):
    total_q, num_heads, head_dim = q.shape
    group_size = num_heads // num_kv_heads
    out = torch.empty_like(q)
    token_start = 0

    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        for local_q in range(q_len):
            token = token_start + local_q
            query_position = prefix_len + local_q
            for kv_head in range(num_kv_heads):
                logical_blocks = topk_idx[kv_head, token]
                logical_blocks = logical_blocks[logical_blocks >= 0].long()
                keys = []
                values = []
                for logical_block in logical_blocks.tolist():
                    physical_page = int(block_table[req_id, logical_block])
                    block_start = logical_block * SPARSE_BLOCK_SIZE
                    valid_tokens = min(
                        SPARSE_BLOCK_SIZE,
                        seq_len - block_start,
                        query_position - block_start + 1,
                    )
                    if valid_tokens <= 0:
                        continue
                    page = kv_cache[physical_page, kv_head, :valid_tokens]
                    keys.append(page[:, :head_dim])
                    values.append(page[:, head_dim:])

                k = torch.cat(keys, dim=0).float()
                v = torch.cat(values, dim=0).float()
                head_start = kv_head * group_size
                head_end = head_start + group_size
                query = q[token, head_start:head_end].float()
                scores = torch.einsum("hd,kd->hk", query, k) * sm_scale
                probs = torch.softmax(scores, dim=-1)
                out[token, head_start:head_end] = torch.einsum(
                    "hk,kd->hd", probs, v
                ).to(q.dtype)
        token_start += q_len
    assert token_start == total_q
    return out


def _make_block_table(batch, max_blocks):
    pages = torch.randperm(batch * max_blocks, device=DEVICE, dtype=torch.int32)
    return pages.reshape(batch, max_blocks)


def test_public_exports_use_metax_implementation():
    assert flaggems_vllm.minimax_m3_index_decode is minimax_m3_index_decode
    assert flaggems_vllm.minimax_m3_sparse_attn is minimax_m3_sparse_attn


@torch.inference_mode()
def test_minimax_m3_prefill_index_topk_matches_reference():
    torch.manual_seed(0)
    q_lens = torch.tensor([4, 3], device=DEVICE, dtype=torch.int32)
    prefix_lens = torch.tensor([0, 1024], device=DEVICE, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    max_query_len = int(q_lens.max())
    max_seq_len = int(seq_lens.max())
    batch = q_lens.numel()
    max_blocks = (max_seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    block_table = _make_block_table(batch, max_blocks)

    cu_seqlens = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)
    idx_q = torch.ones(
        int(q_lens.sum()), 2, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    index_kv_cache = torch.empty(
        batch * max_blocks,
        SPARSE_BLOCK_SIZE,
        HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    for req_id in range(batch):
        for logical_block in range(max_blocks):
            page = block_table[req_id, logical_block]
            index_kv_cache[page].fill_(logical_block + 1)

    score = minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        num_kv_heads=2,
    )
    actual = minimax_m3_index_topk(
        score,
        cu_seqlens,
        prefix_lens,
        max_query_len,
        topk=6,
        init_blocks=0,
        local_blocks=1,
    )
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        topk=6,
        init_blocks=0,
        local_blocks=1,
    )
    _assert_topk_equal_unordered(actual, expected)


@torch.inference_mode()
def test_minimax_m3_decode_index_topk_matches_reference():
    torch.manual_seed(1)
    seq_lens = torch.tensor([1024, 1536, 2048], device=DEVICE, dtype=torch.int32)
    q_lens = torch.ones_like(seq_lens)
    prefix_lens = seq_lens - 1
    max_seq_len = int(seq_lens.max())
    batch = seq_lens.numel()
    max_blocks = (max_seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    block_table = _make_block_table(batch, max_blocks)
    idx_q = torch.ones(
        batch, 1, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    index_kv_cache = torch.empty(
        batch * max_blocks,
        SPARSE_BLOCK_SIZE,
        HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    for req_id in range(batch):
        for logical_block in range(max_blocks):
            page = block_table[req_id, logical_block]
            index_kv_cache[page].fill_(logical_block + 1)

    actual = minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk=4,
        init_blocks=1,
        local_blocks=1,
        num_kv_heads=1,
        decode_query_len=1,
        max_decode_query_len=1,
    )
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        topk=4,
        init_blocks=1,
        local_blocks=1,
    )
    _assert_topk_equal_unordered(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_minimax_m3_sparse_prefill_matches_reference(dtype):
    torch.manual_seed(2)
    q_lens = torch.tensor([3, 2], device=DEVICE, dtype=torch.int32)
    prefix_lens = torch.tensor([256, 128], device=DEVICE, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_blocks = 3
    block_table = _make_block_table(batch, max_blocks)
    cu_seqlens = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)

    num_kv_heads = 1
    num_heads = 8
    q = torch.randn(
        int(q_lens.sum()), num_heads, HEAD_DIM, device=DEVICE, dtype=dtype
    ) * 0.1
    kv_cache = torch.randn(
        batch * max_blocks,
        num_kv_heads,
        SPARSE_BLOCK_SIZE,
        2 * HEAD_DIM,
        device=DEVICE,
        dtype=dtype,
    ) * 0.1
    topk_idx = torch.empty(
        num_kv_heads, int(q_lens.sum()), 2, device=DEVICE, dtype=torch.int32
    )
    topk_idx[:, :3] = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)
    topk_idx[:, 3:] = torch.tensor([0, 1], device=DEVICE, dtype=torch.int32)
    output = torch.empty_like(q)
    scale = HEAD_DIM**-0.5

    minimax_m3_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=int(q_lens.max()),
        num_kv_heads=num_kv_heads,
        sm_scale=scale,
        output=output,
    )
    expected = _reference_sparse_attention(
        q,
        kv_cache,
        topk_idx,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        scale,
    )
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_minimax_m3_sparse_decode_matches_reference(dtype):
    torch.manual_seed(3)
    seq_lens = torch.tensor([129, 257, 385, 513], device=DEVICE, dtype=torch.int32)
    q_lens = torch.ones_like(seq_lens)
    prefix_lens = seq_lens - 1
    batch = seq_lens.numel()
    max_blocks = 5
    block_table = _make_block_table(batch, max_blocks)
    num_kv_heads = 1
    num_heads = 8
    q = torch.randn(batch, num_heads, HEAD_DIM, device=DEVICE, dtype=dtype) * 0.1
    kv_cache = torch.randn(
        batch * max_blocks,
        num_kv_heads,
        SPARSE_BLOCK_SIZE,
        2 * HEAD_DIM,
        device=DEVICE,
        dtype=dtype,
    ) * 0.1
    topk_idx = torch.empty(
        num_kv_heads, batch, 2, device=DEVICE, dtype=torch.int32
    )
    for req_id, seq_len in enumerate(seq_lens.tolist()):
        visible = (seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        topk_idx[0, req_id] = torch.tensor(
            [max(0, visible - 2), visible - 1], device=DEVICE, dtype=torch.int32
        )
    output = torch.empty_like(q)
    scale = HEAD_DIM**-0.5

    minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        scale,
        output,
        decode_query_len=1,
    )
    expected = _reference_sparse_attention(
        q,
        kv_cache,
        topk_idx,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        scale,
    )
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
