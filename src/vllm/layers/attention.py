import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl


def prefill_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    head_dim: int,
    scale: float
) -> torch.Tensor:
    # make tensor contiguous for faster access
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # output placeholder
    output = torch.empty_like(q)

    # how many tokens can be loaded into SRAM to perform attention operations (< 48KB)
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (float 32 attention scores) + BLOCK_M * head_dim * 4 (for Q) + BLOCK_N * head_dim * 4 (for K, V)
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # number of sequences
    num_seq = cu_seqlens.size(0) - 1

    # geting the max seqlen
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seqlen = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # split the calculation different grid and launch all kernel.
    grid = (triton.cdiv(max_seqlen, BLOCK_M), num_q_head, num_seq)

    flash_attention_varlen_kernel[grid](
        q, k, v, output, 
        cu_seqlens,
        scale,
        num_q_head=num_q_head,
        num_kv_head=num_kv_head,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output

@triton.jit
def flash_attention_varlen_kernel(
    q, k, v, o,                         # pointer (GPU kneral does not reciew the entire tensor, only pointer to GPU memory)
    cu_seqlens_pointer,                 # pointer
    scale,
    num_q_head: tl.constexpr,
    num_kv_head: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # corresponding to the grid above
    block_index = tl.program_id(0)
    head_index = tl.program_id(1)
    sequence_index = tl.program_id(2)

    # determine which kv head index is used for this query (for GQA)
    kv_head_idx = head_index // (num_q_head // num_kv_head)

    # cu_seqlens is a pointer to the array, gets the length of the sequence
    seq_start = tl.load(cu_seqlens_pointer + sequence_index)
    seq_end = tl.load(cu_seqlens_pointer + sequence_index + 1)
    seqlen = seq_end - seq_start

    # if current processing token is larger than seqlen, skip
    if block_index * BLOCK_M >= seqlen:
        return

    # offset for getting all values of one token's one head
    offset_d = tl.arange(0, head_dim)
    # token positions in relation to the sequence start for q
    offset_q = block_index * BLOCK_M + tl.arange(0, BLOCK_M)

    # the starting postion of selected token in the q tensor
    q_token_memory_offset = (seq_start + offset_q[:, None]) * num_q_head * head_dim # (BLOCK_M, 1)
    # head offset in relation to the start of the selected token
    q_head_memory_offset = head_index * head_dim
    # offset for getting all values of one token's one head
    q_dim_offset = offset_d[None, :] # (1, head_dim)

    # total offset of q of this block
    q_offset = q_token_memory_offset + q_head_memory_offset + q_dim_offset # (BLOCK_M, head_dim)
    # pointer to the q value of this block
    q_ptr = q + q_offset

    # mask any token that is outside of the sequence
    mask_q = offset_q < seqlen
    q = tl.load(q_ptr, mask=mask_q[:, None], other=0.0)

    # number of KV blocks
    num_blocks = tl.cdiv(seqlen, BLOCK_N)

    # Initialize output accumulators
    sum = tl.zeros([BLOCK_M], dtype=tl.float32)             # sum of 
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    for block_n in range(num_blocks):
        # token position relative to the seq_start position
        offset_kv = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # mask any token that is outside of the sequence
        mask_kv = offset_kv < seqlen

        # getting the offset for the k block
        k_token_memory_offset = (seq_start + offset_kv[None, :]) * num_kv_head * head_dim # (1, BLOCK_N)
        k_head_memory_offset = kv_head_idx * head_dim # variable
        k_dim_offset = offset_d[:, None] # (head_dim, 1)

        k_offset = k_token_memory_offset + k_head_memory_offset + k_dim_offset # (head_dim, BLOCK_N) transposed automatically
        k_ptr = k + k_offset

        # loading the k block from HBM
        k_block = tl.load(k_ptr, mask=mask_kv[None, :], other=0.0)

        # Attention calculation
        qk = tl.dot(q, k_block)   # (BLOCK_M, BLOCK_N)
        qk = qk * scale

        # only attent postion that is smaller than current position
        mask_casual = (seq_start + offset_q[:, None]) >= (seq_start + offset_kv[None, :])
        qk = tl.where(mask_casual & mask_kv[None, :], qk, -1e10)

        # online softmax update
        current_max = tl.max(qk, axis=1)            
        max_new = tl.maximum(current_max, max)
        alpha = tl.exp(max - max_new)           # old calculation need to be modified by this
        p = tl.exp(qk - max_new[:, None])       # current numerator value

        # used to start previous value of V
        acc = acc * alpha[:, None]

        # loading v block
        v_token_memory_offset = (seq_start + offset_kv[:, None]) * num_kv_head * head_dim # (BLOCK_N, 1)
        v_head_memory_offset = kv_head_idx * head_dim # variable
        v_dim_offset = offset_d[None, :] # (head_dim, 1)

        v_offset = v_token_memory_offset + v_head_memory_offset + v_dim_offset # (BLOCK_N, head_dim)
        v_ptr = v + v_offset

        v_block = tl.load(v_ptr, mask=mask_kv[:, None], other=0.0)

        # add current result to previous results
        acc = acc + tl.dot(p.to(v_block.dtype), v_block) 

        # update the demoninator for softmax
        sum = sum * alpha + tl.sum(p, axis=1)
        max = max_new

    # finialize final result after division
    acc = acc / sum[:, None]

    # get output offset
    o_token_memory_offset = (seq_start + offset_q[:, None]) * num_q_head * head_dim # (BLOCK_M, 1)
    o_head_memory_offset = head_index * head_dim # variable
    o_dim_offset = offset_d[None, :] # (1, head_dim)

    o_offset = o_token_memory_offset + o_head_memory_offset + o_dim_offset
    o_ptr = o_offset + o

    # write output to HBM
    tl.store(o_ptr, acc.to(o.dtype.element_ty), mask=mask_q[:, None])

        




        

