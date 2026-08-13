import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl

from minivllm.utils.context import get_context


def prefill_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    head_dim: int,
    scale: float,
    max_seqlen: int | None = None,
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

    # geting the max seqlen -- prefer the caller's host-side value, since reading
    # cu_seqlens off the device syncs and this runs once per layer
    if max_seqlen is None:
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

# paged prefill (prefill_flash_attention with the K/V loads swapped for block-table gathers)
def paged_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    head_dim: int,
    block_size: int,
    scale: float,
    max_seqlen_q: int | None = None,
) -> torch.Tensor:
    
    q = q.contiguous()
    output = torch.empty_like(q)

    if head_dim <= 64:
        BLOCK_M = BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = BLOCK_N = 32
    else:
        BLOCK_M = BLOCK_N = 16

    num_seq = cu_seqlens_q.size(0) - 1
    assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0), (
        "cu_seqlens_q and cu_seqlens_k must describe the same number of sequences"
    )

    if max_seqlen_q is None:
        cu_seqlens_q_cpu = cu_seqlens_q.cpu()
        q_lens = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]
        max_seqlen_q = q_lens.max().item()

        # num_cached = k_len - q_len is computed per sequence inside the kernel
        # and drives both the causal mask and RoPE. A key span shorter than its
        # query span makes it negative, corrupting both silently. Only checked
        # on this path, since it is the one already paying for a sync.
        cu_seqlens_k_cpu = cu_seqlens_k.cpu()
        k_lens = cu_seqlens_k_cpu[1:] - cu_seqlens_k_cpu[:-1]
        assert bool((k_lens >= q_lens).all()), (
            f"key span must cover the query span per sequence, got q={q_lens.tolist()} "
            f"k={k_lens.tolist()}"
        )

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), num_q_head, num_seq)

    paged_prefill_attention_kernel[grid](
        q, k_cache, v_cache, output,
        cu_seqlens_q, cu_seqlens_k, block_tables,
        scale,
        num_q_head=num_q_head,
        num_kv_head=num_kv_head,
        head_dim=head_dim,
        block_size=block_size,
        max_block_size=block_tables.shape[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


@triton.jit
def paged_prefill_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    block_tables_ptr,
    scale,
    num_q_head: tl.constexpr,
    num_kv_head: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # corresponding to the grid above
    block_index = tl.program_id(0)
    head_index = tl.program_id(1)
    seq_index = tl.program_id(2)

    # determine which kv head index is used for this query (for GQA)
    kv_head_idx = head_index // (num_q_head // num_kv_head)

    # queries packed varlen, exactly as in the contiguous prefill kernel
    q_start = tl.load(cu_seqlens_q_ptr + seq_index)
    q_end = tl.load(cu_seqlens_q_ptr + seq_index + 1)
    q_len = q_end - q_start

    # keys span the whole context: cached prefix plus the tokens of this pass
    k_start = tl.load(cu_seqlens_k_ptr + seq_index)
    k_end = tl.load(cu_seqlens_k_ptr + seq_index + 1)
    k_len = k_end - k_start

    # if current processing token is larger than q_len, skip
    if block_index * BLOCK_M >= q_len:
        return

    # the cached prefix sits before this pass's tokens, so query i in this pass
    # is at absolute position num_cached + i. This is the offset that both the
    # causal mask and (in qwen3) RoPE have to agree on.
    num_cached = k_len - q_len

    # offset for getting all values of one token's one head
    offset_d = tl.arange(0, head_dim)
    # token positions in relation to the start of this pass's queries
    offset_q = block_index * BLOCK_M + tl.arange(0, BLOCK_M)
    # ... and the same tokens as absolute positions in the full sequence
    q_position = num_cached + offset_q

    q_offset = (
        (q_start + offset_q[:, None]) * num_q_head * head_dim
        + head_index * head_dim
        + offset_d[None, :]
    )
    mask_q = offset_q < q_len
    q = tl.load(q_ptr + q_offset, mask=mask_q[:, None], other=0.0)

    # Initialize output accumulators
    sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # walk the entire context, not just this pass's tokens
    num_blocks = tl.cdiv(k_len, BLOCK_N)

    for block_n in range(num_blocks):
        # absolute key positions within the sequence
        offset_kv = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # logically which page does each key belong to, and where inside it
        block_table_num = offset_kv // block_size
        block_offset = offset_kv % block_size

        in_range = (offset_kv < k_len) & (block_table_num < max_block_size)

        # where is the physical position of the kv_cache from the block_table
        block_table_index_ptr = block_tables_ptr + seq_index * max_block_size + block_table_num
        block_table_offset = tl.load(block_table_index_ptr, mask=in_range, other=-1)

        # discard if any block that does not show position, use 0 as a place holder
        valid = in_range & (block_table_offset != -1)
        block_table_offset = tl.where(valid, block_table_offset, 0).to(tl.int64)

        # cache shape: (num_blocks, block_size, num_kv_heads, head_dim)
        # K is loaded transposed, (head_dim, BLOCK_N), ready for tl.dot
        k_cache_offset = (
            block_table_offset[None, :] * block_size * num_kv_head * head_dim +
            block_offset[None, :] * num_kv_head * head_dim +
            kv_head_idx * head_dim +
            offset_d[:, None]
        )
        k_block = tl.load(k_cache_ptr + k_cache_offset, mask=valid[None, :], other=0.0)

        qk = tl.dot(q, k_block)   # (BLOCK_M, BLOCK_N)
        qk = qk * scale

        # causal in absolute positions: a query may see the whole cached prefix
        # plus the earlier tokens of this pass, but nothing after itself
        mask_casual = q_position[:, None] >= offset_kv[None, :]
        qk = tl.where(mask_casual & valid[None, :], qk, -1e10)

        # online softmax update
        current_max = tl.max(qk, axis=1)
        max_new = tl.maximum(current_max, max)
        alpha = tl.exp(max - max_new)
        p = tl.exp(qk - max_new[:, None])

        # used to scale previous value of V
        acc = acc * alpha[:, None]

        # V is loaded untransposed, (BLOCK_N, head_dim)
        v_cache_offset = (
            block_table_offset[:, None] * block_size * num_kv_head * head_dim +
            block_offset[:, None] * num_kv_head * head_dim +
            kv_head_idx * head_dim +
            offset_d[None, :]
        )
        v_block = tl.load(v_cache_ptr + v_cache_offset, mask=valid[:, None], other=0.0)

        # add current result to previous results
        acc = acc + tl.dot(p.to(v_block.dtype), v_block)

        # update the denominator for softmax
        sum = sum * alpha + tl.sum(p, axis=1)
        max = max_new

    # finalize final result after division
    acc = acc / sum[:, None]

    o_offset = (
        (q_start + offset_q[:, None]) * num_q_head * head_dim
        + head_index * head_dim
        + offset_d[None, :]
    )
    tl.store(o_ptr + o_offset, acc.to(o_ptr.dtype.element_ty), mask=mask_q[:, None])


@triton.jit
def paged_attention_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    block_tables_ptr,
    context_lens_ptr,
    num_q_head: tl.constexpr,
    num_kv_head: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    max_block_size: tl.constexpr,
    BLOCK_N: tl.constexpr
):  
    # grid vlaue
    block_index = tl.program_id(0)
    head_index = tl.program_id(1)

    # determine which kv head index is used for this query (for GQA)
    kv_head_idx = head_index // (num_q_head // num_kv_head)

    # load q
    q_dim_offset = tl.arange(0, head_dim)
    q_offset = block_index * num_q_head * head_dim + head_index * head_dim + q_dim_offset
    q = tl.load(q_ptr + q_offset)      #(head_dim)

    # length of the current sequence length of the block
    context_lens = tl.load(context_lens_ptr + block_index)

    # offset for getting all values of one token's one head
    offset_d = tl.arange(0, head_dim)

    # number of loop to load all kv_block to SRAM
    max_chunk = tl.cdiv(max_block_size * block_size, BLOCK_N)

    # online softmax variables
    max = -1e10
    sum = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)


    for chunk_index in range(max_chunk):
        start = chunk_index * BLOCK_N 

        if start < context_lens:
            # start index of each token in the sequence
            kv_dim = start + tl.arange(0, BLOCK_N) 

            # placeholder of qk
            qk = tl.zeros((BLOCK_N,), dtype=tl.float32) - 1e10

            # logically which page/block does the tokens belong to within the sequence  
            block_table_num = kv_dim // block_size
            # what is the offset from the start of the page/block
            block_offset = kv_dim % block_size

            # check if any token is outside of the sequence
            in_range = (kv_dim < context_lens) & (block_table_num < max_block_size)

            # where is the physical position of the kv_cache from the block_table
            block_table_index_ptr = block_tables_ptr + block_index * max_block_size + block_table_num
            block_table_offset = tl.load(block_table_index_ptr, mask=in_range, other=-1)

            # discard if any block that does not show position, use 0 as a place holder 
            valid = in_range & (block_table_offset != -1)
            block_table_offset = tl.where(valid, block_table_offset, 0).to(tl.int64)

            # cache shape: (num_blocks, block_size, num_kv_heads, head_dim)
            # where is the current block, block_offset, head
            cache_offset = (
                block_table_offset[None, :] * block_size * num_kv_head * head_dim +     #(1, BLOCK_N)
                block_offset[None, :] * num_kv_head * head_dim +                        #(1, BLOCK_N)
                kv_head_idx * head_dim +                                                 #(1)
                offset_d[:, None]                                                       #(head_dim, 1)
            )

            # load k_cache
            k_cache = tl.load(k_cache_ptr + cache_offset, mask=valid[None, :], other=0.0) # (head_dim, BLOCK_N)

            # calculate qk for single-token decode
            qk = tl.sum(q[:, None] * k_cache, axis=0) * scale      #(BLOCK_N)
            qk = tl.where(valid, qk, -1e10) 
        
            # for i in range(BLOCK_N):
            #     token_index = start + i

            #     if token_index < context_lens:
            #         block_num = token_index // block_size
            #         block_offset = token_index % block_size

            #         block_table_index_ptr = block_tables_ptr + block_index * max_block_size + block_num
            #         block_table_offset = tl.load(block_table_index_ptr) 

            #         if block_table_offset != -1:
            #             k_cache_offset = block_table_offset * block_size * num_kv_head * head_dim + block_offset * num_kv_head * head_dim + kv_head_idx * head_dim
            #             k_cache_dim = k_cache_offset + offset_d
            #             k_cache = tl.load(k_cache_ptr + k_cache_dim)

            #             score = tl.sum(q * k_cache) * scale

            #             mask_i = tl.arange(0, BLOCK_N) == i
            #             qk = tl.where(mask_i, score, qk)

            # online soft mask calculation
            current_max = tl.max(qk)            
            max_new = tl.maximum(current_max, max)
            alpha = tl.exp(max - max_new)
            p = tl.exp(qk - max_new)

            # Rescale accumulator
            acc = acc * alpha
            sum = sum * alpha

            # obtain v_cache from the same offset because kv cache share the same shape
            v_cache = tl.load(v_cache_ptr + cache_offset, mask=valid[None, :], other=0.0) #(head_dim, BLOCK_N)

            # accumulate new values
            weight = tl.where(valid, p, 0.0) # (BLOCK_N)
            acc = acc + tl.sum(weight[None, :] * v_cache, axis=1)
            sum = sum + tl.sum(weight)

            # for i in range(BLOCK_N):
            #     token_index = start + i

            #     if token_index < context_lens:
            #         block_num = token_index // block_size
            #         block_offset = token_index % block_size

            #         block_table_index_ptr = block_tables_ptr + block_index * max_block_size + block_num
            #         block_table_offset = tl.load(block_table_index_ptr) 

            #         if block_table_offset != -1:
            #             v_cache_offset = block_table_offset * block_size * num_kv_head * head_dim + block_offset * num_kv_head * head_dim + kv_head_idx * head_dim
            #             v_cache_dim = v_cache_offset + offset_d
            #             v_cache = tl.load(v_cache_ptr + v_cache_dim)

            #             mask_i = tl.arange(0, BLOCK_N) == i
            #             weight = tl.sum(tl.where(mask_i, p, 0.0))

            #             acc = acc + weight * v_cache
            #             sum = sum + weight

            max = max_new
    # final rescale
    output = acc / sum

    # load result to output
    output_offset = block_index * num_q_head * head_dim + head_index * head_dim + offset_d
    tl.store(output_ptr+output_offset, output)

    
def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
    head_dim: int,
    block_size: int,
    scale: float
):
    batch_size = q.shape[0]
    max_num_blocks = block_tables.shape[1]

    q = q.contiguous()
    output = torch.empty_like(q)

    if head_dim <= 128:
        BLOCK_N = 64
    else:
        BLOCK_N = 32

    grid = (batch_size, num_q_head)

    paged_attention_decode_kernel[grid](
        q, k_cache, v_cache, output, 
        block_tables, context_lens,
        num_q_head=num_q_head, num_kv_head=num_kv_head, head_dim=head_dim,
        block_size=block_size, scale=scale, max_block_size=max_num_blocks,
        BLOCK_N=BLOCK_N
    )

    return output

def save_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int
):
    # shape of the k
    token_num, num_kv_head, head_dim = k.shape

    # make kv continguous
    k = k.contiguous()
    v = v.contiguous()

    # make sure k and v have the same shape and slot mapping have the same number of token
    assert k.shape == v.shape
    assert slot_mapping.numel() == token_num

    # how to parallel the work
    grid = (token_num, num_kv_head)

    save_kv_cache_kernel[grid](
        k, v, k_cache, v_cache,
        slot_mapping,
        head_dim=head_dim,
        num_kv_head=num_kv_head,
    )

@triton.jit
def save_kv_cache_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    head_dim: tl.constexpr,
    num_kv_head: tl.constexpr,
):  
    # get grid parameters
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)

    offset_d = tl.arange(0, head_dim)

    # which slot in physical memory should the cache be stored
    slot_index = tl.load(slot_mapping_ptr + token_index)

    if slot_index == -1:
        return

    # load k and v
    offset = token_index * num_kv_head * head_dim + head_index * head_dim + offset_d
    k = tl.load(k_ptr + offset)
    v = tl.load(v_ptr + offset)

    # store k and v into the phsyical memory
    cache_offset = slot_index * num_kv_head * head_dim + head_index * head_dim + offset_d
    tl.store(k_cache_ptr + cache_offset, k)
    tl.store(v_cache_ptr + cache_offset, v)

class Attention(nn.Module):
    def __init__(
        self, 
        num_q_head: int,
        head_dim: int, 
        scale: float = 1.0,
        num_kv_head: int = None, 
        block_size: int = 16
    ):
        super().__init__()
        self.num_q_head = num_q_head
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_head = num_kv_head if num_kv_head is not None else num_q_head
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.Tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()

        # store kv_cache if  k_cache and v_cache is intialized
        if self.k_cache.numel() > 0 and self.v_cache.numel() > 0 and context.slot_mapping is not None:
            #Ensure k and v have the right shape
            if k.dim() == 4:  #(batch, token, head, head_dim)
                batch, token, head, head_dim = k.shape
                k = k.reshape(batch*token, head, head_dim)
                v = v.reshape(batch*token, head, head_dim)
            else:
                k = k.contiguous()
                v = v.contiguous()

            save_kv_cache(k, v, self.k_cache, self.v_cache, context.slot_mapping, block_size=self.block_size)

        scale = self.scale / self.head_dim ** 0.5

        if context.is_prefill:
            # prefill use flash attention
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q are needed for multiple sequences flash attention")

            # A cached prefix means the keys this pass needs are not the ones it
            # computed, so attention has to read them back out of the cache. The
            # kv cache write above already put this pass's K/V there.
            use_paged = (
                self.k_cache.numel() > 0
                and context.block_tables is not None
                and context.cu_seqlens_k is not None
                and context.slot_mapping is not None
            )

            if use_paged:
                # output shape: (token, head, head_dim)
                output = paged_prefill_attention(
                    q, self.k_cache, self.v_cache,
                    cu_seqlens, context.cu_seqlens_k, context.block_tables,
                    self.num_q_head, self.num_kv_head, self.head_dim,
                    self.block_size, scale,
                    # host-side already, so the wrapper needs no device read
                    max_seqlen_q=context.max_seqlen_q or None,
                )
            else:
                # no cache (or nothing cached yet): keys are this pass's own
                # output shape: (token, head, head_dim)
                output = prefill_flash_attention(
                    q, k, v, cu_seqlens, self.num_q_head, self.num_kv_head,
                    self.head_dim, scale, max_seqlen=context.max_seqlen_q or None,
                )
            # combine results from multiple heads
            return output.reshape(output.shape[0], self.num_q_head * self.head_dim)
        else:
            # decode use paged attention
            block_table = context.block_tables
            context_lens = context.context_lens

            if block_table is None or context_lens is None:
                raise ValueError("block_table and context_lens are needed for paged attention")
            
            #output shape: (batch, head, head_dim)
            output = paged_attention_decode(q, self.k_cache, self.v_cache, block_table, context_lens, self.num_q_head, self.num_kv_head, self.head_dim, self.block_size, scale)
            # combine results from multiple heads
            return output.reshape(output.shape[0], self.num_q_head * self.head_dim)
