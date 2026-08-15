import torch
from transformers import Qwen3Config

from minivllm.engine.sequence import Sequence
from minivllm.layers.attention import Attention
from minivllm.model.qwen3 import Qwen3ForCausalLM
from minivllm.engine.sampler import sample
from minivllm.utils.context import set_context, reset_context
from minivllm.utils.loader import load_model


class ModelRunner:
    def __init__(
        self,
        config: Qwen3Config,
        path: str | None = None,
        block_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        max_model_length: int | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        self.device = torch.device(device)

        # bf16 needs Ampere (sm_80+); on older cards it is emulated, slower, and
        # less accurate than fp16. Same rule as test_parity.
        if dtype is None:
            major, _ = torch.cuda.get_device_capability(self.device)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.dtype = dtype

        # set block size to config
        config.block_size = block_size
        self.config = config
        self.block_size = block_size

        # set maximum length 
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_length = min(
            max_model_length or config.max_position_embeddings,
            config.max_position_embeddings,
        )

        # build the model object
        self.model = Qwen3ForCausalLM(config).to(device=self.device, dtype=self.dtype)
        # load pretrained model parameter from huggingface
        if path is not None:
            load_model(self.model, path)
        self.model.eval()

        # Order matters. Warmup runs while the caches are still empty tensors,
        # so Attention skips its cache write and needs no block tables -- and it
        # is what tells allocate_kv_cache how much transient memory to reserve.
        self.warmup_model()
        self.num_blocks = self.allocate_kv_cache(gpu_memory_utilization)


    @torch.inference_mode()
    def warmup_model(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # how many sequence to fill up the max_num_batched_tokens
        num_seqs = max(1, self.max_num_batched_tokens // self.max_model_length)
        seqs = [
            Sequence(token_ids=[0] * self.max_model_length, block_size=self.block_size)
            for _ in range(num_seqs)
        ]
        # run prefill of a full max_num_batched_tokens run
        self.run(seqs, is_prefill=True)

        torch.cuda.empty_cache()

    def allocate_kv_cache(self, gpu_memory_utilization: float) -> int:
        # get the total and free amount of gpu memory
        free, total = torch.cuda.mem_get_info(self.device)

        # get current bytes and peak bytes allocated (from the warm up)
        stats = torch.cuda.memory_stats(self.device)
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]

        # (total - free): gpu occupied by other things
        # (peak - current): gpu occupied peak memory usage during model execution
        available = total * gpu_memory_utilization - (total - free) - (peak - current)

        # layer number, kv heads, head dimensions
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim or (
            self.config.hidden_size // self.config.num_attention_heads
        )

        # 2 = K and V
        # number of bytes for one block 
        block_bytes = (
            2 * num_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        )
        # number of blocks available for this gpu
        num_blocks = int(available) // block_bytes
        assert num_blocks >= 1, (
            f"not enough free memory for even one KV block: {int(available)} bytes "
            f"available, {block_bytes} per block"
        )

        # zeros, not empty: a masked-off read of an untouched block should see 0,
        # not whatever NaN happened to be resident.
        kv_cache = torch.zeros(
            2, num_layers, num_blocks, self.block_size, num_kv_heads, head_dim,
            dtype=self.dtype, device=self.device,
        )

        # modules() yields in registration order, i.e. layer order
        layers = [m for m in self.model.modules() if isinstance(m, Attention)]
        assert len(layers) == num_layers, (
            f"found {len(layers)} Attention modules but config says {num_layers} layers"
        )
        # assign each layer a chunk of kv cache
        for i, layer in enumerate(layers):
            layer.k_cache = kv_cache[0, i]
            layer.v_cache = kv_cache[1, i]

        self.kv_cache = kv_cache
        return num_blocks


    def _to_gpu(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        # move tensor from CPU RAM to GPU
        # pin_memory=True, page lock in physical RAM and enable direct CPU->GPU transfer (Direct memory access)
        # non_blocking=True, CPU does not wait till the transfer is complete
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(
            self.device, non_blocking=True
        )

    # pad the block table so blocktable have the same length
    def _pad_block_tables(self, seqs: list[Sequence]) -> list[list[int]]:
        max_num_block = max(len(seq.block_table) for seq in seqs)
        return [
            seq.block_table + [-1] * (max_num_block - len(seq.block_table))
            for seq in seqs
        ]

    # slot index of a block in physical gpu memory 
    def _slots(self, seq: Sequence, start: int, end: int) -> list[int]:
        return [
            seq.block_table[i // self.block_size] * self.block_size + i % self.block_size
            for i in range(start, end)
        ]


    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # the cumulative sequence length of each of the input sequences, length: num_seqs + 1
        cu_seqlens_q = [0]
        # the cumulative sequence length of each of the input sequences, length: num_seqs + 1
        cu_seqlens_k = [0]
        # the sequence length of each of the input sequences, length: num_seqs
        seqlens_q = []
        # the sequence length of each of the input sequences, length: num_seqs
        seqlens_k = []
        # what tokens are runned, length: sum of all input_ids after prefix cache
        input_ids = []
        # the physical position the tokens are in cache, length: same as input_ids
        slot_mappings = []
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []

        # for each sequence 
        for seq in seqs:
            num_cached_tokens = seq.num_cached_tokens
            # input ids for uncached tokens
            input_ids.extend(seq[num_cached_tokens:])
            # only count uncached sequence length for q
            seqlens_q.append(seq.num_tokens - num_cached_tokens)
            # count all sequence length for k
            seqlens_k.append(seq.num_tokens)
            # calculate cumulative sequence length
            cu_seqlens_q.append(seqlens_q[-1] + cu_seqlens_q[-1])
            cu_seqlens_k.append(seqlens_k[-1] + cu_seqlens_k[-1])
            # build slot mapping
            if seq.block_table:
                slot_mappings.extend(self._slots(seq, num_cached_tokens, seq.num_tokens))

        # if any thing hit the cache, if yes, build the block table
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            block_tables = self._pad_block_tables(seqs)

        # initialize context
        set_context(
            is_prefill=True,
            cu_seqlens_q=self._to_gpu(cu_seqlens_q, torch.int32),
            cu_seqlens_k=self._to_gpu(cu_seqlens_k, torch.int32),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=self._to_gpu(slot_mappings, torch.long) if slot_mappings else None,
            context_lens=None,
            block_tables=self._to_gpu(block_tables, torch.int32) if block_tables else None,
        )

        # return input_ids
        return self._to_gpu(input_ids, torch.long)


    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []
        slot_mappings = []

        for seq in seqs:
            # for decode only the last token is needed
            input_ids.append(seq.last_token)
            # build context lens for each sequence
            context_lens.append(seq.num_tokens)
            # build slot mapping for just the last token
            slot_mappings.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        # set context for decode
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=self._to_gpu(slot_mappings, torch.long),
            context_lens=self._to_gpu(context_lens, torch.long),
            block_tables=self._to_gpu(self._pad_block_tables(seqs), torch.int32),
        )

        return self._to_gpu(input_ids, torch.long)


    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, seqs: list[Sequence], is_prefill: bool
    ) -> torch.Tensor:
        # calculate model output
        hidden_states = self.model(input_ids)
        # convert output into logits
        return self.model.compute_logits(hidden_states)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """One engine step: prepare -> forward -> sample."""
        # get input token ids
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)

        # get logits for each sequence
        logits = self.run_model(input_ids, seqs, is_prefill)
        # get token ids for each sequence
        token_ids = sample(logits, seqs)

        # clear context
        reset_context()
        return token_ids
