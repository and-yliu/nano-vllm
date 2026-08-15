import torch

from minivllm.engine.sequence import Sequence


@torch.inference_mode()
def sample(logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
    """Pick one token per sequence. logits: [num_seqs, vocab_size].

    Sampling is done in fp32 -- the softmax denominator is a reduction over
    150k vocab entries, and reductions need more precision than storage.

    temperature <= 0 means greedy. Note SamplingParams.__post_init__ currently
    rejects temperature <= 1e-10, so greedy has to be requested by relaxing that
    assert; greedy is what the parity oracle needs, so it is worth allowing.
    """
    logits = logits.float()
    temperatures = torch.tensor(
        [s.sampling_params.temperature for s in seqs],
        dtype=torch.float32,
        device=logits.device,
    )

    # get the max probable token
    greedy_tokens = logits.argmax(dim=-1)

    # do a multinomial distribution to get the next token, softmax to get each token's probability
    safe_temps = temperatures.clamp_min(1e-5).unsqueeze(-1)
    probs = torch.softmax(logits / safe_temps, dim=-1)
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    tokens = torch.where(temperatures <= 0, greedy_tokens, sampled_tokens)
    return tokens.tolist()
