"""Interleaved AR / block-diffusion decoding driven by a router.

AR and diffusion share one backbone and one KV cache. The cache always holds
the causal prefix, so the decoder can hand over from one mode to the other at
any segment boundary:

    AR segment:   `ar_chunk` tokens, one forward pass each, causal attention.
    DLM segment:  one block of `block_length` tokens, denoised with
                  bidirectional attention, then one causal pass to commit the
                  block to the cache and get the next-token logits.

After each segment the router picks the mode for the next one.
"""

import dataclasses

import torch

from src import router as router_lib
from src.nemotron import modeling_nemotron_labs_diffusion as modeling

# Prompt tokens per prefill forward pass.
PREFILL_CHUNK = 8192


@dataclasses.dataclass
class Segment:
    """One routed segment of the output."""

    mode: router_lib.Mode
    num_tokens: int


@dataclasses.dataclass
class InterleavedOutput:
    """Result of interleaved decoding (batch size 1).

    Attributes:
        token_ids: Generated tokens, without the prompt.
        nfe: Number of model forward passes.
        segments: Mode and size of each segment, in order.
    """

    token_ids: list[int]
    nfe: int
    segments: list[Segment]


def _set_causal(model: modeling.NemotronLabsDiffusionModel, causal: bool):
    for layer in model.encoder.layers:
        layer.self_attn.diffusion_lm = not causal


def _argmax(logit: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logit, dim=-1, keepdim=True)


@torch.inference_mode()
def interleaved_generate(
    model: modeling.NemotronLabsDiffusionModel,
    prompt_ids: torch.Tensor,
    router: router_lib.Router,
    eos_token_id: int,
    max_new_tokens: int = 256,
    block_length: int = 32,
    threshold: float = 0.9,
    ar_chunk: int = 8,
) -> InterleavedOutput:
    """Generates greedily, letting `router` pick AR or diffusion per segment.

    Args:
        model: Loaded Nemotron-Labs-Diffusion model.
        prompt_ids: Prompt token ids, shape (1, prompt_len).
        router: Decides the mode of each segment.
        eos_token_id: Generation stops after this token.
        max_new_tokens: Generation budget.
        block_length: Tokens per diffusion segment.
        threshold: Diffusion confidence threshold for unmasking.
        ar_chunk: Tokens per AR segment.

    Returns:
        The generated tokens with NFE and the routing trace.
    """
    assert prompt_ids.shape[0] == 1, "interleaved decoding is batch size 1"
    mask_id = model.mask_token_id
    device = prompt_ids.device

    # Learned routers need the last hidden state; capture it from the encoder.
    captured: dict[str, torch.Tensor] = {}
    hook = model.encoder.register_forward_hook(
        lambda _m, _i, o: captured.update(h=o.last_hidden_state[0, -1])
    )

    # Causal prefill: builds the cache and the first next-token logits. Long
    # prompts are prefilled in chunks so activation memory stays bounded
    # (exact for causal attention over the cache), and only the last position
    # goes through the LM head (full-prompt logits cost prompt_len x vocab).
    _set_causal(model, True)
    cache = None
    for start in range(0, prompt_ids.shape[1], PREFILL_CHUNK):
        enc = model.encoder(
            input_ids=prompt_ids[:, start : start + PREFILL_CHUNK],
            past_key_values=cache,
            use_cache=True,
            use_causal_mask=True,
        )
        cache = enc.past_key_values
    next_logit = model.diffusion_head(enc.last_hidden_state[:, -1, :])
    nfe = 1

    tokens: list[int] = []
    segments: list[Segment] = []
    last_mode: router_lib.Mode | None = None
    done = False

    while len(tokens) < max_new_tokens and not done:
        state = router_lib.RouterState(
            next_logit=next_logit[0],
            hidden=captured["h"],
            num_generated=len(tokens),
            last_mode=last_mode,
        )
        mode = router(state)
        budget = max_new_tokens - len(tokens)
        new: list[int] = []

        if mode == "ar":
            _set_causal(model, True)
            for _ in range(min(ar_chunk, budget)):
                token = _argmax(next_logit)
                new.append(int(token))
                if new[-1] == eos_token_id:
                    done = True
                    break
                out = model(
                    token,
                    past_key_values=cache,
                    use_cache=True,
                    use_causal_mask=True,
                )
                cache = out.past_key_values
                next_logit = out.logits[:, -1, :]
                nfe += 1
        else:
            length = min(block_length, budget)
            block = torch.full(
                (1, length), mask_id, device=device, dtype=prompt_ids.dtype
            )
            block[:, 0] = _argmax(next_logit)[:, 0]
            _set_causal(model, False)
            for _ in range(length):
                masked = block == mask_id
                if not masked.any():
                    break
                logits = model(
                    block, past_key_values=cache, use_cache=False
                ).logits
                nfe += 1
                num_transfer = modeling._get_num_transfer_tokens(masked, length)
                x0, transfer = modeling._get_transfer_index(
                    logits,
                    0.0,
                    masked,
                    block,
                    num_transfer_tokens=num_transfer[:, 0],
                    threshold=threshold,
                )
                block = torch.where(transfer, x0, block)
            new = block[0].tolist()
            if eos_token_id in new:
                new = new[: new.index(eos_token_id) + 1]
                done = True
            if not done:
                # Commit the block to the cache with a causal pass.
                _set_causal(model, True)
                out = model(
                    block,
                    past_key_values=cache,
                    use_cache=True,
                    use_causal_mask=True,
                )
                cache = out.past_key_values
                next_logit = out.logits[:, -1, :]
                nfe += 1

        tokens.extend(new)
        segments.append(Segment(mode, len(new)))
        last_mode = mode

    hook.remove()
    return InterleavedOutput(tokens, nfe, segments)
