"""Span-switched decoding: the model itself decides when to use diffusion.

Decoding is AR until the model emits `<diff>`. From there it denoises
diffusion blocks of `block_size` tokens against the causal KV cache until a
block produces `</diff>`; the block is truncated after the first `</diff>`
(positions after it are ignored once everything before it is unmasked), the
kept tokens are committed with one causal pass, and decoding returns to AR.
No router: switching is a token the model predicts.

With `mode="ar"` every token is decoded AR (spans included), the reference
for what the same weights produce without diffusion.
"""

from collections.abc import Callable

import torch

from src import spans
from src.interleave import InterleavedOutput, Segment, _argmax, _set_causal
from src.nemotron import modeling_nemotron_labs_diffusion as modeling

# Safety cap on diffusion blocks per span.
MAX_SPAN_BLOCKS = 64


@torch.inference_mode()
def span_generate(
    model: modeling.NemotronLabsDiffusionModel,
    prompt_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int = 1024,
    block_size: int = 8,
    threshold: float = 0.9,
    mode: str = "spans",
    seed_ar: bool = True,
    on_event: Callable[[dict], None] | None = None,
) -> InterleavedOutput:
    """Greedy span-switched generation (batch size 1).

    Args:
        model: Backbone (with span adapters applied, see `span_model`).
        prompt_ids: (1, prompt_len) prompt ending with the assistant header.
        eos_token_id: Generation stops after this token.
        max_new_tokens: Generation budget.
        block_size: Diffusion block length inside spans.
        threshold: Diffusion confidence threshold for unmasking.
        mode: "spans" (diffusion inside <diff> spans) or "ar" (all AR).
        seed_ar: Seed each diffusion block's first token with the AR argmax.
        on_event: Optional callback for streaming / visualization, called
            with {"type": "ar", "token"}, {"type": "block", "tokens"} (a
            diffusion block after seeding and after every denoising pass,
            masks included), {"type": "commit", "tokens"} (the kept block);
            every event carries the running forward-pass count "nfe".

    Returns:
        Tokens, forward passes and the AR / diffusion segments.
    """
    assert prompt_ids.shape[0] == 1
    assert mode in ("spans", "ar"), mode
    mask_id = model.mask_token_id
    device = prompt_ids.device

    _set_causal(model, True)
    # Only the last position goes through the LM head (full-prompt logits
    # cost prompt_len x 131k and ran six workers per GPU out of memory).
    enc = model.encoder(input_ids=prompt_ids, use_cache=True, use_causal_mask=True)
    cache = enc.past_key_values
    next_logit = model.diffusion_head(enc.last_hidden_state[:, -1, :])
    nfe = 1
    tokens: list[int] = []
    segments: list[Segment] = []

    def emit(kind: str, **kw) -> None:
        if on_event is not None:
            on_event({"type": kind, "nfe": nfe, **kw})

    def add(mode_: str, n: int) -> None:
        if segments and segments[-1].mode == mode_:
            segments[-1] = Segment(
                mode_, segments[-1].num_tokens + n
            )  # ty: ignore[invalid-argument-type]
        else:
            segments.append(
                Segment(mode_, n)
            )  # ty: ignore[invalid-argument-type]

    def commit(block: torch.Tensor) -> None:
        nonlocal cache, next_logit, nfe
        _set_causal(model, True)
        o = model(
            block, past_key_values=cache, use_cache=True, use_causal_mask=True
        )
        cache, next_logit = o.past_key_values, o.logits[:, -1, :]
        nfe += 1

    while len(tokens) < max_new_tokens:
        token = int(_argmax(next_logit))
        tokens.append(token)
        add("ar", 1)
        emit("ar", token=token)
        if token == eos_token_id:
            break
        commit(torch.tensor([[token]], device=device))
        if token != spans.DIFF_OPEN or mode == "ar":
            continue
        for _ in range(MAX_SPAN_BLOCKS):
            budget = max_new_tokens - len(tokens)
            if budget <= 0:
                break
            length = min(block_size, budget)
            block = torch.full(
                (1, length), mask_id, device=device, dtype=prompt_ids.dtype
            )
            if seed_ar:
                block[:, 0] = _argmax(next_logit)[:, 0]
            _set_causal(model, False)
            close = None
            emit("block", tokens=block[0].tolist(), mask_id=mask_id)
            while True:
                masked = block == mask_id
                hits = (block[0] == spans.DIFF_CLOSE).nonzero()
                if len(hits) and not masked[0, : int(hits[0])].any():
                    close = int(hits[0])
                    break
                if not masked.any():
                    break
                logits = model(
                    block, past_key_values=cache, use_cache=False
                ).logits
                nfe += 1
                num_transfer = modeling._get_num_transfer_tokens(masked, length)
                x0, transfer = modeling._get_transfer_index(
                    logits, 0.0, masked, block,
                    num_transfer_tokens=num_transfer[:, 0], threshold=threshold,
                )  # fmt: skip
                block = torch.where(transfer, x0, block)
                emit("block", tokens=block[0].tolist(), mask_id=mask_id)
            keep = block[:, : close + 1] if close is not None else block
            new = keep[0].tolist()
            if eos_token_id in new:
                new = new[: new.index(eos_token_id) + 1]
                tokens.extend(new)
                add("dlm", len(new))
                emit("commit", tokens=new)
                return InterleavedOutput(tokens, nfe, segments)
            tokens.extend(new)
            add("dlm", len(new))
            commit(keep)
            emit("commit", tokens=new)
            if close is not None:
                break
    _set_causal(model, True)
    return InterleavedOutput(tokens[:max_new_tokens], nfe, segments)
