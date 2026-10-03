"""Checks that AR <-> diffusion switching can be driven by the causal state.

Two checks, both against a fresh causal (AR) pass over the same tokens:

1. Segment boundaries: at every router decision in an interleaved run, the
   hidden state handed to the router matches a fresh causal pass over the
   prompt plus the tokens generated so far. So the router's input depends
   only on the tokens, not on which mode produced them.
2. Mid-block: after a diffusion block is committed with one causal pass, the
   hidden state at any position k matches a fresh causal pass, and cropping
   the KV cache to position k and continuing in AR reproduces `ar_generate`
   from that prefix. So a switch can happen at any position of a committed
   block, not only at its end.

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.check_switch

Exits with status 1 if any check fails.
"""

import argparse
import sys

import torch
from transformers import PreTrainedTokenizerBase

from src import engine as engine_lib
from src import interleave
from src import router as router_lib
from src.nemotron import modeling_nemotron_labs_diffusion as modeling

# bf16 incremental decoding vs one full prefill differs slightly.
MIN_COSINE = 0.999

Model = modeling.NemotronLabsDiffusionModel


class RecordingRouter:
    """Wraps a router and records every state it is asked about."""

    def __init__(self, inner: router_lib.Router) -> None:
        """Initializes the recorder.

        Args:
            inner: Router that makes the actual decisions.
        """
        self.inner = inner
        self.states: list[router_lib.RouterState] = []

    def __call__(self, state: router_lib.RouterState) -> router_lib.Mode:
        """Records `state` and delegates the decision."""
        self.states.append(state)
        return self.inner(state)


def set_causal(model: Model, causal: bool) -> None:
    """Switches every attention layer between causal and bidirectional."""
    for layer in model.encoder.layers:
        layer.self_attn.diffusion_lm = not causal


def fresh_causal_hidden(model: Model, seq: torch.Tensor) -> torch.Tensor:
    """Returns the last-position hidden state of a causal pass over `seq`."""
    set_causal(model, True)
    out = model.encoder(input_ids=seq, use_causal_mask=True)
    return out.last_hidden_state[0, -1].float()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two vectors."""
    return torch.nn.functional.cosine_similarity(
        a.float(), b.float(), dim=0
    ).item()


def check_boundaries(
    model: Model, prompt_ids: torch.Tensor, eos_token_id: int
) -> bool:
    """Runs check 1. Returns True if every switch point matches."""
    recorder = RecordingRouter(router_lib.CycleRouter(("ar", "dlm")))
    out = interleave.interleaved_generate(
        model, prompt_ids, recorder, eos_token_id, max_new_tokens=96
    )
    segments = " ".join(f"{s.mode}{s.num_tokens}" for s in out.segments)
    print(f"[1] segment boundaries ({segments})")

    ok = True
    for state in recorder.states:
        generated = torch.tensor(
            [out.token_ids[: state.num_generated]],
            device=prompt_ids.device,
            dtype=prompt_ids.dtype,
        )
        ref = fresh_causal_hidden(model, torch.cat([prompt_ids, generated], 1))
        ref_token = model.diffusion_head(ref.to(model.dtype)).argmax().item()
        cos = cosine(state.hidden, ref)
        same = int(state.next_logit.argmax()) == ref_token
        ok &= cos >= MIN_COSINE and same
        print(
            f"    switch@{state.num_generated:3d}: cosine={cos:.5f} "
            f"same next token={same}"
        )
    return ok


def denoise_block(
    model: Model,
    cache: modeling.DynamicCache,
    first_token: torch.Tensor,
    length: int,
    threshold: float,
) -> torch.Tensor:
    """Denoises one block with bidirectional attention, without caching it."""
    block = torch.full(
        (1, length),
        model.mask_token_id,
        device=first_token.device,
        dtype=first_token.dtype,
    )
    block[0, 0] = first_token
    set_causal(model, False)
    for _ in range(length):
        masked = block == model.mask_token_id
        if not masked.any():
            break
        logits = model(block, past_key_values=cache, use_cache=False).logits
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
    return block


def check_mid_block(
    model: Model,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    cut: int,
    ar_tokens: int = 16,
) -> bool:
    """Runs check 2. Returns True if mid-block switching is exact."""
    set_causal(model, True)
    prefill = model(prompt_ids, use_cache=True, use_causal_mask=True)
    cache = prefill.past_key_values
    first = prefill.logits[:, -1].argmax(-1)
    block = denoise_block(model, cache, first, length=32, threshold=0.9)

    # Commit the block causally: one pass gives a causal state per position.
    set_causal(model, True)
    commit = model.encoder(
        input_ids=block,
        past_key_values=cache,
        use_cache=True,
        use_causal_mask=True,
    )
    print(f"[2] mid-block, block={tokenizer.decode(block[0])[:60]!r}")

    ok = True
    for k in (3, cut, block.shape[1] - 1):
        ref = fresh_causal_hidden(
            model, torch.cat([prompt_ids, block[:, : k + 1]], 1)
        )
        cos = cosine(commit.last_hidden_state[0, k], ref)
        ok &= cos >= MIN_COSINE
        print(f"    position {k:2d}: cosine={cos:.5f}")

    # Switch to AR at `cut`: drop the rest of the block from the cache.
    modeling._crop_dynamic_cache(cache, prompt_ids.shape[1] + cut + 1)
    token = model.diffusion_head(commit.last_hidden_state[:, cut]).argmax(
        -1, keepdim=True
    )
    continued = []
    for _ in range(ar_tokens):
        continued.append(int(token))
        out = model(
            token, past_key_values=cache, use_cache=True, use_causal_mask=True
        )
        token = out.logits[:, -1].argmax(-1, keepdim=True)

    prefix = torch.cat([prompt_ids, block[:, : cut + 1]], 1)
    ref_ids, _ = model.ar_generate(prefix, max_new_tokens=ar_tokens)
    same = continued == ref_ids[0, prefix.shape[1] :].tolist()
    ok &= same
    print(
        f"    crop at {cut}, then {ar_tokens} AR tokens == ar_generate: {same}"
    )
    return ok


@torch.inference_mode()
def main() -> None:
    """Loads the model and runs both checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=engine_lib.DEFAULT_REPO)
    parser.add_argument(
        "--prompt",
        default="Write a Python function that checks if a number is prime.",
    )
    parser.add_argument("--cut", type=int, default=11)
    args = parser.parse_args()

    engine = engine_lib.NemotronEngine(args.model, device="cuda")
    model, tokenizer = engine.model, engine.tokenizer
    prompt_ids = engine.encode_chat(args.prompt)

    ok = check_boundaries(model, prompt_ids, tokenizer.eos_token_id)
    ok &= check_mid_block(model, tokenizer, prompt_ids, args.cut)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
