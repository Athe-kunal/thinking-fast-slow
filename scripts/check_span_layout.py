"""Checks that the span-SFT training mask matches inference-time attention.

For one held-out conversation (base model, no adapters) compares
  (a) clean-sequence logits under the span mask vs a plain causal forward;
  (b) each noisy block's logits under the span mask vs the decoder's
      computation: causal prefill of the prefix, then a bidirectional pass
      over the (partially masked) block against that KV cache.
Reports max |diff| of log-probs and argmax agreement (bf16 numerics).

    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.check_span_layout
"""

import torch
from transformers import AutoTokenizer

from scripts import sft_data
from src import span_model, spans
from src.interleave import _set_causal


def main() -> None:
    tok = AutoTokenizer.from_pretrained(
        span_model.DEFAULT_REPO, trust_remote_code=True
    )
    model = span_model.load_base(span_model.DEFAULT_REPO, "cuda").eval()
    rec = sft_data.load(sft_data.cache_path([0], 4096), "heldout", 1, 0)[0]
    ex = spans.render(tok, rec["messages"], rec["tools"], diff_spans=True)
    g = torch.Generator().manual_seed(0)
    lay = spans.layout(ex, 8, model.mask_token_id, g)
    n = len(ex.ids)
    with torch.no_grad():
        with span_model.flex_attention(model):
            h = span_model.span_hidden(model, lay)
        lp_train = torch.log_softmax(model.diffusion_head(h).float(), -1)

        # (a) clean part vs plain causal forward.
        _set_causal(model, True)
        ids = torch.tensor([ex.ids], device="cuda")
        out = model.encoder(input_ids=ids, use_causal_mask=True)
        lp_ref = torch.log_softmax(
            model.diffusion_head(out.last_hidden_state[0]).float(), -1
        )
        d = (lp_train[:n] - lp_ref).abs().max(-1).values
        agree = (lp_train[:n].argmax(-1) == lp_ref.argmax(-1)).float().mean()
        print(
            f"clean: max|dlogp| {d.max():.3f} (median {d.median():.4f}), argmax agree {agree:.4f}"
        )

        # (b) noisy blocks vs prefill + bidirectional block pass.
        blocks = sorted(set(lay.block[lay.block >= 0].tolist()))
        for b in blocks:
            idx = (lay.block == b).nonzero()[:, 0]
            start = int(lay.block_start[idx[0]])
            _set_causal(model, True)
            enc = model.encoder(
                input_ids=ids[:, :start], use_cache=True, use_causal_mask=True
            )
            _set_causal(model, False)
            blk = lay.input_ids[:, idx].cuda()
            o = model(blk, past_key_values=enc.past_key_values, use_cache=False)
            lp_blk = torch.log_softmax(o.logits[0].float(), -1)
            d = (lp_train[idx.cuda()] - lp_blk).abs().max(-1).values
            agree = (
                (lp_train[idx.cuda()].argmax(-1) == lp_blk.argmax(-1))
                .float()
                .mean()
            )
            print(
                f"block {b} @ {start}: max|dlogp| {d.max():.3f}, argmax agree {agree:.3f}"
            )
        _set_causal(model, True)


if __name__ == "__main__":
    main()
