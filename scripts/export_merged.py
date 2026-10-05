"""Exports a span-SFT adapter as a full checkpoint for SGLang serving.

SGLang loads plain weights, so the LoRA deltas are merged into the backbone
and the trained `<diff>` / `</diff>` rows are written into the embedding and
LM-head matrices. The export (~6 GB) is a derived serving artifact: the
adapter checkpoint stays the source of truth, and the export can be deleted
and regenerated.

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.export_merged \
        --adapter runs/sft/e17_span_sft/train/adapter_final.pt \
        --out runs/sft/e17_span_sft/merged
"""

import argparse
import pathlib
import shutil

import torch
from huggingface_hub import snapshot_download

from src import span_model


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    model = span_model.load_base(span_model.DEFAULT_REPO, "cuda")
    model = span_model.load_adapter(model, args.adapter)
    rows = model._span_rows
    with torch.no_grad():
        for k, tok_id in enumerate(span_model.SPAN_TOKENS):
            model.encoder.embed_tokens.weight[tok_id] = rows.emb[k].to(torch.bfloat16)
            model.diffusion_head.weight[tok_id] = rows.head[k].to(torch.bfloat16)
    # Drop the hooks: the rows now live in the weight matrices.
    model.encoder.embed_tokens._forward_hooks.clear()
    model.diffusion_head._forward_hooks.clear()
    del model._span_rows

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # Config, tokenizer and remote code come from the original repo, so the
    # export loads exactly like the base model.
    snap = pathlib.Path(snapshot_download(span_model.DEFAULT_REPO))
    for f in snap.iterdir():
        if f.suffix in (".json", ".py", ".jinja", ".model", ".txt") and not f.name.endswith(".index.json"):
            shutil.copy(f, args.out / f.name)
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
