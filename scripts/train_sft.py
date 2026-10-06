"""Cold-start span SFT: LoRA + span-token rows, mixed AR / diffusion loss.

Run through `scripts.run_sft` (which writes the resolved config); directly:

    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.train_sft \
        --config runs/sft/<run>/config.yaml --out runs/sft/<run>/train

Writes `metrics.jsonl` (per optimizer step) and adapter checkpoints
(`adapter_stepNNNN.pt`, `adapter_final.pt`: LoRA + span rows only).
"""

import argparse
import json
import math
import pathlib
import time

import torch
import yaml
from transformers import AutoTokenizer

from scripts import sft_data
from src import span_model, spans

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    d, lo, t, lr = cfg["data"], cfg["loss"], cfg["train"], cfg["lora"]
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "adapter_final.pt"
    if final.exists():
        print("already trained")
        return
    torch.manual_seed(t["seed"])

    tok = AutoTokenizer.from_pretrained(
        span_model.DEFAULT_REPO, trust_remote_code=True
    )
    path = sft_data.build(d["shards"], d["max_len"], d["heldout_percent"])
    recs = sft_data.load(path, "train", d["train_limit"], t["seed"])
    if d["span_labels"]:  # E18b: spans from scripts.label_spans
        labeled = spans.load_labels(ROOT / d["span_labels"])
        examples = [labeled[r["uuid"]] for r in recs]
    else:
        examples = [
            spans.render(tok, r["messages"], r["tools"], d["diff_spans"])
            for r in recs
        ]
    print(f"{len(examples)} training conversations")

    model = span_model.load_base(span_model.DEFAULT_REPO, "cuda")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    pmodel, rows = span_model.add_adapters(
        model, tok, lr["rank"], lr["alpha"]
    )
    pmodel.train()
    params = [p for p in pmodel.parameters() if p.requires_grad]
    row_params = list(rows.parameters())
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M"
          f" + {sum(p.numel() for p in row_params)} span-row values")  # fmt: skip
    opt = torch.optim.AdamW(
        [{"params": params}, {"params": row_params, "lr": t["rows_lr"]}],
        lr=t["lr"],
        weight_decay=0.0,
    )

    batch = t["batch"]
    steps = t["epochs"] * len(examples) // batch
    warmup = t["warmup"]

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        frac = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator().manual_seed(t["seed"])
    order = torch.cat(
        [
            torch.randperm(len(examples), generator=gen)
            for _ in range(t["epochs"])
        ]
    )
    metrics = (args.out / "metrics.jsonl").open("a")
    with span_model.flex_attention(pmodel):
        for step in range(steps):
            t0 = time.time()
            idx = order[step * batch : (step + 1) * batch].tolist()
            lays = [
                spans.layout(examples[i], lo["block_size"], model.mask_token_id, gen,
                             ar_in_spans=lo["ar_in_spans"])  # fmt: skip
                for i in idx
            ]
            ar_norm = max(1, sum(len(x.ar_index) for x in lays))
            dlm_norm = max(1, sum(x.num_noisy for x in lays))
            tot: dict[str, float] = {}
            for lay in lays:
                loss, stats = span_model.span_loss(
                    pmodel,
                    lay,
                    ar_norm,
                    dlm_norm,
                    lo["ar_weight"],
                    lo["dlm_weight"],
                )
                loss.backward()
                for k, v in stats.items():
                    tot[k] = tot.get(k, 0.0) + v
            # Clipped per group: the span rows sit in every softmax and
            # would otherwise dominate the global norm and shrink LoRA.
            gnorm = torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            rnorm = torch.nn.utils.clip_grad_norm_(row_params, t["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            row = {
                "step": step + 1,
                "ar_loss": tot.get("ar_sum", 0) / max(1, tot.get("ar_n", 0)),
                "dlm_loss": tot.get("dlm_sum", 0) / max(1, tot.get("dlm_n", 0)),
                "dlm_ce": tot.get("dlm_ce_sum", 0)
                / max(1, tot.get("dlm_masked", 0)),
                "ar_tokens": tot.get("ar_n", 0),
                "noisy_tokens": tot.get("dlm_n", 0),
                "grad_norm": float(gnorm),
                "rows_grad_norm": float(rnorm),
                "lr": sched.get_last_lr()[0],
                "sec": time.time() - t0,
            }
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            if (step + 1) % 10 == 0 or step == 0:
                print(json.dumps(row), flush=True)
            if (step + 1) % t["save_every"] == 0:
                span_model.save(
                    args.out / f"adapter_step{step + 1:04d}.pt",
                    pmodel,
                    rows,
                    {"step": step + 1},
                )
    span_model.save(final, pmodel, rows, {"step": steps})
    print("done")


if __name__ == "__main__":
    main()
