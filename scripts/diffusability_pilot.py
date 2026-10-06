"""E18a pilot: where in a generated turn could diffusion take over?

Scores every 8-token block start of the span-SFT model's own AR outputs
(held-out turns, E17 `ar` setting) with `src.diffusability`, tiles exact and
cheap blocks into spans and reports coverage by region (thinking, tool call,
final answer) plus an annotated text dump for reading.

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.diffusability_pilot --n 25
"""

import argparse
import json
import pathlib
import random

from transformers import AutoTokenizer

from scripts.eval_toolcall import items
from src import diffusability, span_model, spans

ROOT = pathlib.Path(__file__).resolve().parent.parent


def regions(tok, gen: list[int]) -> list[str]:
    """Region of each generated token: think / call / answer."""
    out, state = [], "answer"
    text = ""
    for t in gen:
        piece = tok.decode([t])
        text += piece
        if t == spans.DIFF_OPEN:
            state = "call"
        tail = text[-40:]
        if state != "call":
            state = "think" if text.count("<think>") > text.count("</think>") else "answer"
        out.append(state)
        if t == spans.DIFF_CLOSE:
            state = "answer"
        if state != "call" and "</think>" in tail and text.endswith("</think>"):
            state = "answer"
    return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="e17_span_sft")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / "sft" / args.run
    out_dir = run_dir / "diffusability"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(span_model.DEFAULT_REPO, trust_remote_code=True)
    prompts = {it["id"]: it for it in items(None, True, tok)}
    gens = [
        g for g in map(json.loads, (run_dir / "eval" / "generations.jsonl").open())
        if g["setting"] == "ar" and g["text"].endswith("<|im_end|>")
    ]  # fmt: skip
    random.Random(args.seed).shuffle(gens)
    model = span_model.load_base(span_model.DEFAULT_REPO, "cuda")
    model = span_model.load_adapter(model, run_dir / "train" / "adapter_final.pt").eval()

    rows = []
    for g in gens:
        if len(rows) >= args.n:
            break
        gen = spans.encode(tok, g["text"])
        if len(gen) != g["num_tokens"] or len(gen) < 2 * args.block:
            continue
        prompt = prompts[g["id"]]["prompt"]
        ids = prompt + gen
        sc = diffusability.score_turn(model, ids, len(prompt), args.block, args.threshold)
        rows.append({
            "id": g["id"], "gen": gen, "prompt_len": len(prompt),
            "regions": regions(tok, gen), "scores": sc.__dict__,
        })  # fmt: skip
        print(f"{len(rows)}/{args.n} {g['id'][:8]} {len(gen)} tokens", flush=True)
    (out_dir / "pilot.json").write_text(json.dumps(rows))
    print(f"saved {out_dir / 'pilot.json'}")


if __name__ == "__main__":
    main()
