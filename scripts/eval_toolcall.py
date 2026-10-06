"""Held-out tool-calling evaluation for span SFT.

Each held-out conversation contributes one assistant turn (chosen by a hash
of its uuid): the prompt is the conversation up to that turn (true history,
rendered with or without <diff> markers), the model generates the turn, and
the tool calls in it are compared with the reference turn:

    exact     same multiset of calls (function name + parameter values)
    names     same multiset of function names
    decision  calls iff the reference calls (vs answering in text)
    parsed    every <tool_call> in the output parses

plus tokens / forward pass and the diffusion token share.

    uv run python -m scripts.eval_toolcall --out runs/sft/<run>/eval \
        --setting spans --adapter runs/sft/<run>/train/adapter_final.pt --gpus 2
"""

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import pathlib
import re
import time

import torch

from scripts import sft_data
from scripts.eval_router_benchmarks import wilson

_FUNC = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text: str) -> tuple[list, bool]:
    """(sorted calls, all parsed) from a turn.

    Calls are `<function=name> <parameter=k> v </parameter> </function>`
    blocks, with or without a `<tool_call>` wrapper (the base model often
    omits it).
    """
    funcs = _FUNC.findall(text)
    ok = text.count("<function=") == len(funcs)
    calls = []
    for name, body in funcs:
        params = sorted((k.strip(), v.strip()) for k, v in _PARAM.findall(body))
        calls.append((name.strip(), tuple(params)))
    return sorted(calls), ok


def items(
    limit: int | None, diff_spans: bool, tok, span_labels: str | None = None
) -> list[dict]:
    """Prompt / reference pairs for the held-out split.

    With `span_labels` (scripts.label_spans output) conversations are
    rendered with those spans, as in E18b training.
    """
    from src import spans

    path = sft_data.cache_path([0], 4096)
    labeled = spans.load_labels(span_labels) if span_labels else None
    out = []
    for rec in sft_data.load(path, "heldout", None, 0):
        if labeled is not None:
            ex = labeled[rec["uuid"]]
        else:
            ex = spans.render(tok, rec["messages"], rec["tools"], diff_spans)
        starts = [
            i
            for i in range(len(ex.ids))
            if ex.assistant[i] and not ex.assistant[i - 1]
        ]
        k = int(hashlib.sha1(rec["uuid"].encode()).hexdigest(), 16) % len(
            starts
        )
        s = starts[k]
        e = s
        while e < len(ex.ids) and ex.assistant[e]:
            e += 1
        out.append({"id": rec["uuid"], "turn": k, "prompt": ex.ids[:s],
                    "reference": tok.decode(ex.ids[s:e], skip_special_tokens=False)})  # fmt: skip
    return out[:limit] if limit else out


def worker(
    rank: int,
    gpu: str,
    args: argparse.Namespace,
    todo: list,
    out_file: pathlib.Path,
) -> None:
    """Generates for a shard of items on one GPU."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from transformers import AutoTokenizer

    from src import span_decode, span_model

    tok = AutoTokenizer.from_pretrained(
        span_model.DEFAULT_REPO, trust_remote_code=True
    )
    model = span_model.load_base(span_model.DEFAULT_REPO, "cuda")
    if args.adapter:
        model = span_model.load_adapter(model, pathlib.Path(args.adapter))
    model.eval()
    with out_file.open("a") as f:
        for it in todo:
            t0 = time.time()
            res = span_decode.span_generate(
                model, torch.tensor([it["prompt"]], device="cuda"),
                eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                max_new_tokens=args.max_new_tokens, block_size=args.block_size,
                threshold=args.threshold, mode=args.mode,
            )  # fmt: skip
            torch.cuda.synchronize()
            sec = time.time() - t0
            text = tok.decode(res.token_ids, skip_special_tokens=False)
            f.write(json.dumps({
                "id": it["id"], "setting": args.setting, "turn": it["turn"],
                "text": text, "reference": it["reference"], "nfe": res.nfe,
                "num_tokens": len(res.token_ids), "sec": sec,
                "segments": " ".join(f"{s.mode}{s.num_tokens}" for s in res.segments),
            }) + "\n")  # fmt: skip
            f.flush()


def score(path: pathlib.Path) -> dict:
    """Aggregates generation records into per-setting metrics."""
    rows = collections.defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        gen, ok = parse_calls(r["text"])
        ref, _ = parse_calls(r["reference"])
        r.update(
            exact=gen == ref,
            names=sorted(c[0] for c in gen) == sorted(c[0] for c in ref),
            decision=bool(gen) == bool(ref),
            parsed=ok,
            ref_calls=bool(ref),
            closed=r["text"].rstrip().endswith("<|im_end|>"),
        )
        rows[r["setting"]].append(r)
    summary = {}
    for setting, rs in rows.items():
        n = len(rs)
        seg = " ".join(r["segments"] for r in rs)
        dlm = sum(int(c) for c in re.findall(r"dlm(\d+)", seg))
        ar = sum(int(c) for c in re.findall(r"ar(\d+)", seg))
        calls = [r for r in rs if r["ref_calls"]]
        k = sum(r["exact"] for r in rs)
        summary[setting] = {
            "n": n,
            "exact": k / n,
            "exact_ci": wilson(k, n),
            "exact_call_turns": sum(r["exact"] for r in calls)
            / max(1, len(calls)),
            "n_call_turns": len(calls),
            "names": sum(r["names"] for r in rs) / n,
            "decision": sum(r["decision"] for r in rs) / n,
            "parsed": sum(r["parsed"] for r in rs) / n,
            "closed": sum(r["closed"] for r in rs) / n,
            "dlm_share": dlm / max(1, dlm + ar),
            "tokens_per_forward": sum(r["num_tokens"] for r in rs)
            / sum(r["nfe"] for r in rs),
            "mean_tokens": sum(r["num_tokens"] for r in rs) / n,
            "tokens_per_sec": (
                sum(r["num_tokens"] for r in rs) / sum(r["sec"] for r in rs)
                if all("sec" in r for r in rs)
                else None
            ),
        }
    return summary


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--setting", required=True, help="label of this setting"
    )
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--mode", choices=["spans", "ar"], default="spans")
    parser.add_argument(
        "--diff-spans", action="store_true", help="prompts carry <diff> markers"
    )
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--span-labels", default=None, help="E18b label file for prompts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gen_file = args.out / "generations.jsonl"

    if not args.score_only:
        from transformers import AutoTokenizer

        from src import span_model

        tok = AutoTokenizer.from_pretrained(
            span_model.DEFAULT_REPO, trust_remote_code=True
        )
        done = set()
        if gen_file.exists():
            done = {
                (r["id"], r["setting"])
                for r in map(json.loads, gen_file.open())
            }
        todo = [
            it
            for it in items(args.limit, args.diff_spans, tok, args.span_labels)
            if (it["id"], args.setting) not in done
        ]
        slots = [g for g in args.gpus for _ in range(args.workers_per_gpu)]
        shards = [todo[i :: len(slots)] for i in range(len(slots))]
        ctx = mp.get_context("spawn")
        procs = []
        for rank, (gpu, shard) in enumerate(zip(slots, shards, strict=True)):
            if shard:
                p = ctx.Process(
                    target=worker,
                    args=(
                        rank,
                        gpu,
                        args,
                        shard,
                        args.out / f"worker{rank}.jsonl",
                    ),
                )
                p.start()
                procs.append(p)
        for p in procs:
            p.join()
        codes = [p.exitcode for p in procs]
        print("worker exit codes:", codes)
        with gen_file.open("a") as f:
            for w in sorted(args.out.glob("worker*.jsonl")):
                f.write(w.read_text())
                w.unlink()
        if any(codes):
            raise SystemExit(1)
    summary = score(gen_file)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for setting, s in sorted(summary.items()):
        lo, hi = s["exact_ci"]
        print(f"{setting:24} n={s['n']:4d} exact {s['exact']:.1%} [{lo:.1%}, {hi:.1%}] "
              f"call-turns {s['exact_call_turns']:.1%} decision {s['decision']:.1%} parsed {s['parsed']:.1%} "
              f"dlm {s['dlm_share']:.1%} tok/fwd {s['tokens_per_forward']:.2f} len {s['mean_tokens']:.0f}")  # fmt: skip


if __name__ == "__main__":
    main()
