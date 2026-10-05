"""Closed-loop multi-turn rollouts of a span-SFT model on held-out tool use.

For each held-out conversation with >= `min_calls` tool-calling assistant
turns, the model generates assistant turns one after another on its own
history: after a turn whose calls match the reference calls, the dataset's
tool responses (synthetic, written for the reference calls) are appended
and the next turn is generated. A turn continues the rollout when its called
function names match the reference turn's (argument differences are recorded
as `match: false`); the rollout stops at the first turn calling different
functions (or answering instead of calling, or vice versa), or at the end. The result
is the model's own interleaving of AR thinking and diffusion tool calls.

    uv run python -m scripts.rollout_toolcall --run e17_span_sft --gpus 0 1 2 3
"""

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch

from scripts import sft_data
from scripts.eval_toolcall import parse_calls

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pieces(ex) -> list[tuple[str, list[int]]]:
    """Splits a rendered conversation into ("ctx" | "asst", ids) runs."""
    out: list[tuple[str, list[int]]] = []
    for tok_id, a in zip(ex.ids, ex.assistant, strict=True):
        kind = "asst" if a else "ctx"
        if out and out[-1][0] == kind:
            out[-1][1].append(tok_id)
        else:
            out.append((kind, [tok_id]))
    return out


def rollout(rec: dict, tok, generate) -> dict:
    """Rolls out one conversation with `generate(history, rid) -> dict`.

    `generate` returns token_ids (ending with <|im_end|> when the turn
    finished), segments ("ar6 dlm37 ..."), nfe and sec.
    """
    from src import spans

    ex = spans.render(tok, rec["messages"], rec["tools"], True)
    parts = pieces(ex)
    history: list[int] = []
    turns = []
    context_full = False
    for kind, ids in parts:
        if kind == "ctx":
            history += ids
            if turns:
                turns[-1]["after"] = tok.decode(ids)
            continue
        ref_text = tok.decode(ids)
        res = generate(history, f"{rec['uuid']}-t{len(turns)}")
        if res is None:  # no context left for another turn
            context_full = True
            break
        text = tok.decode(res["token_ids"])
        gen, _ = parse_calls(text)
        ref, _ = parse_calls(ref_text)
        names = sorted(c[0] for c in gen) == sorted(c[0] for c in ref)
        turns.append({
            "text": text, "reference": ref_text, "segments": res["segments"],
            "nfe": res["nfe"], "num_tokens": len(res["token_ids"]),
            "sec": res["sec"], "match": gen == ref, "names_match": names,
            "calls": [c[0] for c in gen], "ref_calls": [c[0] for c in ref],
            "after": "",
        })  # fmt: skip
        if not names:
            break
        history += res["token_ids"]
    first_user = next(
        m["content"] for m in rec["messages"] if m["role"] == "user"
    )
    n_ref = sum(1 for kind, _ in parts if kind == "asst")
    return {
        "id": rec["uuid"], "user": first_user, "turns": turns,
        "ref_turns": n_ref, "context_full": context_full,
        "completed": bool(turns) and len(turns) == n_ref and turns[-1]["names_match"],
    }  # fmt: skip


def worker(gpu: str, args, todo: list, out_file: pathlib.Path) -> None:
    """HF engine: rolls out a shard of conversations on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from transformers import AutoTokenizer

    from src import span_decode, span_model

    tok = AutoTokenizer.from_pretrained(
        span_model.DEFAULT_REPO, trust_remote_code=True
    )
    model = span_model.load_base(span_model.DEFAULT_REPO, "cuda")
    model = span_model.load_adapter(model, pathlib.Path(args.adapter)).eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def generate(history: list, rid: str) -> dict:
        t0 = time.time()
        res = span_decode.span_generate(
            model, torch.tensor([history], device="cuda"),
            eos_token_id=eos, max_new_tokens=args.max_new_tokens,
            block_size=args.block_size, threshold=args.threshold,
            mode=args.mode,
        )  # fmt: skip
        torch.cuda.synchronize()
        return {
            "token_ids": res.token_ids, "nfe": res.nfe, "sec": time.time() - t0,
            "segments": " ".join(f"{s.mode}{s.num_tokens}" for s in res.segments),
        }  # fmt: skip

    with out_file.open("a") as f:
        for rec in todo:
            f.write(json.dumps(rollout(rec, tok, generate)) + "\n")
            f.flush()


def sglang_generate(args, eos: int):
    """SGLang engine: `generate` over HTTP; costs from the server's traces.

    The server must run `launch_sglang.sh span` with TRACE_DIR=args.trace_dir.
    """

    def generate(history: list, rid: str) -> dict | None:
        budget = min(args.max_new_tokens, args.context_len - len(history))
        if budget < 64:
            return None
        trace = pathlib.Path(args.trace_dir) / f"{rid}.jsonl"
        trace.unlink(missing_ok=True)
        payload = {
            "input_ids": history, "rid": rid,
            "sampling_params": {"max_new_tokens": budget,
                                "temperature": 0, "skip_special_tokens": False},
        }  # fmt: skip
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as resp:
            out = json.load(resp)
        ids = list(out["output_ids"])
        if out["meta_info"]["finish_reason"]["type"] == "stop" and (
            not ids or ids[-1] != eos
        ):
            ids.append(eos)  # SGLang drops the matched stop token
        calls = [json.loads(line) for line in trace.open()] if trace.exists() else []
        segs: list[list] = []
        for c in calls:
            if segs and segs[-1][0] == c["mode"]:
                segs[-1][1] += c["tokens"]
            else:
                segs.append([c["mode"], c["tokens"]])
        return {
            "token_ids": ids, "sec": time.time() - t0,
            "nfe": 1 + sum(c["nfe"] for c in calls),  # + prompt prefill
            "segments": " ".join(f"{m}{n}" for m, n in segs),
        }  # fmt: skip

    return generate


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="e17_span_sft")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--mode", choices=["spans", "ar"], default="spans")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--workers-per-gpu", type=int, default=6)
    parser.add_argument("--min-calls", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--engine", choices=["hf", "sglang"], default="hf")
    parser.add_argument("--port", type=int, default=30310)
    parser.add_argument("--trace-dir", default=None, help="server TRACE_DIR")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=8192,
                        help="server context length (SGLang engine)")  # fmt: skip
    args = parser.parse_args()
    run_dir = ROOT / "runs" / "sft" / args.run
    args.adapter = args.adapter or str(run_dir / "train" / "adapter_final.pt")
    out_dir = run_dir / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.mode}.jsonl"

    recs = sft_data.load(
        sft_data.cache_path([0], 4096), "heldout", None, 0
    )
    recs = [
        r for r in recs
        if sum(1 for m in r["messages"] if m["role"] == "assistant"
               and m.get("tool_calls")) >= args.min_calls
    ][: args.limit]  # fmt: skip
    done = set()
    if out_file.exists():
        done = {json.loads(line)["id"] for line in out_file.open()}
    todo = [r for r in recs if r["uuid"] not in done]
    print(f"{len(todo)} conversations to roll out ({len(done)} done)")
    if args.engine == "sglang":
        run_sglang(args, todo, out_file)
    else:
        run_hf(args, todo, out_dir)
    rows = [json.loads(line) for line in out_file.open()]
    turns = [t for r in rows for t in r["turns"]]
    print(
        f"{len(rows)} rollouts, completed {sum(r['completed'] for r in rows)}, "
        f"mean turns {len(turns) / max(1, len(rows)):.2f} "
        f"(reference {sum(r['ref_turns'] for r in rows) / max(1, len(rows)):.2f})"
    )


def run_sglang(args, todo: list, out_file: pathlib.Path) -> None:
    """All conversations concurrently against one SGLang span server."""
    from transformers import AutoTokenizer

    from src import span_model

    tok = AutoTokenizer.from_pretrained(
        span_model.DEFAULT_REPO, trust_remote_code=True
    )
    generate = sglang_generate(args, tok.convert_tokens_to_ids("<|im_end|>"))
    lock = threading.Lock()
    t0 = time.time()
    with out_file.open("a") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(rollout, rec, tok, generate) for rec in todo]
        for n, fut in enumerate(futures, 1):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)


def run_hf(args, todo: list, out_dir: pathlib.Path) -> None:
    """HF reference engine: worker processes per GPU, one turn at a time."""
    out_file = out_dir / f"{args.mode}.jsonl"
    slots = [g for g in args.gpus for _ in range(args.workers_per_gpu)]
    ctx = mp.get_context("spawn")
    procs = []
    for i, gpu in enumerate(slots):
        shard = todo[i :: len(slots)]
        if shard:
            p = ctx.Process(
                target=worker,
                args=(gpu, args, shard, out_dir / f"{args.mode}.w{i}.jsonl"),
            )
            p.start()
            procs.append(p)
    for p in procs:
        p.join()
    with out_file.open("a") as f:
        for w in sorted(out_dir.glob(f"{args.mode}.w*.jsonl")):
            f.write(w.read_text())
            w.unlink()
    codes = [p.exitcode for p in procs]
    print("worker exit codes:", codes)
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
