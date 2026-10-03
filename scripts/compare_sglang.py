"""Compares the SGLang DLLM server against the HF reference implementation.

Sends the same GSM8K questions (greedy) to a running SGLang server and to the
HF reference code (`NemotronLabsDiffusionModel`), then reports per engine:
exact-output agreement, GSM8K accuracy and single-request throughput.

    scripts/launch_sglang.sh dlm   # in another shell, PORT=30000
    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.compare_sglang --mode dlm

`--mode` must match the mode the server was launched with.
"""

import argparse
import json
import re
import time
import urllib.request

import torch

from src import engine as engine_lib

GSM8K_ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k"
    "&config=main&split=test&offset=0&length={n}"
)
SUFFIX = "\nPut the final numeric answer after '####'."


def load_gsm8k(n: int) -> list[tuple[str, str]]:
    """Returns the first `n` GSM8K test (question, gold answer) pairs."""
    with urllib.request.urlopen(GSM8K_ROWS.format(n=n), timeout=60) as resp:
        rows = json.load(resp)["rows"]
    return [
        (r["row"]["question"], r["row"]["answer"].split("####")[-1].strip())
        for r in rows
    ]


def last_number(text: str) -> str | None:
    """Extracts the answer: the number after '####', else the last number."""
    tail = text.split("####")[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail) or re.findall(
        r"-?\d[\d,]*\.?\d*", text
    )
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def query_sglang(
    port: int, question: str, max_tokens: int
) -> tuple[str, int, float]:
    """Sends one greedy chat request. Returns (text, tokens, seconds)."""
    body = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    seconds = time.perf_counter() - start
    text = out["choices"][0]["message"]["content"]
    return text, out["usage"]["completion_tokens"], seconds


def query_hf(
    engine: engine_lib.NemotronEngine,
    mode: str,
    question: str,
    max_tokens: int,
) -> tuple[str, int, float]:
    """Runs the HF reference greedily. Returns (text, tokens, seconds)."""
    if mode in ("ar", "dlm"):
        result = engine.generate(question, mode=mode, max_new_tokens=max_tokens)
        return result.text, result.num_tokens, result.seconds

    prompt_ids = engine.encode_chat(question)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out_ids, _ = engine.model.linear_spec_generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            block_length=32,
            eos_token_id=engine.tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    new_ids = out_ids[0, prompt_ids.shape[1] :].tolist()
    text = str(engine.tokenizer.decode(new_ids, skip_special_tokens=True))
    return text, len(new_ids), seconds


def main() -> None:
    """Runs both engines on the same questions and prints a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("ar", "dlm", "linear_spec")
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default=engine_lib.DEFAULT_REPO)
    parser.add_argument("-n", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    data = load_gsm8k(args.n)
    engine = engine_lib.NemotronEngine(args.model, device="cuda")
    # Warm up both engines so first-call overheads do not skew throughput.
    query_sglang(args.port, "Hi", 32)
    query_hf(engine, args.mode, "Hi", 32)

    stats = {
        name: {"correct": 0, "tokens": 0, "seconds": 0.0}
        for name in ("sglang", "hf")
    }
    identical = 0
    for i, (question, gold) in enumerate(data):
        texts = {}
        for name in ("sglang", "hf"):
            if name == "sglang":
                text, n_tok, sec = query_sglang(
                    args.port, question + SUFFIX, args.max_tokens
                )
            else:
                text, n_tok, sec = query_hf(
                    engine, args.mode, question + SUFFIX, args.max_tokens
                )
            texts[name] = text.strip()
            stats[name]["correct"] += last_number(text) == gold
            stats[name]["tokens"] += n_tok
            stats[name]["seconds"] += sec
        identical += texts["sglang"] == texts["hf"]
        print(
            f"[{i + 1:3d}/{len(data)}] gold={gold:>8} "
            f"sglang={last_number(texts['sglang'])!s:>8} "
            f"hf={last_number(texts['hf'])!s:>8} "
            f"identical={texts['sglang'] == texts['hf']}",
            flush=True,
        )

    n = len(data)
    print(
        f"\nmode={args.mode}  questions={n}  identical outputs={identical}/{n}"
    )
    for name, s in stats.items():
        print(
            f"  {name:6s} accuracy={s['correct'] / n:6.1%}  "
            f"throughput={s['tokens'] / s['seconds']:7.1f} tok/s "
            f"(concurrency 1)"
        )


if __name__ == "__main__":
    main()
