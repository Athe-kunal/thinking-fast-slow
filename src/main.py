"""CLI for Nemotron-Labs-Diffusion with runtime AR <-> diffusion switching.

    uv run python -m src.main                         # interactive chat
    uv run python -m src.main --prompt "Hi" --compare # both modes, one prompt

Interactive commands: :ar  :dlm  :mix  :compare  :reset  :q
"""

import argparse
from typing import Any

import torch

from src import engine as engine_lib
from src import router as router_lib

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def format_stats(result: engine_lib.GenerationResult) -> str:
    """Formats a one-line summary of a generation result."""
    return (
        f"[{result.mode}] {result.num_tokens} tok, NFE={result.nfe}, "
        f"{result.tokens_per_forward:.2f} tok/fwd, "
        f"{result.tokens_per_second:.1f} tok/s"
        + (f", segments: {result.segments}" if result.segments else "")
    )


def parse_args() -> argparse.Namespace:
    """Parses command-line flags."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=engine_lib.DEFAULT_REPO)
    parser.add_argument("--mode", default="ar", choices=engine_lib.MODES)
    parser.add_argument(
        "--prompt", help="Single-shot prompt; omit for interactive chat."
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run every mode on --prompt."
    )
    parser.add_argument(
        "--router", default="entropy", choices=router_lib.ROUTERS
    )
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--ar-chunk", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPES)
    return parser.parse_args()


def run_once(
    engine: engine_lib.NemotronEngine,
    args: argparse.Namespace,
    gen_kwargs: dict[str, Any],
) -> None:
    """Answers `args.prompt` in one or all modes and prints the replies."""
    modes = engine_lib.MODES if args.compare else (args.mode,)
    for mode in modes:
        result = engine.generate(args.prompt, mode=mode, **gen_kwargs)
        print(f"\n=== {format_stats(result)} ===\n{result.text}")


def run_chat(
    engine: engine_lib.NemotronEngine, gen_kwargs: dict[str, Any]
) -> None:
    """Runs an interactive multi-turn chat with runtime mode switching."""
    print(
        f"Ready (mode={engine.mode}). "
        "Commands: :ar :dlm :mix :compare :reset :q"
    )
    history: engine_lib.Messages = []
    compare = False
    while True:
        label = "compare" if compare else engine.mode
        try:
            user = input(f"\nUser [{label}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not user:
            continue
        if user in (":q", ":quit"):
            return
        if user == ":reset":
            history.clear()
        elif user == ":compare":
            compare = True
        elif user in (":ar", ":dlm", ":mix"):
            compare = False
            engine.set_mode(user[1:])
        else:
            history.append({"role": "user", "content": user})
            modes = engine_lib.MODES if compare else (engine.mode,)
            for mode in modes:
                result = engine.generate(history, mode=mode, **gen_kwargs)
                print(f"\n{format_stats(result)}\n{result.text}")
            # In compare mode the last mode's reply is kept as history.
            history.append({"role": "assistant", "content": result.text})


def main() -> None:
    """Entry point."""
    args = parse_args()
    engine = engine_lib.NemotronEngine(
        args.model,
        device=args.device,
        dtype=DTYPES[args.dtype],
        mode=args.mode,
        router=router_lib.make_router(args.router, args.entropy_threshold),
        ar_chunk=args.ar_chunk,
    )
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "threshold": args.threshold,
    }
    if args.prompt:
        run_once(engine, args, gen_kwargs)
    else:
        run_chat(engine, gen_kwargs)


if __name__ == "__main__":
    main()
