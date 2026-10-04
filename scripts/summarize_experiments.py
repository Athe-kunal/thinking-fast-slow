"""Tabulates results of `scripts.run_experiments` runs.

Reads `runs/experiments/*/eval/scored.jsonl` (all runs, or those matching
`--runs` glob patterns) and prints, per benchmark, every evaluated setting
with accuracy, 95% CI, diffusion token share, tokens per forward pass and
mean answer length. With `--reference RUN:SETTING_PREFIX` it adds a paired
exact McNemar test of each row against that reference (same benchmark items).

    uv run python -m scripts.summarize_experiments
    uv run python -m scripts.summarize_experiments --runs 'e14*' \
        --reference baselines:random:0.0
"""

import argparse
import fnmatch
import json
import pathlib
import re

from scripts.eval_router_benchmarks import mcnemar, setting_of, wilson

RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs" / "experiments"


def load(patterns: list[str]) -> dict[tuple[str, str], list[dict]]:
    """Scored records keyed by (run, setting)."""
    out: dict[tuple[str, str], list[dict]] = {}
    for scored in sorted(RUNS.glob("*/eval/scored.jsonl")):
        run = scored.parent.parent.name
        if patterns and not any(fnmatch.fnmatch(run, p) for p in patterns):
            continue
        for line in scored.read_text().splitlines():
            r = json.loads(line)
            out.setdefault((run, setting_of(r)), []).append(r)
    return out


def short(setting: str) -> str:
    """Shortens learned-checkpoint paths to their file name."""
    return re.sub(r"learned:\S*/", "learned:", setting)


def main() -> None:
    """Prints the summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument(
        "--reference",
        default=None,
        help="RUN:SETTING_PREFIX for paired tests, e.g. baselines:random:0.0",
    )
    args = parser.parse_args()

    data = load(args.runs)
    ref = None
    if args.reference:
        ref_run, _, ref_prefix = args.reference.partition(":")
        ref = {
            bench: {r["id"]: r["correct"] for r in recs if r["bench"] == bench}
            for (run, setting), recs in load([ref_run]).items()
            if setting.startswith(ref_prefix)
            for bench in {r["bench"] for r in recs}
        }

    benches = sorted({r["bench"] for recs in data.values() for r in recs})
    for bench in benches:
        print(f"\n== {bench}")
        print(
            f"{'run':28} {'setting':40} {'n':>5} {'acc':>7} {'95% CI':>15} "
            f"{'dlm':>6} {'tok/fwd':>7} {'len':>5}"
            + ("  paired vs ref" if ref else "")
        )
        for (run, setting), recs in sorted(data.items()):
            rs = [r for r in recs if r["bench"] == bench]
            if not rs:
                continue
            k, n = sum(r["correct"] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            seg = " ".join(r["segments"] for r in rs)
            dlm = sum(int(c) for c in re.findall(r"dlm(\d+)", seg))
            ar = sum(int(c) for c in re.findall(r"ar(\d+)", seg))
            tokens = sum(r["num_tokens"] for r in rs)
            line = (
                f"{run[:28]:28} {short(setting)[:40]:40} {n:5d} {k / n:7.1%} "
                f"[{lo:5.1%}, {hi:5.1%}] {dlm / max(1, dlm + ar):6.1%} "
                f"{tokens / sum(r['nfe'] for r in rs):7.2f} {tokens / n:5.0f}"
            )
            if ref and bench in ref:
                x, y, p = mcnemar(
                    ref[bench], {r["id"]: r["correct"] for r in rs}
                )
                line += f"  ref-only {x:3d} / row-only {y:3d}, p={p:.3f}"
            print(line)


if __name__ == "__main__":
    main()
