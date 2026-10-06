"""Summarizes and annotates an E18a diffusability pilot (`pilot.json`).

Prints coverage by region and predicted tokens per forward pass for span
tilings (exact blocks within `max_passes` denoising passes, at least
`min_blocks` blocks per span), how well first-pass masked entropy vs AR
entropy separates good from bad blocks (AUC), and writes `annotated.md`
with every turn's text and its diffusion spans marked ⟦ ... ⟧, plus examples
of near-miss blocks (what diffusion wrote vs AR).

    uv run python -m scripts.diffusability_report --run e17_span_sft
"""

import argparse
import collections
import json
import pathlib
import random

from transformers import AutoTokenizer

from src import diffusability, span_model
from src.diffusability import BlockScores

ROOT = pathlib.Path(__file__).resolve().parent.parent
B = 8


def auc(good: list, bad: list) -> float:
    """P(score of a good block < score of a bad block); ties count half."""
    if not good or not bad:
        return float("nan")
    rng = random.Random(0)
    good = rng.sample(good, min(1500, len(good)))
    bad = rng.sample(bad, min(1500, len(bad)))
    wins = sum((g < b) + 0.5 * (g == b) for g in good for b in bad)
    return wins / (len(good) * len(bad))


def cost(rows: list, max_passes: int, min_blocks: int):
    """(coverage by region, overall coverage, tokens/pass, #spans, spans)."""
    total, cov = collections.Counter(), collections.Counter()
    ntok = passes = nspans = 0
    all_spans = []
    for r in rows:
        sc = BlockScores(**r["scores"])
        n, g0 = len(r["gen"]), r["prompt_len"]
        sp = diffusability.tile(n, sc, g0, B, max_passes, min_blocks)
        all_spans.append(sp)
        pmap = dict(zip(sc.starts, sc.passes, strict=True))
        inside = set()
        for a, b in sp:
            nspans += 1
            passes += 1  # the <diff> token (one AR step)
            passes += sum(pmap[s] + 1 for s in range(a, b, B))  # denoise + commit
            inside |= set(range(a, b))
        for i, region in enumerate(r["regions"]):
            total[region] += 1
            if g0 + i in inside:
                cov[region] += 1
            else:
                passes += 1
        ntok += n
    by_region = {k: cov[k] / total[k] for k in total}
    return by_region, sum(cov.values()) / ntok, ntok / passes, nspans, all_spans


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="e17_span_sft")
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--min-blocks", type=int, default=1)
    args = parser.parse_args()
    out_dir = ROOT / "runs" / "sft" / args.run / "diffusability"
    rows = json.loads((out_dir / "pilot.json").read_text())
    tok = AutoTokenizer.from_pretrained(span_model.DEFAULT_REPO, trust_remote_code=True)

    regions = collections.Counter(g for r in rows for g in r["regions"])
    print(f"{len(rows)} turns, {sum(regions.values())} generated tokens: {dict(regions)}")
    blocks = [(m, p) for r in rows for m, p in zip(r["scores"]["matches"], r["scores"]["passes"], strict=True)]  # fmt: skip
    print(f"{len(blocks)} candidate blocks: exact {sum(m == B for m, _ in blocks) / len(blocks):.1%}, "
          f"exact in <=2 passes {sum(m == B and p <= 2 for m, p in blocks) / len(blocks):.1%}, "
          f"<=3 passes {sum(m == B and p <= 3 for m, p in blocks) / len(blocks):.1%}")  # fmt: skip
    for mp in (2, 3):
        for mb in (1, 2):
            by_region, overall, tpp, nspans, _ = cost(rows, mp, mb)
            regs = ", ".join(f"{k} {v:.0%}" for k, v in sorted(by_region.items()))
            print(f"max_passes {mp}, min_blocks {mb}: {nspans} spans, covered {overall:.1%} "
                  f"({regs}); predicted tokens/pass {tpp:.2f}")  # fmt: skip

    good_m, bad_m, good_a, bad_a = [], [], [], []
    for r in rows:
        s = r["scores"]
        for m, p, me, ae in zip(s["matches"], s["passes"], s["masked_entropy"], s["ar_entropy"], strict=True):  # fmt: skip
            ok = m == B and p <= args.max_passes
            (good_m if ok else bad_m).append(me)
            (good_a if ok else bad_a).append(ae)
    print(f"AUC for 'good block' (lower score = good): first-pass masked entropy "
          f"{auc(good_m, bad_m):.3f}, AR entropy over the block {auc(good_a, bad_a):.3f}")  # fmt: skip

    _, _, _, _, all_spans = cost(rows, args.max_passes, args.min_blocks)
    lines = [f"# Diffusability pilot ({args.run}): spans = exact blocks in <= "
             f"{args.max_passes} passes, >= {args.min_blocks} block(s)\n"]  # fmt: skip
    for r, sp in zip(rows, all_spans, strict=True):
        g0, gen = r["prompt_len"], r["gen"]
        marks = {a - g0: "⟦" for a, _ in sp} | {b - g0: "⟧" for _, b in sp}
        text = "".join(marks.get(i, "") + tok.decode([t]) for i, t in enumerate(gen))
        text += marks.get(len(gen), "")
        text = text.replace("<SPECIAL_18>", "<diff>").replace("<SPECIAL_19>", "</diff>")
        lines.append(f"## {r['id'][:8]} ({len(gen)} tokens, {len(sp)} spans)\n\n```\n{text}\n```\n")
    near = []
    for r in rows:
        s = r["scores"]
        for st, m, f in zip(s["starts"], s["matches"], s["filled"], strict=True):
            if m in (6, 7):
                i = st - r["prompt_len"]
                near.append((tok.decode(r["gen"][i : i + B]), tok.decode(f)))
    random.Random(0).shuffle(near)
    lines.append("## Near misses (6-7 of 8 tokens right): AR vs diffusion\n")
    lines += [f"- AR `{a!r}`\n  DLM `{d!r}`" for a, d in near[:40]]
    (out_dir / "annotated.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'annotated.md'}")


if __name__ == "__main__":
    main()
