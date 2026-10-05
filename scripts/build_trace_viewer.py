"""Builds a self-contained HTML viewer of span-decoding traces.

Reads `eval/generations.jsonl` of a span-SFT run (setting `spans`) and marks
each generated turn by decoding mode: AR text plain, diffusion spans (from
after `<diff>` through `</diff>`) highlighted, with the 8-token diffusion
blocks delimited where the text re-tokenizes to the recorded token count.

    uv run python -m scripts.build_trace_viewer --run e17_span_sft
"""

import argparse
import json
import pathlib
import re

from transformers import AutoTokenizer

from scripts import sft_data
from scripts.eval_toolcall import parse_calls
from src import span_model, spans

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts" / "trace_viewer.html"
OPEN, CLOSE = "<SPECIAL_18>", "<SPECIAL_19>"


def segments(text: str, ids: list[int] | None, tok, block: int) -> list:
    """[mode, text or block texts, tokens] per segment, in output order.

    mode is "ar" or "dlm"; dlm segments hold their 8-token block texts.
    Token counts are exact when `ids` is given, else estimated by characters.
    """
    out: list = []
    if ids is not None:
        # Token-exact: diffusion runs from after <diff> through </diff>.
        start = i = 0
        while i < len(ids):
            if ids[i] != spans.DIFF_OPEN:
                i += 1
                continue
            out.append(["ar", tok.decode(ids[start : i + 1]), i + 1 - start])
            j = i + 1
            while j < len(ids) and ids[j] != spans.DIFF_CLOSE:
                j += 1
            body = ids[i + 1 : j + 1]
            blocks = [
                tok.decode(body[k : k + block])
                for k in range(0, len(body), block)
            ]
            out.append(["dlm", blocks, len(body)])
            start = i = j + 1
        if start < len(ids):
            out.append(["ar", tok.decode(ids[start:]), len(ids) - start])
        return [s for s in out if s[1]]
    # Fallback: split on the marker strings (no block boundaries).
    for part in re.split(
        f"({re.escape(OPEN)}.*?{re.escape(CLOSE)}|{re.escape(OPEN)}.*$)",
        text,
        flags=re.S,
    ):
        if not part:
            continue
        if part.startswith(OPEN):
            out.append(["ar", OPEN, 0])
            out.append(["dlm", [part[len(OPEN) :]], 0])
        else:
            out.append(["ar", part, 0])
    return out


def with_counts(segs: list, ar: int, dlm: int) -> list:
    """Fills missing token counts from the per-mode totals by character share."""
    if all(c for _, _, c in segs):
        return segs
    size = [len(t if m == "ar" else "".join(t)) for m, t, _ in segs]
    total = {
        m: sum(n for (mm, _, _), n in zip(segs, size, strict=True) if mm == m)
        or 1
        for m in ("ar", "dlm")
    }
    want = {"ar": ar, "dlm": dlm}
    return [
        [m, t, round(want[m] * n / total[m])]
        for (m, t, _), n in zip(segs, size, strict=True)
    ]


def context(rec: dict, turn: int) -> list:
    """Messages since the previous assistant turn (what the model answers)."""
    asst = [
        i for i, m in enumerate(rec["messages"]) if m["role"] == "assistant"
    ]
    k = asst[turn] if turn < len(asst) else len(rec["messages"])
    prev = asst[turn - 1] if turn > 0 else -1
    return [
        {"role": m["role"], "content": m["content"][:4000]}
        for m in rec["messages"][prev + 1 : k]
    ]


def tool_text(after: str) -> str:
    """The tool / user message(s) between two assistant turns, as plain text."""
    t = re.sub(r"<\|im_start\|>(user|tool)\n", "", after)
    t = t.replace("<|im_end|>", "").replace("<|im_start|>assistant", "")
    return t.strip()


def load_rollouts(path: pathlib.Path, tok, block: int) -> list:
    """Closed-loop rollouts (scripts.rollout_toolcall) in viewer form."""
    if not path.exists():
        return []
    out = []
    for line in path.open():
        r = json.loads(line)
        turns = []
        for t in r["turns"]:
            ids = spans.encode(tok, t["text"])
            exact = ids if len(ids) == t["num_tokens"] else None
            ar = sum(int(n) for n in re.findall(r"ar(\d+)", t["segments"]))
            dlm = sum(int(n) for n in re.findall(r"dlm(\d+)", t["segments"]))
            turns.append({
                "segs": with_counts(segments(t["text"], exact, tok, block), ar, dlm),
                "calls": t["calls"], "ref_calls": t["ref_calls"],
                "match": t["match"], "names_match": t.get("names_match", t["match"]),
                "tokens": t["num_tokens"], "nfe": t["nfe"], "ar": ar, "dlm": dlm,
                "after": tool_text(t["after"])[:3000],
            })  # fmt: skip
        out.append({"id": r["id"], "user": r["user"][:3000], "turns": turns,
                    "ref_turns": r["ref_turns"], "completed": r["completed"]})  # fmt: skip
    out.sort(key=lambda x: (-len(x["turns"]), x["id"]))
    return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="e17_span_sft")
    parser.add_argument("--setting", default="spans")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / "sft" / args.run
    out = args.out or run_dir / "trace_viewer.html"

    tok = AutoTokenizer.from_pretrained(
        span_model.DEFAULT_REPO, trust_remote_code=True
    )
    recs = {
        r["uuid"]: r
        for r in sft_data.load(
            sft_data.cache_path([0], 4096), "heldout", None, 0
        )
    }
    items = []
    for line in (run_dir / "eval" / "generations.jsonl").open():
        g = json.loads(line)
        if g["setting"] != args.setting:
            continue
        ids = spans.encode(tok, g["text"])
        exact_ids = ids if len(ids) == g["num_tokens"] else None
        gen, _ = parse_calls(g["text"])
        ref, _ = parse_calls(g["reference"])
        ar = sum(int(n) for n in re.findall(r"ar(\d+)", g["segments"]))
        dlm = sum(int(n) for n in re.findall(r"dlm(\d+)", g["segments"]))
        items.append({
            "id": g["id"], "turn": g["turn"],
            "exact": gen == ref, "calls": [c[0] for c in gen],
            "ref_calls": [c[0] for c in ref], "decision": bool(gen) == bool(ref),
            "tokens": g["num_tokens"], "nfe": g["nfe"], "ar": ar, "dlm": dlm,
            "sec": round(g.get("sec", 0), 2),
            "segs": with_counts(
                segments(g["text"], exact_ids, tok, args.block_size), ar, dlm
            ),
            "reference": g["reference"],
            "context": context(recs[g["id"]], g["turn"]),
        })  # fmt: skip
    items.sort(key=lambda x: (-x["dlm"], x["id"]))
    rollouts = load_rollouts(run_dir / "rollouts" / f"{args.setting}.jsonl", tok, args.block_size)
    summary = json.loads((run_dir / "eval" / "summary.json").read_text())[
        args.setting
    ]
    data = {"run": args.run, "setting": args.setting, "block": args.block_size,
            "summary": summary, "items": items, "rollouts": rollouts}  # fmt: skip
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__TRACE_DATA__", blob)
    out.write_text(html)
    print(f"{len(items)} turns, {len(rollouts)} rollouts -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
