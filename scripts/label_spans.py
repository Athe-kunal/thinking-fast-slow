"""E18b span labels: structure + masked entropy on the dataset text.

For every conversation (train subset + held-out) the assistant text is split
into spans that span SFT will train with block diffusion:

1. Structure (always diffusion): tool calls, fenced code blocks, markdown
   tables, JSON objects, each if >= one block long. (List items are
   entropy-gated like other text.)
2. Elsewhere, 8-token blocks whose first diffusion pass on the dataset text
   is confident and right: seed = the dataset token, the other 7 positions
   masked against the true prefix; the block qualifies if the summed masked
   entropy is < `tau` nats and every masked position's argmax equals the
   dataset token. Consecutive qualifying blocks form one span.

Spans never include the turn-final <|im_end|>. Output JSONL rows hold the
marker-free token ids, the assistant mask and the span token ranges; training
inserts <diff> / </diff> with `spans.with_spans`.

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.label_spans \
        --model-run e17_span_sft --tau 2.5 --train-limit 8000
"""

import argparse
import collections
import json
import pathlib
import re
import time

import torch
from transformers import AutoTokenizer

from scripts import sft_data
from src import diffusability, span_model, spans

ROOT = pathlib.Path(__file__).resolve().parent.parent
EOS = 11  # <|im_end|>

_STRUCT = [  # (kind, pattern), in priority order after tool calls
    ("code", re.compile(r"```.*?```", re.S)),
    ("table", re.compile(r"(?:^[ \t]*\|.*\|[ \t]*\n?){2,}", re.M)),
    ("json", re.compile(r"\{[^{}]*\"[^{}]*\"\s*:[^{}]*\}")),
]
# List items are not structure: their text is often free prose, so they are
# entropy-gated like other text (E18b option 2).
_THINK = re.compile(r"<think>.*?</think>", re.S)


def char_to_tokens(pr: spans.PlainRender, c0: int, c1: int) -> tuple[int, int]:
    """Token range [a, b) of tokens lying entirely inside chars [c0, c1)."""
    toks = [i for i, (x, y) in enumerate(pr.offsets) if x >= c0 and y <= c1 and y > x]
    return (toks[0], toks[-1] + 1) if toks else (0, 0)


def structure_ranges(pr: spans.PlainRender, block: int) -> list[tuple[int, int, str]]:
    """Disjoint structured token ranges inside assistant turns."""
    out = [(a, b, "call") for a, b in pr.calls]
    for kind, pat in _STRUCT:
        for m in pat.finditer(pr.text):
            a, b = char_to_tokens(pr, *m.span())
            if b - a < block or not all(pr.assistant[a:b]) or EOS in pr.ids[a:b]:
                continue
            if any(a < y and x < b for x, y, _ in out):
                continue
            out.append((a, b, kind))
    return sorted(out)


def _batch_mask(lays: list, device: torch.device):
    """Flex BlockMask for a batch of padded span layouts (per-sample masks)."""
    from torch.nn.attention.flex_attention import create_block_mask

    noisy = torch.stack([x.noisy for x in lays]).to(device)
    block = torch.stack([x.block for x in lays]).to(device)
    start = torch.stack([x.block_start for x in lays]).to(device)
    pos = torch.stack([x.position_ids[0] for x in lays]).to(device)

    def mask_mod(b, h, q, kv):
        clean = ~noisy[b, q] & ~noisy[b, kv] & (kv <= q)
        same = noisy[b, q] & noisy[b, kv] & (block[b, q] == block[b, kv])
        prefix = noisy[b, q] & ~noisy[b, kv] & (pos[b, kv] < start[b, q])
        return clean | same | prefix

    t = noisy.shape[1]
    # _compile: evaluate the mask per 128x128 tile instead of materializing
    # the full B x T x T grid.
    return create_block_mask(mask_mod, len(lays), None, t, t, device=device, _compile=True)


def _pad(lay: spans.Layout, length: int, mask_id: int) -> spans.Layout:
    """Pads a layout to `length` with isolated noisy padding tokens."""
    extra = length - lay.input_ids.shape[1]
    if extra <= 0:
        return lay
    return spans.Layout(
        input_ids=torch.cat([lay.input_ids, torch.full((1, extra), mask_id)], 1),
        position_ids=torch.cat([lay.position_ids, torch.zeros(1, extra, dtype=torch.long)], 1),
        noisy=torch.cat([lay.noisy, torch.ones(extra, dtype=torch.bool)]),
        block=torch.cat([lay.block, torch.full((extra,), -2)]),
        block_start=torch.cat([lay.block_start, torch.zeros(extra, dtype=torch.long)]),
        ar_index=lay.ar_index, ar_target=lay.ar_target, dlm_index=lay.dlm_index,
        dlm_target=lay.dlm_target, dlm_weight=lay.dlm_weight, num_noisy=0,
    )  # fmt: skip


@torch.no_grad()
def entropy_blocks_batch(model, jobs: list, block: int, chunk: int = 2048) -> list:
    """First-pass (summed masked entropy, all argmax right) per start, for a
    batch of (ids, starts) jobs in one forward pass."""
    mask_id = model.mask_token_id
    lays = [diffusability._layout(ids, st, block, mask_id) for ids, st in jobs]
    length = max(x.input_ids.shape[1] for x in lays)
    lays = [_pad(x, length, mask_id) for x in lays]
    device = model.device
    enc = model.encoder
    ids_t = torch.cat([x.input_ids for x in lays]).to(device)
    pos_t = torch.cat([x.position_ids for x in lays]).to(device)
    with span_model.flex_attention(model):
        mask = _batch_mask(lays, device)
        h = enc.embed_tokens(ids_t)
        pe = enc.rotary_emb(h, position_ids=pos_t)
        for layer in enc.layers:
            h = layer(h, attention_mask=mask, position_ids=pos_t,
                      position_embeddings=pe, cache_position=pos_t[0], use_cache=False)  # fmt: skip
        h = enc.norm(h)
    out = []
    for b, (ids, st) in enumerate(jobs):
        if not st:
            out.append(([], []))
            continue
        n, nb = len(ids), len(st)
        idx = (n + torch.arange(nb)[:, None] * block + torch.arange(1, block)[None]).flatten()
        target = torch.tensor([ids[s + j] for s in st for j in range(1, block)])
        ent, right = [], []
        for c in range(0, len(idx), chunk):
            logits = model.diffusion_head(h[b, idx[c : c + chunk].to(device)]).float()
            logits[:, mask_id] = -float("inf")
            probs = torch.softmax(logits, -1)
            ent.append(torch.special.entr(probs).sum(-1).cpu())
            right.append(probs.argmax(-1).cpu() == target[c : c + chunk])
        out.append((
            torch.cat(ent).view(nb, block - 1).sum(-1).tolist(),
            torch.cat(right).view(nb, block - 1).all(-1).tolist(),
        ))  # fmt: skip
    return out


def tile(n: int, struct: list, good: set, assistant: list, ids: list, block: int) -> list:
    """Left-to-right spans: structure ranges, then runs of good blocks."""
    by_start = {a: (b, k) for a, b, k in struct}
    struct_starts = sorted(by_start)
    out, i = [], 0
    while i < n:
        if i in by_start:
            b, kind = by_start[i]
            out.append((i, b, kind))
            i = b
            continue
        if i in good:
            j = i
            while j in good and not any(j <= s < j + block for s in struct_starts):
                j += block
            if j > i:
                out.append((i, j, "entropy"))
                i = j
                continue
        i += 1
    merged: list = []
    for a, b, k in out:  # adjacent spans become one span
        if merged and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b, merged[-1][2] + "+" + k)
        else:
            merged.append((a, b, k))
    return merged


def prepare(tok, rec: dict, block: int) -> dict:
    """Rendering, structure ranges and candidate block starts."""
    pr = spans.render_plain(tok, rec["messages"], rec["tools"])
    n = len(pr.ids)
    struct = structure_ranges(pr, block)
    in_struct = set()
    for a, b, _ in struct:
        in_struct |= set(range(a, b))
    starts = [
        s for s in range(n - block + 1)
        if all(pr.assistant[s : s + block]) and EOS not in pr.ids[s : s + block]
        and not in_struct.intersection(range(s, s + block))
    ]  # fmt: skip
    return {"rec": rec, "pr": pr, "struct": struct, "starts": starts}


def finish(job: dict, ent: list, right: list, tau: float, block: int) -> dict:
    """Spans and stats for one prepared conversation."""
    rec, pr, struct, starts = job["rec"], job["pr"], job["struct"], job["starts"]
    n = len(pr.ids)
    good = {s for s, e, r in zip(starts, ent, right, strict=True) if e < tau and r}
    sp = tile(n, struct, good, pr.assistant, pr.ids, block)
    think = set()
    for m in _THINK.finditer(pr.text):
        a, b = char_to_tokens(pr, *m.span())
        think |= set(range(a, b))
    call_tok = set()
    for a, b in pr.calls:
        call_tok |= set(range(a, b))
    region = ["call" if i in call_tok else "think" if i in think else "answer" for i in range(n)]
    covered = set()
    kinds = collections.Counter()
    for a, b, k in sp:
        covered |= set(range(a, b))
        for part in k.split("+"):
            kinds[part] += 1
    tot, cov = collections.Counter(), collections.Counter()
    for i in range(n):
        if pr.assistant[i]:
            tot[region[i]] += 1
            cov[region[i]] += i in covered
    return {
        "uuid": rec["uuid"], "split": rec["split"], "ids": pr.ids,
        "assistant": pr.assistant, "spans": [[a, b] for a, b, _ in sp],
        "kinds": dict(kinds), "tokens": dict(tot), "covered": dict(cov),
    }  # fmt: skip


def label(model, tok, rec: dict, tau: float, block: int) -> dict:
    """Spans and stats for one conversation (unbatched)."""
    job = prepare(tok, rec, block)
    ((ent, right),) = entropy_blocks_batch(model, [(job["pr"].ids, job["starts"])], block)
    return finish(job, ent, right, tau, block)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", default="e17_span_sft")
    parser.add_argument("--tau", type=float, default=2.5)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shards", type=int, nargs="+", default=[0],
                        help="shard indices this process labels")  # fmt: skip
    parser.add_argument("--token-budget", type=int, default=49152,
                        help="padded tokens per batched forward")  # fmt: skip
    args = parser.parse_args()
    out = args.out or ROOT / "runs" / "sft_cache" / f"labels_e18b_tau{args.tau}_listgated.jsonl"

    path = sft_data.cache_path([0], 4096)
    recs = sft_data.load(path, "train", args.train_limit, args.seed)
    recs += sft_data.load(path, "heldout", None, 0)
    done = set()
    for f in [out, *out.parent.glob(out.name + ".shard*")]:
        if f.exists():
            done |= {json.loads(line)["uuid"] for line in f.open()}
    todo = [r for r in recs if r["uuid"] not in done]
    if args.num_shards > 1:  # deterministic split; each shard has its own file
        todo = [r for k, r in enumerate(todo) if k % args.num_shards in args.shards]
        out = out.with_name(out.name + ".shard" + "-".join(map(str, args.shards)))
    print(f"{len(todo)} conversations to label ({len(done)} done) -> {out}", flush=True)

    tok = AutoTokenizer.from_pretrained(span_model.DEFAULT_REPO, trust_remote_code=True)
    model = span_model.load_base(span_model.DEFAULT_REPO, "cuda")
    adapter = ROOT / "runs" / "sft" / args.model_run / "train" / "adapter_final.pt"
    model = span_model.load_adapter(model, adapter).eval()
    t0 = time.time()
    jobs = [prepare(tok, r, args.block) for r in todo]
    print(f"prepared {len(jobs)} in {time.time() - t0:.0f}s", flush=True)

    def layout_len(j):  # clean tokens + one block per start, padded to 1024
        return -(-(len(j["pr"].ids) + args.block * len(j["starts"])) // 1024) * 1024

    jobs.sort(key=layout_len)
    batches, cur = [], []
    for j in jobs:  # padded batch tokens <= budget
        if cur and layout_len(j) * (len(cur) + 1) > args.token_budget:
            batches.append(cur)
            cur = []
        cur.append(j)
    if cur:
        batches.append(cur)
    done_n = 0
    with out.open("a") as f:
        for batch in batches:
            res = entropy_blocks_batch(
                model, [(j["pr"].ids, j["starts"]) for j in batch], args.block
            )
            for j, (ent, right) in zip(batch, res, strict=True):
                f.write(json.dumps(finish(j, ent, right, args.tau, args.block)) + "\n")
            f.flush()
            done_n += len(batch)
            print(f"{done_n}/{len(jobs)} (batch {len(batch)}) {time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(line) for line in out.open()]
    tot, cov, kinds = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        tot.update(r["tokens"]), cov.update(r["covered"]), kinds.update(r["kinds"])
    print("coverage:", {k: f"{cov[k] / tot[k]:.1%}" for k in tot},
          f"overall {sum(cov.values()) / sum(tot.values()):.1%}", "| spans by kind:", dict(kinds))  # fmt: skip


if __name__ == "__main__":
    main()
