"""Diffusability of every position of a generated turn.

For each candidate start s in the generated text, simulates what the span
decoder would do if a diffusion block started there: the KV cache holds the
true prefix (tokens < s), position 0 of the block is seeded with the AR token
at s, the other positions start masked and are filled by confidence-threshold
unmasking (most confident mask plus every mask >= threshold per pass). All
candidate blocks of a turn are denoised together: each iteration is one
forward pass of the span-SFT layout [clean sequence ; all noisy blocks], whose
mask gives every block exactly the decoder's view (own block bidirectional,
clean prefix before the block start).

Per start the result holds the denoising passes used, how many of the block's
tokens equal the AR tokens, the block's first-pass masked entropy (summed over
its masked positions) and the AR next-token entropies over the same positions.
"""

import dataclasses

import torch

from src import span_model, spans


@dataclasses.dataclass
class BlockScores:
    """Per-start results (index i <-> start position starts[i])."""

    starts: list[int]
    passes: list[int]  # denoising passes until the block was full
    matches: list[int]  # tokens equal to the AR tokens (incl. the seed)
    masked_entropy: list[float]  # first pass, summed over masked positions
    ar_entropy: list[float]  # AR entropies over the same positions, summed
    filled: list[list[int]]  # the block diffusion produced


def _layout(ids: list[int], starts: list[int], block: int, mask_id: int) -> spans.Layout:
    """Clean `ids` plus one seeded, otherwise masked block per start."""
    n = len(ids)
    tok, pos = list(ids), list(range(n))
    noisy, blk, bstart = [0] * n, [-1] * n, [0] * n
    for k, s in enumerate(starts):
        for j in range(block):
            tok.append(ids[s] if j == 0 else mask_id)
            pos.append(s + j)
            noisy.append(1), blk.append(k), bstart.append(s)
    pad = -(-len(tok) // 1024) * 1024 - len(tok)
    tok += [mask_id] * pad
    pos += [0] * pad
    noisy += [1] * pad
    blk += [-2] * pad
    bstart += [0] * pad
    empty = torch.zeros(0, dtype=torch.long)
    return spans.Layout(
        input_ids=torch.tensor([tok]), position_ids=torch.tensor([pos]),
        noisy=torch.tensor(noisy, dtype=torch.bool), block=torch.tensor(blk),
        block_start=torch.tensor(bstart), ar_index=empty, ar_target=empty,
        dlm_index=empty, dlm_target=empty,
        dlm_weight=torch.zeros(0), num_noisy=0,
    )  # fmt: skip


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy in nats; safe for -inf logits (0 log 0 = 0)."""
    return torch.special.entr(torch.softmax(logits.float(), -1)).sum(-1)


@torch.no_grad()
def score_turn(
    model,
    ids: list[int],
    gen_start: int,
    block: int = 8,
    threshold: float = 0.9,
    stride: int = 1,
) -> BlockScores:
    """Scores every block start in `ids[gen_start:]` (full blocks only)."""
    mask_id = model.mask_token_id
    starts = list(range(gen_start, len(ids) - block + 1, stride))
    lay = _layout(ids, starts, block, mask_id)
    n = len(ids)
    nb = len(starts)
    noisy_idx = torch.arange(n, n + nb * block).view(nb, block)
    passes = torch.zeros(nb, dtype=torch.long)
    m_ent = torch.zeros(nb)
    head = model.diffusion_head
    device = model.device
    with span_model.flex_attention(model):
        for it in range(block):
            cur = lay.input_ids[0, noisy_idx]  # (nb, block)
            masked = cur == mask_id
            todo = masked.any(-1)
            if not todo.any():
                break
            h = span_model.span_hidden(model, lay)
            logits = head(h[noisy_idx.flatten().to(device)]).float().view(nb, block, -1)
            logits[..., mask_id] = -float("inf")
            probs = logits.softmax(-1)
            conf, x0 = probs.max(-1)
            conf, x0 = conf.cpu(), x0.cpu()
            if it == 0:
                ent = _entropy(logits).cpu()
                m_ent = (ent * masked).sum(-1)
                # AR entropies from the clean half of the same pass.
                ar_ent = _entropy(head(h[:n]).float()).cpu()
            conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
            # Unmask: most confident mask plus all masks >= threshold.
            take = (conf >= threshold) & masked
            top = conf.argmax(-1)
            take[torch.arange(nb), top] |= masked[torch.arange(nb), top]
            take &= todo.unsqueeze(-1)
            new = torch.where(take, x0, cur)
            lay.input_ids[0, noisy_idx] = new
            passes += todo.long()
    final = lay.input_ids[0, noisy_idx]
    ref = torch.tensor([ids[s : s + block] for s in starts])
    # AR entropy of predicting token s+j (logits at s+j-1), masked positions j>=1.
    ar_sum = [float(sum(ar_ent[s + j - 1] for j in range(1, block))) for s in starts]
    return BlockScores(
        starts=starts,
        passes=passes.tolist(),
        matches=(final == ref).sum(-1).tolist(),
        masked_entropy=m_ent.tolist(),
        ar_entropy=ar_sum,
        filled=final.tolist(),
    )


def tile(
    n_gen: int,
    scores: BlockScores,
    gen_start: int,
    block: int = 8,
    max_passes: int = 3,
    min_blocks: int = 1,
) -> list[tuple[int, int]]:
    """Greedy left-to-right spans of exact, cheap blocks.

    A block at s qualifies if diffusion reproduces the AR tokens exactly in at
    most `max_passes` denoising passes. Consecutive qualifying blocks form one
    span; spans shorter than `min_blocks` blocks are dropped. Returns
    [start, end) token ranges (absolute positions).
    """
    good = {
        s
        for s, p, m in zip(scores.starts, scores.passes, scores.matches, strict=True)
        if m == block and p <= max_passes
    }
    out, i, end = [], gen_start, gen_start + n_gen
    while i < end:
        if i in good:
            j = i
            while j in good and j + block <= end:
                j += block
            if (j - i) // block >= min_blocks:
                out.append((i, j))
                i = j
                continue
        i += 1
    return out
