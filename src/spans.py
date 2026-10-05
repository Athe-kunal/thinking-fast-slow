"""Diffusion spans: data layout and mixed AR/diffusion loss for span SFT.

Assistant turns are trained with two objectives on one backbone:

    AR loss         next-token loss on the clean (causal) sequence, for every
                    assistant token outside <diff> spans (including the <diff>
                    token itself: AR decides when to switch) and, optionally,
                    for tokens inside spans too.
    diffusion loss  Nemotron/LLaDA masked-token loss (weighted 1/t) inside
                    <diff> ... </diff> spans, in blocks of `block_size` tokens
                    that start at the span start (as the decoder does).

One training example is laid out as [clean sequence ; noisy span blocks]:

    clean token i  attends clean tokens j <= i (causal).
    noisy token    attends noisy tokens of its own block (bidirectional) and
                   clean tokens before the block start (the causal prefix the
                   decoder holds in its KV cache when it denoises the block).

Noisy blocks reuse the positions of the clean tokens they cover; the last
block of a span is padded to `block_size` with `</diff>` targets, so the model
learns to close a span inside a block (the decoder truncates after the first
`</diff>`).
"""

import dataclasses
import json
import re

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

# Reserved single-id tokens of the Nemotron tokenizer used as span markers.
DIFF_OPEN = 18  # <SPECIAL_18> -> <diff>
DIFF_CLOSE = 19  # <SPECIAL_19> -> </diff>
# Special tokens the base model never trained; tokenized as plain text.
PLAIN_MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>",
                 "<tool_response>", "</tool_response>")  # fmt: skip

_ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)
_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.S)


@dataclasses.dataclass
class Example:
    """A tokenized conversation.

    Attributes:
        ids: Token ids.
        assistant: 1 for tokens of assistant turns (loss region).
        span: 1 for diffusion targets: tokens inside <diff> spans and the
            closing </diff> (the opening <diff> is an AR target).
    """

    ids: list[int]
    assistant: list[int]
    span: list[int]


def to_chat(record: dict) -> tuple[list[dict], list[dict]]:
    """Converts a Nemotron post-training record to (messages, tools)."""
    tools = json.loads(record["metadata"])["tools"]
    messages = []
    for m in record["messages"]:
        m = {"role": m["role"], "content": m["content"] or "",
             "tool_calls": m.get("tool_calls") or []}  # fmt: skip
        if m["tool_calls"]:
            m["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": c["function"]["name"],
                        "arguments": json.loads(c["function"]["arguments"]),
                    },
                }
                for c in m["tool_calls"]
            ]
        else:
            del m["tool_calls"]
        messages.append(m)
    return messages, tools


def plain_tokenizer(tokenizer):
    """Backend tokenizer that spells `PLAIN_MARKERS` as ordinary text.

    The base model never trained the special-token rows of <think>,
    <tool_call>, ... (their embedding / LM-head rows are at initialization);
    it writes these markers as text pieces, so training data must too.
    <|im_start|>, <|im_end|> and the reserved span tokens stay special.
    Built once per tokenizer object.
    """
    plain = getattr(tokenizer, "_plain_markers_tokenizer", None)
    if plain is None:
        import tokenizers

        spec = json.loads(tokenizer.backend_tokenizer.to_str())
        spec["added_tokens"] = [
            t for t in spec["added_tokens"] if t["content"] not in PLAIN_MARKERS
        ]
        plain = tokenizers.Tokenizer.from_str(json.dumps(spec))
        tokenizer._plain_markers_tokenizer = plain
    return plain


def encode(tokenizer, text: str) -> list[int]:
    """Token ids of `text` with markers spelled as text."""
    return plain_tokenizer(tokenizer).encode(text, add_special_tokens=False).ids


def render(tokenizer, messages: list, tools: list, diff_spans: bool) -> Example:
    """Tokenizes a conversation, marking assistant turns and tool-call spans.

    The chat-template text is cut at assistant-turn and tool-call boundaries
    and each segment is tokenized separately (clean span boundaries). With
    `diff_spans`, every `<tool_call> ... </tool_call>` inside an assistant
    turn is wrapped as `<diff><tool_call> ... </tool_call></diff>`.
    """
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    segments = []  # (start, end, assistant, call)
    pos = 0
    for m in _ASSISTANT.finditer(text):
        a, b = m.span(1)
        segments.append((pos, a, 0, 0))
        cur = a
        for c in _TOOL_CALL.finditer(text, a, b):
            segments += [(cur, c.start(), 1, 0), (c.start(), c.end(), 1, 1)]
            cur = c.end()
        segments.append((cur, b, 1, 0))
        pos = b
    segments.append((pos, len(text), 0, 0))
    ids, assistant, span = [], [], []
    for start, end, asst, call in segments:
        if start == end:
            continue
        toks = encode(tokenizer, text[start:end])
        in_span = int(call and diff_spans)
        if in_span:
            ids.append(DIFF_OPEN), assistant.append(1), span.append(0)
        ids += toks
        assistant += [asst] * len(toks)
        span += [in_span] * len(toks)
        if in_span:
            ids.append(DIFF_CLOSE), assistant.append(1), span.append(1)
    return Example(ids, assistant, span)


@dataclasses.dataclass
class Layout:
    """Tensors for one training forward pass (batch size 1).

    Attributes:
        input_ids: (1, T) clean tokens, then noisy block tokens, then padding.
        position_ids: (1, T) RoPE positions.
        noisy: (T,) bool, token belongs to a noisy block (or padding).
        block: (T,) block id of noisy tokens (-1 for clean, -2 for padding).
        block_start: (T,) position of the first token of a noisy token's block.
        ar_index: positions i of the clean sequence whose logits are trained
            to predict token i + 1.
        ar_target: the targets for `ar_index`.
        dlm_index: indices (into T) of masked noisy tokens.
        dlm_target: the targets for `dlm_index`.
        dlm_weight: 1 / t of the block of each masked noisy token.
        num_noisy: noisy block positions (diffusion loss normalizer).
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    noisy: torch.Tensor
    block: torch.Tensor
    block_start: torch.Tensor
    ar_index: torch.Tensor
    ar_target: torch.Tensor
    dlm_index: torch.Tensor
    dlm_target: torch.Tensor
    dlm_weight: torch.Tensor
    num_noisy: int


def span_runs(span: list[int]) -> list[tuple[int, int]]:
    """[start, end) of each run of span tokens."""
    runs, start = [], None
    for i, s in enumerate([*span, 0]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    return runs


def layout(
    ex: Example,
    block_size: int,
    mask_id: int,
    generator: torch.Generator,
    ar_in_spans: bool = True,
    eps: float = 1e-3,
    pad_to: int = 1024,
) -> Layout:
    """Builds the clean + noisy layout of one example (CPU tensors)."""
    n = len(ex.ids)
    ids = list(ex.ids)
    pos = list(range(n))
    noisy, block, start = [0] * n, [-1] * n, [0] * n
    dlm_index, dlm_target, dlm_weight = [], [], []
    num_noisy = 0
    block_id = 0
    for s, e in span_runs(ex.span):
        for b0 in range(s, e, block_size):
            t = float(torch.rand(1, generator=generator)) * (1 - eps) + eps
            masks = torch.rand(block_size, generator=generator) < t
            for k in range(block_size):
                p = b0 + k
                target = ex.ids[p] if p < e else DIFF_CLOSE
                if masks[k]:
                    dlm_index.append(len(ids))
                    dlm_target.append(target)
                    dlm_weight.append(1.0 / t)
                ids.append(mask_id if masks[k] else target)
                pos.append(p)
                noisy.append(1), block.append(block_id), start.append(b0)
            block_id += 1
            num_noisy += block_size
    total = -(-len(ids) // pad_to) * pad_to
    pad = total - len(ids)
    ids += [mask_id] * pad
    pos += [0] * pad
    noisy += [1] * pad
    block += [-2] * pad
    start += [0] * pad

    ar_index = [
        i
        for i in range(n - 1)
        if ex.assistant[i + 1] and (ar_in_spans or not ex.span[i + 1])
    ]
    return Layout(
        input_ids=torch.tensor([ids]),
        position_ids=torch.tensor([pos]),
        noisy=torch.tensor(noisy, dtype=torch.bool),
        block=torch.tensor(block),
        block_start=torch.tensor(start),
        ar_index=torch.tensor(ar_index, dtype=torch.long),
        ar_target=torch.tensor([ex.ids[i + 1] for i in ar_index]),
        dlm_index=torch.tensor(dlm_index, dtype=torch.long),
        dlm_target=torch.tensor(dlm_target, dtype=torch.long),
        dlm_weight=torch.tensor(dlm_weight, dtype=torch.float32),
        num_noisy=num_noisy,
    )


def block_mask(lay: Layout, device: torch.device) -> BlockMask:
    """Flex-attention mask for a layout (see module docstring)."""
    noisy = lay.noisy.to(device)
    block = lay.block.to(device)
    start = lay.block_start.to(device)
    pos = lay.position_ids[0].to(device)

    def mask_mod(b, h, q, kv):
        clean = ~noisy[q] & ~noisy[kv] & (kv <= q)
        same_block = noisy[q] & noisy[kv] & (block[q] == block[kv])
        prefix = noisy[q] & ~noisy[kv] & (pos[kv] < start[q])
        return clean | same_block | prefix

    t = lay.input_ids.shape[1]
    return create_block_mask(mask_mod, None, None, t, t, device=device)
