"""Span-SFT model wrapper: LoRA + trainable span-token rows + span loss.

Only small tensors are trained and saved: LoRA adapters on the backbone's
attention/MLP projections and the embedding / LM-head rows of the two span
tokens (`<diff>`, `</diff>`), applied by forward hooks so the frozen 131k-row
matrices are never copied. A checkpoint (`adapter.pt`) holds both.
"""

import contextlib
import pathlib

import peft
import torch
from torch import nn

from src import spans
from src.nemotron import modeling_nemotron_labs_diffusion as modeling

DEFAULT_REPO = "nvidia/Nemotron-Labs-Diffusion-3B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # fmt: skip
SPAN_TOKENS = (spans.DIFF_OPEN, spans.DIFF_CLOSE)
# Span-token rows start as the mean of these texts' token rows (the reserved
# tokens' own rows were never trained).
INIT_TEXT = ("<diff>", "</diff>")


class TokenRows(nn.Module):
    """Trainable embedding and LM-head rows for a few token ids (float32)."""

    def __init__(
        self,
        model: modeling.NemotronLabsDiffusionModel,
        init_ids: list[list[int]] | None = None,
    ) -> None:
        """Rows initialized from the mean rows of `init_ids` per token.

        Without `init_ids` (loading a checkpoint) the rows are placeholders.
        """
        super().__init__()
        emb = model.encoder.embed_tokens.weight
        head = model.diffusion_head.weight
        init_ids = init_ids or [[t] for t in SPAN_TOKENS]
        self.register_buffer("ids", torch.tensor(SPAN_TOKENS), persistent=False)
        self.emb = nn.Parameter(
            torch.stack([emb[i].float().mean(0) for i in init_ids]).detach()
        )
        self.head = nn.Parameter(
            torch.stack([head[i].float().mean(0) for i in init_ids]).detach()
        )

    def attach(self, model: modeling.NemotronLabsDiffusionModel) -> list:
        """Installs the hooks; returns their handles."""

        def emb_hook(_m, inputs, out):
            (input_ids,) = inputs
            for k in range(len(self.ids)):
                hit = (input_ids == self.ids[k]).unsqueeze(-1)
                out = torch.where(hit, self.emb[k].to(out.dtype), out)
            return out

        def head_hook(_m, inputs, out):
            (hidden,) = inputs
            rows = hidden.to(self.head.dtype) @ self.head.T
            out = out.clone()
            out[..., self.ids] = rows.to(out.dtype)
            return out

        return [
            model.encoder.embed_tokens.register_forward_hook(emb_hook),
            model.diffusion_head.register_forward_hook(head_hook),
        ]


def load_base(repo: str, device: str) -> modeling.NemotronLabsDiffusionModel:
    """Loads the frozen bf16 backbone."""
    model = modeling.NemotronLabsDiffusionModel.from_pretrained(
        repo, torch_dtype=torch.bfloat16
    )
    model = model.to(device)  # ty: ignore[invalid-argument-type]
    model.requires_grad_(False)
    return model


def add_adapters(
    model, tokenizer, rank: int, alpha: int, dropout: float = 0.0
):
    """Wraps the backbone with LoRA and span-token rows.

    Returns:
        (peft model, TokenRows) - the TokenRows hooks are installed.
    """
    init = [spans.encode(tokenizer, t) for t in INIT_TEXT]
    rows = TokenRows(model, init).to(model.device)
    rows.attach(model)
    cfg = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    pmodel = peft.get_peft_model(model, cfg)
    return pmodel, rows


def trainable_state(pmodel, rows: TokenRows) -> dict:
    """The tensors a checkpoint stores (LoRA + span rows)."""
    lora = peft.get_peft_model_state_dict(pmodel)
    return {
        "lora": {k: v.detach().cpu() for k, v in lora.items()},
        "rows": {k: v.detach().cpu() for k, v in rows.state_dict().items()},
        "lora_config": pmodel.peft_config["default"].to_dict(),
    }


def save(path: pathlib.Path, pmodel, rows: TokenRows, extra: dict) -> None:
    """Saves the trainable tensors (not the backbone)."""
    torch.save({**trainable_state(pmodel, rows), **extra}, path)


def load_adapter(model, path: pathlib.Path):
    """Applies a saved adapter to a base model for inference.

    LoRA weights are merged into the backbone (plain modules afterwards), so
    the decoders run unchanged. Returns the model.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["lora_config"])
    lcfg = peft.LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=list(cfg["target_modules"]),
        bias="none",
    )
    pmodel = peft.get_peft_model(model, lcfg)
    peft.set_peft_model_state_dict(pmodel, ckpt["lora"])
    model = pmodel.merge_and_unload()
    rows = TokenRows(model).to(model.device)
    rows.load_state_dict(ckpt["rows"])
    rows.requires_grad_(False)
    rows.attach(model)
    model._span_rows = rows  # keep the hooked parameters alive
    return model


@contextlib.contextmanager
def flex_attention(model):
    """Routes causal attention through flex attention (for BlockMasks)."""
    enc = (
        model.get_base_model().encoder
        if hasattr(model, "get_base_model")
        else model.encoder
    )
    old = enc.config._attn_implementation
    enc.config._attn_implementation = "flex_attention"
    flags = [layer.self_attn.diffusion_lm for layer in enc.layers]
    for layer in enc.layers:
        layer.self_attn.diffusion_lm = False  # use the attention_mask path
    try:
        yield enc
    finally:
        enc.config._attn_implementation = old
        for layer, f in zip(enc.layers, flags, strict=True):
            layer.self_attn.diffusion_lm = f


def span_hidden(model, lay: spans.Layout) -> torch.Tensor:
    """Final hidden states (T, hidden) of a layout under the span mask.

    Must run inside `flex_attention(model)`; with gradient checkpointing the
    backward pass recomputes layers, so keep the context open through it.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    enc = base.encoder
    assert (
        enc.config._attn_implementation == "flex_attention"
    ), "use flex_attention(model)"
    device = base.device
    mask = spans.block_mask(lay, device)
    ids = lay.input_ids.to(device)
    pos = lay.position_ids.to(device)
    h = enc.embed_tokens(ids)
    pe = enc.rotary_emb(h, position_ids=pos)
    for layer in enc.layers:
        h = layer(
            h,
            attention_mask=mask,
            position_ids=pos,
            position_embeddings=pe,
            cache_position=pos[0],
            use_cache=False,
        )
    return enc.norm(h)[0]


def span_loss(
    model,
    lay: spans.Layout,
    ar_norm: float,
    dlm_norm: float,
    ar_weight: float = 1.0,
    dlm_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Loss of one layout, pre-normalized by the step's token totals.

    `ar_norm` / `dlm_norm` are the numbers of AR targets / noisy positions in
    the whole optimizer step, so summing per-example losses over gradient
    accumulation gives token-averaged losses.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = base.device
    h = span_hidden(model, lay)
    head = base.diffusion_head
    stats = {}
    loss = h.new_zeros((), dtype=torch.float32)
    if len(lay.ar_index):
        logits = head(h[lay.ar_index.to(device)]).float()
        ce = nn.functional.cross_entropy(
            logits, lay.ar_target.to(device), reduction="sum"
        )
        loss = loss + ar_weight * ce / ar_norm
        stats["ar_sum"], stats["ar_n"] = float(ce), len(lay.ar_index)
    if len(lay.dlm_index):
        logits = head(h[lay.dlm_index.to(device)]).float()
        ce = nn.functional.cross_entropy(
            logits, lay.dlm_target.to(device), reduction="none"
        )
        weighted = (ce * lay.dlm_weight.to(device)).sum()
        loss = loss + dlm_weight * weighted / dlm_norm
        stats["dlm_sum"], stats["dlm_n"] = float(weighted), lay.num_noisy
        stats["dlm_ce_sum"], stats["dlm_masked"] = float(ce.sum()), len(ce)
    return loss, stats
