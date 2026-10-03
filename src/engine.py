"""Unified AR / diffusion inference for Nemotron-Labs-Diffusion.

The model is a single set of weights; AR vs. diffusion decoding is just a
different attention pattern / decoding loop. So we load it once and dispatch
per call:

    engine = NemotronEngine()
    engine.generate("Explain KV caching.", mode="ar")
    engine.generate("Explain KV caching.", mode="dlm")
"""

import dataclasses
import time
import typing
from typing import Literal

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src import interleave
from src import router as router_lib
from src.nemotron import modeling_nemotron_labs_diffusion as modeling

Mode = Literal["ar", "dlm", "mix"]
Messages = list[dict[str, str]]
MODES: tuple[Mode, ...] = ("ar", "dlm", "mix")
DEFAULT_REPO = "nvidia/Nemotron-Labs-Diffusion-3B"


@dataclasses.dataclass
class GenerationResult:
    """Output of a single generation call.

    Attributes:
        text: Decoded reply, without special tokens.
        token_ids: Generated token ids (prompt excluded).
        mode: Decoding mode that produced the reply.
        nfe: Number of forward evaluations of the model.
        seconds: Wall-clock generation time.
        segments: Routing trace (mix mode only), e.g. "ar8 dlm32".
    """

    text: str
    token_ids: list[int]
    mode: Mode
    nfe: int
    seconds: float
    segments: str = ""

    @property
    def num_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.token_ids)

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput in tokens per second."""
        return self.num_tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """Average tokens produced per forward evaluation."""
        return self.num_tokens / self.nfe if self.nfe else 0.0


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}.")


class NemotronEngine:
    """Loads Nemotron-Labs-Diffusion once and decodes in AR or diffusion mode.

    Attributes:
        mode: Default decoding mode used when `generate` is not given one.
    """

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        mode: Mode = "ar",
        router: router_lib.Router | None = None,
        ar_chunk: int = 8,
    ) -> None:
        """Loads the tokenizer and model.

        Args:
            repo: Hugging Face repo id or local path.
            device: Torch device to place the model on.
            dtype: Model weight dtype.
            mode: Initial default decoding mode.
            router: Router used in "mix" mode; defaults to the entropy router.
            ar_chunk: Tokens per AR segment in "mix" mode.
        """
        _check_mode(mode)
        self.device = device
        self.tokenizer = typing.cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(repo, trust_remote_code=True),
        )
        model = modeling.NemotronLabsDiffusionModel.from_pretrained(
            repo, torch_dtype=dtype
        )
        # ty mis-resolves transformers' decorated `.to`.
        model = model.to(device)  # ty: ignore[invalid-argument-type]
        self.model = model.eval()
        self.mode = mode
        self.router = router or router_lib.EntropyRouter()
        self.ar_chunk = ar_chunk

    def set_mode(self, mode: Mode) -> None:
        """Sets the default decoding mode.

        Args:
            mode: One of "ar", "dlm" or "mix".

        Raises:
            ValueError: If `mode` is not supported.
        """
        _check_mode(mode)
        self.mode = mode

    def encode_chat(self, messages: Messages | str) -> torch.Tensor:
        """Applies the chat template and tokenizes.

        Args:
            messages: A chat history, or a single user message.

        Returns:
            Prompt token ids of shape (1, prompt_len) on the engine device.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        prompt = str(
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids.to(self.device)

    @torch.inference_mode()
    def generate(
        self,
        messages: Messages | str,
        mode: Mode | None = None,
        max_new_tokens: int = 512,
        block_length: int = 32,
        threshold: float = 0.9,
    ) -> GenerationResult:
        """Generates a reply.

        Args:
            messages: A chat history, or a single user message.
            mode: Decoding mode; defaults to `self.mode`.
            max_new_tokens: Maximum number of tokens to generate.
            block_length: Diffusion block size (dlm and mix modes).
            threshold: Diffusion unmasking confidence (dlm and mix modes).

        Returns:
            The decoded reply with timing and NFE statistics.

        Raises:
            ValueError: If `mode` is not supported.
        """
        mode = mode or self.mode
        _check_mode(mode)
        prompt_ids = self.encode_chat(messages)

        self._sync()
        start = time.perf_counter()
        segments = ""
        if mode == "mix":
            mixed = interleave.interleaved_generate(
                self.model,
                prompt_ids,
                self.router,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                block_length=block_length,
                threshold=threshold,
                ar_chunk=self.ar_chunk,
            )
            out_ids = torch.cat(
                [
                    prompt_ids,
                    torch.tensor([mixed.token_ids], device=self.device),
                ],
                dim=1,
            )
            nfe = mixed.nfe
            segments = " ".join(
                f"{s.mode}{s.num_tokens}" for s in mixed.segments
            )
        elif mode == "ar":
            out_ids, nfe = self.model.ar_generate(
                prompt_ids, max_new_tokens=max_new_tokens
            )
        else:
            out_ids, nfe = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                block_length=block_length,
                threshold=threshold,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self._sync()
        seconds = time.perf_counter() - start

        new_ids = out_ids[0, prompt_ids.shape[1] :].tolist()
        return GenerationResult(
            text=str(self.tokenizer.decode(new_ids, skip_special_tokens=True)),
            token_ids=new_ids,
            mode=mode,
            nfe=int(nfe),
            seconds=seconds,
            segments=segments,
        )

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
