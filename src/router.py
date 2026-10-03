"""Routers that choose between AR and diffusion decoding.

A router is a callable `RouterState -> Mode`. The interleaved decoder calls it
at every decision point, i.e. after each AR chunk or diffusion block.
"""

import dataclasses
import itertools
from typing import Literal, Protocol

import torch

Mode = Literal["ar", "dlm"]


@dataclasses.dataclass
class RouterState:
    """What a router can look at when it makes a decision.

    Attributes:
        next_logit: Next-token logits given everything decoded so far,
            shape (vocab,).
        hidden: Final hidden state at the last position, shape (hidden,).
        num_generated: Number of tokens generated so far.
        last_mode: Mode used for the previous segment, None at the start.
    """

    next_logit: torch.Tensor
    hidden: torch.Tensor
    num_generated: int
    last_mode: Mode | None


class Router(Protocol):
    """Chooses the decoding mode for the next segment."""

    def __call__(self, state: RouterState) -> Mode:
        """Returns "ar" or "dlm"."""
        ...


class FixedRouter:
    """Always returns the same mode (reduces to pure AR or pure diffusion)."""

    def __init__(self, mode: Mode) -> None:
        """Initializes the router.

        Args:
        mode: The mode to always return.
        """
        self.mode = mode

    def __call__(self, state: RouterState) -> Mode:
        """Returns the fixed mode."""
        return self.mode


class CycleRouter:
    """Cycles through a fixed schedule, e.g. ("ar", "dlm") alternates."""

    def __init__(self, schedule: tuple[Mode, ...] = ("ar", "dlm")) -> None:
        """Initializes the router.

        Args:
        schedule: Modes to cycle through, in order.
        """
        self._modes = itertools.cycle(schedule)

    def __call__(self, state: RouterState) -> Mode:
        """Returns the next mode in the schedule."""
        return next(self._modes)


class EntropyRouter:
    """Uses diffusion when the next token is predictable, AR otherwise.

    Heuristic baseline: block diffusion commits many tokens per forward pass,
    which is safest when the model is already confident about what comes next.

    Attributes:
        threshold: Next-token entropy (nats) below which diffusion is used.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initializes the router.

        Args:
        threshold: Entropy (nats) below which diffusion is chosen.
        """
        self.threshold = threshold

    def __call__(self, state: RouterState) -> Mode:
        """Returns "dlm" if the next-token entropy is low, else "ar"."""
        log_probs = torch.log_softmax(state.next_logit.float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum().item()
        return "dlm" if entropy < self.threshold else "ar"


ROUTERS = ("entropy", "cycle")


def make_router(name: str, entropy_threshold: float = 0.5) -> Router:
    """Builds a router by name.

    Args:
        name: One of `ROUTERS`.
        entropy_threshold: Threshold for the entropy router.

    Returns:
        The router.

    Raises:
        ValueError: If `name` is unknown.
    """
    if name == "entropy":
        return EntropyRouter(entropy_threshold)
    if name == "cycle":
        return CycleRouter()
    raise ValueError(f"Unknown router {name!r}; choose from {ROUTERS}.")
