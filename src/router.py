"""Routers that choose between AR and diffusion decoding.

A router is a callable `RouterState -> Mode`. The interleaved decoder calls it
at every decision point, i.e. after each AR chunk or diffusion block.
"""

import dataclasses
import itertools
import random
from typing import Literal, Protocol

import torch

from src import router_policy

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


class RandomRouter:
    """Untrained baseline: picks diffusion with a fixed probability.

    Attributes:
        p_dlm: Probability of choosing "dlm" at each decision.
    """

    def __init__(self, p_dlm: float, seed: int | None = None) -> None:
        """Initializes the router.

        Args:
            p_dlm: Probability of choosing "dlm" at each decision.
            seed: Seed for the router's own RNG; None for nondeterministic.
        """
        if not 0.0 <= p_dlm <= 1.0:
            raise ValueError(f"p_dlm must be in [0, 1], got {p_dlm}.")
        self.p_dlm = p_dlm
        self._rng = random.Random(seed)

    def __call__(self, state: RouterState) -> Mode:
        """Returns "dlm" with probability `p_dlm`, else "ar"."""
        return "dlm" if self._rng.random() < self.p_dlm else "ar"


@dataclasses.dataclass
class Decision:
    """One routing decision made by a `LearnedRouter`."""

    features: torch.Tensor
    action: int
    logp: float
    num_generated: int


class LearnedRouter:
    """Adapts a `RouterPolicy` to the `Router` protocol and records decisions.

    Attributes:
        policy: The learned policy.
        mode: Action mode passed to `RouterPolicy.act` ("greedy" for eval,
            "sample" for exploration).
        trace: Decisions made since the last `reset`.
    """

    def __init__(self, policy: router_policy.RouterPolicy, mode: str) -> None:
        """Initializes the router.

        Args:
            policy: The learned policy.
            mode: Action mode for `RouterPolicy.act`.
        """
        self.policy = policy
        self.mode = mode
        self.trace: list[Decision] = []

    def reset(self) -> None:
        """Clears the decision trace (call between generations)."""
        self.trace = []

    def __call__(self, state: RouterState) -> Mode:
        """Returns the policy's action for this decision point."""
        device = state.hidden.device
        self.policy.to(device)
        last_dlm = torch.tensor(
            [float(state.last_mode == "dlm")], device=device
        )
        feats = router_policy.RouterPolicy.features(
            state.hidden[None], state.next_logit[None], last_dlm
        )
        actions, logp = self.policy.act(feats, mode=self.mode)
        action = int(actions[0])
        self.trace.append(
            Decision(
                feats[0].cpu(), action, float(logp[0]), state.num_generated
            )
        )
        return "dlm" if action == router_policy.ACTION_DLM else "ar"


class ForcedARStart:
    """Always routes to AR until `n_tokens` have been generated.

    Wraps another router; the wrapped router is not called (and so records
    nothing) during the forced AR start. Mirrors `forced_ar_tokens` in the
    SGLang RoutedDecoding algorithm.
    """

    def __init__(self, router: Router, n_tokens: int) -> None:
        """Initializes the wrapper.

        Args:
            router: Router used once the forced AR start is over.
            n_tokens: Number of tokens always decoded with AR.
        """
        self.router = router
        self.n_tokens = n_tokens

    def __call__(self, state: RouterState) -> Mode:
        """Returns "ar" during the forced start, else the router's choice."""
        if state.num_generated < self.n_tokens:
            return "ar"
        return self.router(state)


ROUTERS = ("entropy", "cycle", "random")


def make_router(
    name: str,
    entropy_threshold: float = 0.5,
    p_dlm: float = 0.5,
    seed: int | None = None,
) -> Router:
    """Builds a router by name.

    Args:
        name: One of `ROUTERS`.
        entropy_threshold: Threshold for the entropy router.
        p_dlm: Diffusion probability for the random router.
        seed: Seed for the random router.

    Returns:
        The router.

    Raises:
        ValueError: If `name` is unknown.
    """
    if name == "entropy":
        return EntropyRouter(entropy_threshold)
    if name == "cycle":
        return CycleRouter()
    if name == "random":
        return RandomRouter(p_dlm, seed)
    raise ValueError(f"Unknown router {name!r}; choose from {ROUTERS}.")
