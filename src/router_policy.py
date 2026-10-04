"""Learned AR/diffusion routing policy.

Pure torch with no imports from this repo: the same file is copied into the
SGLang DLLM fork (`sglang/srt/dllm/router_policy.py`) so that training, the HF
reference decoder and SGLang rollouts all run identical code.

The policy sees the decision point's causal state and outputs the probability
of decoding the next segment with diffusion (action 1) instead of AR
(action 0). Features:
    hidden    final-layer hidden state at the last committed token (causal)
    entropy   next-token entropy (nats) of the causal distribution
    top1      probability of the most likely next token
    last_dlm  1 if the previous segment was diffusion, else 0
"""

import torch
from torch import nn

NUM_SCALAR_FEATURES = 3
ACTION_AR = 0
ACTION_DLM = 1


class RouterPolicy(nn.Module):
    """Small MLP mapping decision features to a diffusion logit."""

    def __init__(self, hidden_size: int, width: int = 256) -> None:
        """Initializes the policy at p(diffusion) = 0.5 everywhere.

        Args:
            hidden_size: Backbone hidden size (3072 for the 3B model).
            width: Hidden width of the MLP.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.width = width
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + NUM_SCALAR_FEATURES, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def features(
        hidden: torch.Tensor,
        next_logits: torch.Tensor,
        last_dlm: torch.Tensor,
    ) -> torch.Tensor:
        """Builds float32 features for a batch of decision points.

        Args:
            hidden: (batch, hidden_size) causal hidden states.
            next_logits: (batch, vocab) causal next-token logits.
            last_dlm: (batch,) 1.0 if the previous segment was diffusion.

        Returns:
            (batch, hidden_size + NUM_SCALAR_FEATURES) float32 features.
        """
        log_probs = torch.log_softmax(next_logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        top1 = probs.max(dim=-1).values
        scalars = torch.stack(
            [entropy, top1, last_dlm.float().to(entropy.device)], dim=-1
        )
        return torch.cat([hidden.float(), scalars], dim=-1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Returns the diffusion logit for each row of `feats`."""
        hidden = self.norm(feats[..., : self.hidden_size])
        scalars = feats[..., self.hidden_size :]
        return self.mlp(torch.cat([hidden, scalars], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        feats: torch.Tensor,
        mode: str = "sample",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chooses actions for a batch of decision points.

        Args:
            feats: Features from `features`.
            mode: "sample" (stochastic), "greedy" (p > 0.5), "fixed_ar",
                "fixed_dlm", or "random:<p>" (ignores the network).
            generator: Optional RNG for sampling.

        Returns:
            (actions, logp): int64 actions (1 = diffusion) and the log
            probability of each chosen action under the policy.
        """
        logits = self.forward(feats)
        p_dlm = torch.sigmoid(logits)
        if mode == "sample":
            u = torch.rand(
                p_dlm.shape, generator=generator, device=p_dlm.device
            )
            actions = (u < p_dlm).long()
        elif mode == "greedy":
            actions = (p_dlm > 0.5).long()
        elif mode == "fixed_ar":
            actions = torch.zeros_like(p_dlm, dtype=torch.long)
        elif mode == "fixed_dlm":
            actions = torch.ones_like(p_dlm, dtype=torch.long)
        elif mode.startswith("random:"):
            p = float(mode.split(":", 1)[1])
            u = torch.rand(
                p_dlm.shape, generator=generator, device=p_dlm.device
            )
            actions = (u < p).long()
        else:
            raise ValueError(f"unknown router mode {mode!r}")
        return actions, action_log_prob(logits, actions)

    def save(self, path: str, **extra: object) -> None:
        """Saves weights and shape metadata."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_size": self.hidden_size,
                "width": self.width,
                **extra,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "RouterPolicy":
        """Loads a policy saved with `save`."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        policy = cls(ckpt["hidden_size"], ckpt["width"])
        policy.load_state_dict(ckpt["state_dict"])
        return policy


def action_log_prob(
    logits: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Log probability of Bernoulli `actions` (1 = diffusion) given logits."""
    logits = logits.float()
    return torch.where(
        actions.bool(),
        nn.functional.logsigmoid(logits),
        nn.functional.logsigmoid(-logits),
    )


def bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of the Bernoulli policy with the given logits."""
    p = torch.sigmoid(logits.float())
    eps = 1e-8
    return -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())
