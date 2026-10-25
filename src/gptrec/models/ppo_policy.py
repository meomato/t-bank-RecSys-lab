"""GPTRec actor-critic policy for slate and model-based feedback rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig


@dataclass(frozen=True, slots=True)
class PPOPolicyConfig:
    """Configuration added on top of a pretrained GPTRec backbone."""

    feedback_feature_dim: int = 3
    feedback_dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPTRecActorCritic(nn.Module):
    """A GPTRec policy with a shared value head and optional feedback adapter.

    The policy starts from the exact GPTRec-base weights. Historical KION events
    have zero feedback features because the dataset has no impressions. During a
    model-based rollout, predicted completion, duration and return outcomes can
    be attached to generated item tokens through ``feedback_projection``.
    The projection is initialized to zero, so the initial policy is identical to
    GPTRec-base in both rollout modes.
    """

    def __init__(
        self,
        backbone: GPTRecBase,
        config: PPOPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or PPOPolicyConfig()
        hidden_size = backbone.project_config.hidden_size
        self.feedback_projection = nn.Linear(
            self.config.feedback_feature_dim,
            hidden_size,
            bias=False,
        )
        self.feedback_dropout = nn.Dropout(self.config.feedback_dropout)
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.feedback_projection.weight)
        nn.init.zeros_(self.value_head.bias)

    @classmethod
    def from_gptrec_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: PPOPolicyConfig | None = None,
        *,
        map_location: str | torch.device = "cpu",
    ) -> GPTRecActorCritic:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        backbone_config = GPTRecBaseConfig(**checkpoint["model_config"])
        backbone = GPTRecBase(backbone_config)
        backbone.load_state_dict(checkpoint["model_state_dict"])
        return cls(backbone, config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        feedback_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return next-item logits and the value of the current state."""
        token_embeddings = self.backbone.model.transformer.wte(input_ids)
        if feedback_features is not None:
            if feedback_features.shape[:2] != input_ids.shape:
                raise ValueError("feedback features must align with input_ids")
            if feedback_features.size(-1) != self.config.feedback_feature_dim:
                raise ValueError("unexpected feedback feature dimension")
            token_embeddings = token_embeddings + self.feedback_dropout(
                self.feedback_projection(feedback_features)
            )
        output = self.backbone.model.transformer(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(attention_mask)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        if torch.any(last_positions < 0):
            raise ValueError("every policy state needs at least one observed item")
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        state = output.last_hidden_state[batch_indices, last_positions]
        return self.backbone.model.lm_head(state), self.value_head(state).squeeze(-1)

    def policy_state_dict(self) -> dict[str, Any]:
        return {
            "base_model_config": self.backbone.project_config.to_dict(),
            "policy_config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
        }


def append_policy_step(
    input_ids: Tensor,
    attention_mask: Tensor,
    feedback_features: Tensor,
    action_ids: Tensor,
    action_feedback: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append an action to each state, dropping the oldest token when full."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
    if feedback_features.shape[:2] != input_ids.shape:
        raise ValueError("feedback features must align with input_ids")
    if action_ids.shape != (input_ids.size(0),):
        raise ValueError("action_ids must have shape [batch]")
    if action_feedback.shape != (input_ids.size(0), feedback_features.size(-1)):
        raise ValueError("action_feedback must have shape [batch, feedback_dim]")

    next_ids = input_ids.clone()
    next_mask = attention_mask.clone()
    next_feedback = feedback_features.clone()
    lengths = attention_mask.long().sum(dim=1)
    full = lengths.ge(input_ids.size(1))

    if full.any():
        next_ids[full, :-1] = input_ids[full, 1:].clone()
        next_ids[full, -1] = action_ids[full]
        next_mask[full] = 1
        next_feedback[full, :-1] = feedback_features[full, 1:].clone()
        next_feedback[full, -1] = action_feedback[full]

    not_full = ~full
    if not_full.any():
        rows = torch.arange(input_ids.size(0), device=input_ids.device)[not_full]
        columns = lengths[not_full]
        next_ids[rows, columns] = action_ids[not_full]
        next_mask[rows, columns] = 1
        next_feedback[rows, columns] = action_feedback[not_full]
    return next_ids, next_mask, next_feedback
