"""Multi-task conditional outcome model used as a conservative reward proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig


@dataclass(frozen=True, slots=True)
class RewardModelConfig:
    expert_count: int = 3
    expert_hidden_size: int = 256
    tower_hidden_size: int = 128
    content_type_count: int = 3
    content_type_embedding_size: int = 8
    item_numeric_feature_dim: int = 2
    completion_class_count: int = 6
    dropout: float = 0.1
    freeze_backbone: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> RewardModelConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        return cls(**document.get("model", {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KIONRewardModel(nn.Module):
    """MMoE heads for completion, duration utility and short-horizon return."""

    completion_utilities = (0.0, 0.055, 0.305, 0.70, 0.945, 1.0)

    def __init__(
        self,
        backbone: GPTRecBase,
        config: RewardModelConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        hidden_size = backbone.project_config.hidden_size
        self.content_type_embedding = nn.Embedding(
            config.content_type_count, config.content_type_embedding_size
        )
        fusion_size = (
            hidden_size * 4
            + config.content_type_embedding_size
            + config.item_numeric_feature_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(fusion_size, config.expert_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.expert_hidden_size),
            nn.Dropout(config.dropout),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.expert_hidden_size, config.expert_hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                for _ in range(config.expert_count)
            ]
        )
        self.gates = nn.ModuleDict(
            {
                name: nn.Linear(config.expert_hidden_size, config.expert_count)
                for name in ("completion", "duration", "return")
            }
        )
        self.towers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(config.expert_hidden_size, config.tower_hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                for name in ("completion", "duration", "return")
            }
        )
        self.completion_head = nn.Linear(
            config.tower_hidden_size, config.completion_class_count
        )
        self.duration_head = nn.Linear(config.tower_hidden_size, 2)
        self.return_head = nn.Linear(config.tower_hidden_size, 1)
        self.register_buffer(
            "completion_utility_values",
            torch.tensor(self.completion_utilities, dtype=torch.float32),
        )
        if config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    @classmethod
    def from_gptrec_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: RewardModelConfig,
        *,
        map_location: str | torch.device = "cpu",
    ) -> KIONRewardModel:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        backbone_config = GPTRecBaseConfig(**checkpoint["model_config"])
        backbone = GPTRecBase(backbone_config)
        backbone.load_state_dict(checkpoint["model_state_dict"])
        return cls(backbone, config)

    def _last_history_state(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.backbone.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(attention_mask)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        if torch.any(last_positions < 0):
            raise ValueError("each reward-model example needs non-empty history")
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        return output.last_hidden_state[batch_indices, last_positions]

    def _task_representation(self, shared: Tensor, task: str) -> Tensor:
        expert_values = torch.stack([expert(shared) for expert in self.experts], dim=1)
        weights = self.gates[task](shared).softmax(dim=-1).unsqueeze(-1)
        mixed = (expert_values * weights).sum(dim=1)
        return self.towers[task](mixed)

    def _predict_from_history(
        self,
        history: Tensor,
        action_id: Tensor,
        content_type_id: Tensor,
        item_numeric_features: Tensor,
    ) -> dict[str, Tensor]:
        item = self.backbone.model.transformer.wte(action_id)
        type_embedding = self.content_type_embedding(content_type_id)
        fusion = torch.cat(
            [
                history,
                item,
                history * item,
                torch.abs(history - item),
                type_embedding,
                item_numeric_features,
            ],
            dim=-1,
        )
        shared = self.input_projection(fusion)
        completion_logits = self.completion_head(
            self._task_representation(shared, "completion")
        )
        duration_raw = self.duration_head(self._task_representation(shared, "duration"))
        duration_alpha_beta = F.softplus(duration_raw) + 1.0
        return_logit = self.return_head(
            self._task_representation(shared, "return")
        ).squeeze(-1)
        completion_probabilities = completion_logits.softmax(dim=-1)
        return {
            "completion_logits": completion_logits,
            "completion_utility": (
                completion_probabilities * self.completion_utility_values
            ).sum(dim=-1),
            "duration_alpha": duration_alpha_beta[:, 0],
            "duration_beta": duration_alpha_beta[:, 1],
            "duration_utility": duration_alpha_beta[:, 0]
            / duration_alpha_beta.sum(dim=-1),
            "return_logit": return_logit,
            "return_probability": return_logit.sigmoid(),
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        action_id: Tensor,
        content_type_id: Tensor,
        item_numeric_features: Tensor,
        *,
        compute_choice: bool = True,
    ) -> dict[str, Tensor]:
        history = self._last_history_state(input_ids, attention_mask)
        result = self._predict_from_history(
            history, action_id, content_type_id, item_numeric_features
        )
        if compute_choice:
            choice_logits = self.backbone.model.lm_head(history)
            result["choice_log_probability"] = choice_logits.log_softmax(dim=-1).gather(
                1, action_id.unsqueeze(1)
            ).squeeze(1)
        return result

    def predict_candidates(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        action_ids: Tensor,
        content_type_ids: Tensor,
        item_numeric_features: Tensor,
    ) -> dict[str, Tensor]:
        """Score several candidate actions while encoding each history only once."""
        if action_ids.ndim != 2:
            raise ValueError("candidate action_ids must have shape [batch, candidates]")
        batch_size, candidate_count = action_ids.shape
        history = self._last_history_state(input_ids, attention_mask)
        expanded_history = history.unsqueeze(1).expand(-1, candidate_count, -1)
        flat_predictions = self._predict_from_history(
            expanded_history.reshape(batch_size * candidate_count, -1),
            action_ids.reshape(-1),
            content_type_ids.reshape(-1),
            item_numeric_features.reshape(batch_size * candidate_count, -1),
        )
        return {
            name: value.reshape(batch_size, candidate_count, *value.shape[1:])
            for name, value in flat_predictions.items()
        }

    @staticmethod
    def loss(
        predictions: dict[str, Tensor],
        completion_class: Tensor,
        duration_utility: Tensor,
        return_7d: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        completion_mask = completion_class.ge(0)
        if completion_mask.any():
            completion_loss = F.cross_entropy(
                predictions["completion_logits"][completion_mask],
                completion_class[completion_mask],
            )
        else:
            completion_loss = predictions["completion_logits"].sum() * 0
        duration_distribution = torch.distributions.Beta(
            predictions["duration_alpha"], predictions["duration_beta"]
        )
        duration_loss = -duration_distribution.log_prob(
            duration_utility.clamp(1e-4, 1 - 1e-4)
        ).mean()
        return_loss = F.binary_cross_entropy_with_logits(
            predictions["return_logit"], return_7d
        )
        losses = {
            "completion": completion_loss,
            "duration": duration_loss,
            "return": return_loss,
        }
        return sum(losses.values()), losses


def compose_conservative_reward(
    predictions: dict[str, Tensor],
    *,
    epistemic_uncertainty: Tensor | float = 0.0,
    completion_weight: float = 0.55,
    duration_weight: float = 0.30,
    return_weight: float = 0.15,
    uncertainty_weight: float = 0.25,
) -> Tensor:
    """Combine calibrated outcomes inside a GPTRec-constrained candidate set."""
    core_reward = (
        completion_weight * predictions["completion_utility"]
        + duration_weight * predictions["duration_utility"]
        + return_weight * predictions["return_probability"]
    )
    return core_reward - uncertainty_weight * torch.as_tensor(
        epistemic_uncertainty,
        device=core_reward.device,
        dtype=core_reward.dtype,
    )
