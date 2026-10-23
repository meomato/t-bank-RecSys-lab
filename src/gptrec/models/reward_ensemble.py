"""Inference utilities for conservative ensemble reward estimates."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from gptrec.models.reward_model import (
    KIONRewardModel,
    RewardModelConfig,
    compose_conservative_reward,
)


class RewardModelEnsemble(nn.Module):
    """Average calibrated outcome heads and expose epistemic disagreement."""

    def __init__(
        self,
        models: list[KIONRewardModel],
        completion_temperatures: list[float],
        return_temperatures: list[float],
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("reward ensemble needs at least one model")
        if not (
            len(models) == len(completion_temperatures) == len(return_temperatures)
        ):
            raise ValueError("models and temperature lists must have equal lengths")
        self.models = nn.ModuleList(models)
        self.register_buffer(
            "completion_temperatures",
            torch.tensor(completion_temperatures, dtype=torch.float32),
        )
        self.register_buffer(
            "return_temperatures",
            torch.tensor(return_temperatures, dtype=torch.float32),
        )

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str | Path],
        *,
        backbone_checkpoint: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> RewardModelEnsemble:
        models: list[KIONRewardModel] = []
        completion_temperatures: list[float] = []
        return_temperatures: list[float] = []
        for path in checkpoint_paths:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            config = RewardModelConfig(**checkpoint["model_config"])
            model = KIONRewardModel.from_gptrec_checkpoint(
                backbone_checkpoint, config, map_location=map_location
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)
            temperatures = checkpoint.get("temperatures", {})
            completion_temperatures.append(float(temperatures.get("completion", 1.0)))
            return_temperatures.append(float(temperatures.get("return", 1.0)))
        return cls(models, completion_temperatures, return_temperatures)

    @torch.no_grad()
    def predict(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        action_id: Tensor,
        content_type_id: Tensor,
        item_numeric_features: Tensor,
    ) -> dict[str, Tensor]:
        member_completion: list[Tensor] = []
        member_duration: list[Tensor] = []
        member_return: list[Tensor] = []
        choice_log_probability: Tensor | None = None
        for index, model in enumerate(self.models):
            output = model(
                input_ids,
                attention_mask,
                action_id,
                content_type_id,
                item_numeric_features,
                compute_choice=index == 0,
            )
            completion_probability = (
                output["completion_logits"] / self.completion_temperatures[index]
            ).softmax(dim=-1)
            member_completion.append(
                (completion_probability * model.completion_utility_values).sum(dim=-1)
            )
            member_duration.append(output["duration_utility"])
            member_return.append(
                (output["return_logit"] / self.return_temperatures[index]).sigmoid()
            )
            if index == 0:
                choice_log_probability = output["choice_log_probability"]
        completion = torch.stack(member_completion)
        duration = torch.stack(member_duration)
        return_probability = torch.stack(member_return)
        member_core = 0.55 * completion + 0.30 * duration + 0.15 * return_probability
        assert choice_log_probability is not None
        return {
            "completion_utility": completion.mean(dim=0),
            "duration_utility": duration.mean(dim=0),
            "return_probability": return_probability.mean(dim=0),
            "choice_log_probability": choice_log_probability,
            "epistemic_uncertainty": member_core.std(dim=0, unbiased=False),
        }

    @torch.no_grad()
    def predict_candidates(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        action_ids: Tensor,
        content_type_ids: Tensor,
        item_numeric_features: Tensor,
    ) -> dict[str, Tensor]:
        """Average calibrated predictions for a candidate set per user state."""
        member_completion: list[Tensor] = []
        member_duration: list[Tensor] = []
        member_return: list[Tensor] = []
        for index, model in enumerate(self.models):
            output = model.predict_candidates(
                input_ids,
                attention_mask,
                action_ids,
                content_type_ids,
                item_numeric_features,
            )
            completion_probability = (
                output["completion_logits"] / self.completion_temperatures[index]
            ).softmax(dim=-1)
            member_completion.append(
                (completion_probability * model.completion_utility_values).sum(dim=-1)
            )
            member_duration.append(output["duration_utility"])
            member_return.append(
                (output["return_logit"] / self.return_temperatures[index]).sigmoid()
            )
        completion = torch.stack(member_completion)
        duration = torch.stack(member_duration)
        return_probability = torch.stack(member_return)
        member_core = 0.55 * completion + 0.30 * duration + 0.15 * return_probability
        return {
            "completion_utility": completion.mean(dim=0),
            "duration_utility": duration.mean(dim=0),
            "return_probability": return_probability.mean(dim=0),
            "epistemic_uncertainty": member_core.std(dim=0, unbiased=False),
        }

    @staticmethod
    def conservative_reward(
        predictions: dict[str, Tensor],
    ) -> Tensor:
        return compose_conservative_reward(
            predictions,
            epistemic_uncertainty=predictions["epistemic_uncertainty"],
        )
