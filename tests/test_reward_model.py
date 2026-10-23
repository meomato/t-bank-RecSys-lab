import torch

from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig
from gptrec.models.reward_model import (
    KIONRewardModel,
    RewardModelConfig,
    compose_conservative_reward,
)


def test_reward_model_heads_loss_and_composition() -> None:
    backbone = GPTRecBase(
        GPTRecBaseConfig(
            vocab_size=20,
            max_sequence_length=4,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
        )
    )
    model = KIONRewardModel(
        backbone,
        RewardModelConfig(
            expert_count=2,
            expert_hidden_size=24,
            tower_hidden_size=12,
            content_type_count=2,
            content_type_embedding_size=4,
        ),
    )
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    attention_mask = input_ids.ne(0).long()
    output = model(
        input_ids,
        attention_mask,
        torch.tensor([6, 7]),
        torch.tensor([0, 1]),
        torch.tensor([[0.1, 0.8], [-0.1, 0.2]]),
    )
    loss, parts = model.loss(
        output,
        torch.tensor([2, -1]),
        torch.tensor([0.4, 0.7]),
        torch.tensor([1.0, 0.0]),
    )
    reward = compose_conservative_reward(
        output,
        epistemic_uncertainty=torch.tensor([0.1, 0.2]),
    )
    candidate_output = model.predict_candidates(
        input_ids,
        attention_mask,
        torch.tensor([[6, 8, 9], [7, 8, 9]]),
        torch.tensor([[0, 1, 0], [1, 0, 1]]),
        torch.tensor(
            [
                [[0.1, 0.8], [0.2, 0.7], [0.3, 0.6]],
                [[-0.1, 0.2], [0.0, 0.3], [0.1, 0.4]],
            ]
        ),
    )

    assert output["completion_logits"].shape == (2, 6)
    assert output["choice_log_probability"].shape == (2,)
    assert torch.all((output["duration_utility"] > 0) & (output["duration_utility"] < 1))
    assert set(parts) == {"completion", "duration", "return"}
    assert torch.isfinite(loss)
    assert reward.shape == (2,)
    assert candidate_output["completion_utility"].shape == (2, 3)
    assert candidate_output["completion_logits"].shape == (2, 3, 6)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
