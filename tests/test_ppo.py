import torch

from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig
from gptrec.models.ppo_policy import GPTRecActorCritic, append_policy_step
from gptrec.training.ppo_trainer import (
    RewardNormalizer,
    compose_normalized_reward,
    generalized_advantage_estimate,
)


def _policy() -> GPTRecActorCritic:
    backbone = GPTRecBase(
        GPTRecBaseConfig(
            vocab_size=20,
            max_sequence_length=4,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            dropout=0.0,
        )
    )
    return GPTRecActorCritic(backbone)


def test_actor_critic_preserves_base_logits_before_feedback_training() -> None:
    policy = _policy().eval()
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    attention_mask = input_ids.ne(0).long()
    feedback = torch.randn(2, 4, 3)

    with torch.no_grad():
        base_logits = policy.backbone.next_item_logits(input_ids, attention_mask)
        policy_logits, values = policy(input_ids, attention_mask, feedback)

    assert torch.allclose(policy_logits, base_logits)
    assert values.shape == (2,)


def test_append_policy_step_handles_space_and_full_histories() -> None:
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = input_ids.ne(0).long()
    feedback = torch.zeros(2, 3, 3)
    action_feedback = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    next_ids, next_mask, next_feedback = append_policy_step(
        input_ids,
        attention_mask,
        feedback,
        torch.tensor([6, 7]),
        action_feedback,
    )

    assert next_ids.tolist() == [[1, 2, 6], [4, 5, 7]]
    assert next_mask.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert torch.allclose(next_feedback[0, 2], action_feedback[0])
    assert torch.allclose(next_feedback[1, 2], action_feedback[1])


def test_reward_normalization_and_gae_are_finite() -> None:
    samples = {
        "completion_utility": torch.tensor([0.2, 0.4, 0.8]),
        "duration_utility": torch.tensor([0.1, 0.5, 0.9]),
        "return_probability": torch.tensor([0.3, 0.6, 0.9]),
        "epistemic_uncertainty": torch.tensor([0.01, 0.02, 0.04]),
    }
    normalizer = RewardNormalizer.fit(samples)
    rewards = compose_normalized_reward(samples, normalizer, (0.55, 0.30, 0.15, 0.25))
    reward_matrix = rewards.unsqueeze(0).repeat(2, 1)
    values = torch.zeros_like(reward_matrix)
    advantages, returns = generalized_advantage_estimate(
        reward_matrix,
        values,
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert torch.isfinite(rewards).all()
    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages, returns)
    assert advantages.shape == (2, 3)
