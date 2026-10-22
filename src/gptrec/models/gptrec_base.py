"""Atomic-item GPT-2 baseline for next-item prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass(frozen=True, slots=True)
class GPTRecBaseConfig:
    """Configuration for the randomly initialized GPTRec-base model."""

    vocab_size: int
    name: str = "GPTRec-base"
    max_sequence_length: int = 32
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must include padding and at least one item")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    @classmethod
    def from_yaml(cls, path: str | Path, *, vocab_size: int) -> GPTRecBaseConfig:
        """Load model settings while injecting the train-derived vocabulary size."""
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        values: dict[str, Any] = document.get("model", {})
        return cls(vocab_size=vocab_size, **values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPTRecBase(nn.Module):
    """GPT-2 architecture trained from scratch with one token per item.

    The class deliberately calls ``GPT2LMHeadModel(config)`` rather than
    ``from_pretrained``: no text tokenizer or pretrained text weights are used.
    Hugging Face shifts ``labels`` internally for causal next-token loss.
    Padding positions in labels must be replaced with ``-100``.
    """

    def __init__(self, config: GPTRecBaseConfig) -> None:
        super().__init__()
        self.project_config = config
        hf_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_sequence_length,
            n_ctx=config.max_sequence_length,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
            activation_function="gelu_new",
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            pad_token_id=config.pad_token_id,
            bos_token_id=None,
            eos_token_id=None,
            tie_word_embeddings=config.tie_word_embeddings,
            use_cache=False,
        )
        self.model = GPT2LMHeadModel(hf_config)
        # Recent Transformers versions cannot infer the loss family from the
        # GPT2LMHeadModel class name and otherwise emit a misleading warning.
        self.model.loss_type = "ForCausalLM"

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> CausalLMOutputWithCrossAttentions:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    @staticmethod
    def labels_from_input(input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Create labels for observed item-to-next-item transitions.

        Padding and the first observed item are ignored. The first item has no
        preceding observed item and therefore must not become a popularity-only
        target merely because a sequence was padded on the left.
        """
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)
        first_positions = attention_mask.long().argmax(dim=1, keepdim=True)
        return labels.scatter(1, first_positions, -100)

    @torch.no_grad()
    def next_item_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return full-catalog logits at the last non-padding position."""
        transformer_output = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(attention_mask)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        if torch.any(last_positions < 0):
            raise ValueError("Every sequence must contain at least one non-padding item")
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden_states = transformer_output.last_hidden_state[
            batch_indices,
            last_positions,
        ]
        return self.model.lm_head(last_hidden_states)

