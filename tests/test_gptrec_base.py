import torch

from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig


def test_gptrec_base_shapes_and_tied_embeddings() -> None:
    config = GPTRecBaseConfig(
        vocab_size=101,
        max_sequence_length=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    model = GPTRecBase(config)
    model.eval()
    input_ids = torch.tensor([[0, 0, 4, 8, 15], [0, 2, 3, 5, 7]])
    attention_mask = input_ids.ne(config.pad_token_id).long()
    labels = model.labels_from_input(input_ids, attention_mask)

    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    next_logits = model.next_item_logits(input_ids, attention_mask)

    assert output.loss.ndim == 0
    assert output.logits.shape == (2, 5, config.vocab_size)
    assert next_logits.shape == (2, config.vocab_size)
    assert torch.equal(next_logits, output.logits[:, -1])
    assert torch.equal(labels[:, :2], torch.full((2, 2), -100))
    assert model.model.transformer.wte.weight.data_ptr() == model.model.lm_head.weight.data_ptr()

