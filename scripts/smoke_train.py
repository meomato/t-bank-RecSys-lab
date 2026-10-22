"""Run a short GPTRec-base training smoke test and estimate epoch duration."""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from gptrec.data.sequence_dataset import GPTRecEventStore, GPTRecTrainDataset
from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/processed/gptrec_base/events.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/gptrec_base/manifest.json"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/gptrec_base.yaml"),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    torch.manual_seed(args.seed)
    device = choose_device()

    with args.manifest.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    vocab_size = int(manifest["vocabulary"]["vocab_size_with_pad"])
    model_config = GPTRecBaseConfig.from_yaml(
        args.model_config,
        vocab_size=vocab_size,
    )
    store = GPTRecEventStore.from_parquet(args.events, seed=args.seed)
    dataset = GPTRecTrainDataset(
        store,
        max_sequence_length=model_config.max_sequence_length,
        mode="standard",
        pad_token_id=model_config.pad_token_id,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = GPTRecBase(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    model.train()

    print(f"device={device.type}")
    print(f"train_samples={len(dataset):,}")
    print(f"batches_per_epoch={len(loader):,}")
    print(f"smoke_steps={min(args.steps, len(loader)):,}")

    losses: list[float] = []
    processed_examples = 0
    synchronize(device)
    started_at = time.perf_counter()

    for step, batch in enumerate(islice(loader, args.steps), start=1):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        gradient_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        processed_examples += int(batch["input_ids"].size(0))
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                f"step={step} loss={losses[-1]:.4f} "
                f"grad_norm={float(gradient_norm.detach().cpu()):.4f}"
            )

    synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    completed_steps = len(losses)
    seconds_per_step = elapsed_seconds / completed_steps
    estimated_epoch_minutes = seconds_per_step * len(loader) / 60

    print(f"mean_loss={sum(losses) / completed_steps:.4f}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    print(f"steps_per_second={completed_steps / elapsed_seconds:.3f}")
    print(f"examples_per_second={processed_examples / elapsed_seconds:.1f}")
    print(f"estimated_epoch_minutes={math.ceil(estimated_epoch_minutes)}")
    print("[done] smoke test completed; no checkpoint was saved")


if __name__ == "__main__":
    main()

