"""Supervised training and full-catalog evaluation for GPTRec-base."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from gptrec.data.sequence_dataset import (
    GPTRecEvalDataset,
    GPTRecEventStore,
    GPTRecTrainDataset,
    SequenceMode,
)
from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    name: str
    seed: int
    mode: SequenceMode
    events_path: Path
    manifest_path: Path
    model_config_path: Path
    checkpoint_dir: Path
    report_dir: Path
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    log_every_steps: int
    num_workers: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_min_epochs: int
    cutoffs: tuple[int, ...]
    filter_seen_items: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        experiment = document["experiment"]
        paths = document["paths"]
        training = document["training"]
        evaluation = document["evaluation"]
        config = cls(
            name=str(experiment["name"]),
            seed=int(experiment["seed"]),
            mode=str(experiment["mode"]),
            events_path=Path(paths["events"]),
            manifest_path=Path(paths["manifest"]),
            model_config_path=Path(paths["model_config"]),
            checkpoint_dir=Path(paths["checkpoint_dir"]),
            report_dir=Path(paths["report_dir"]),
            epochs=int(training["epochs"]),
            train_batch_size=int(training["train_batch_size"]),
            eval_batch_size=int(training["eval_batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            warmup_ratio=float(training["warmup_ratio"]),
            max_grad_norm=float(training["max_grad_norm"]),
            log_every_steps=int(training["log_every_steps"]),
            num_workers=int(training["num_workers"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            early_stopping_min_delta=float(training["early_stopping_min_delta"]),
            early_stopping_min_epochs=int(training["early_stopping_min_epochs"]),
            cutoffs=tuple(sorted({int(k) for k in evaluation["cutoffs"]})),
            filter_seen_items=bool(evaluation["filter_seen_items"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in ("standard", "day_aware"):
            raise ValueError(f"unknown mode: {self.mode}")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.train_batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if not 1 <= self.early_stopping_min_epochs <= self.epochs:
            raise ValueError("early_stopping_min_epochs must be between 1 and epochs")
        if not self.cutoffs or self.cutoffs[0] < 1:
            raise ValueError("evaluation cutoffs must be positive")
        if 10 not in self.cutoffs:
            raise ValueError("evaluation cutoffs must contain 10 for model selection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "mode": self.mode,
            "events_path": str(self.events_path),
            "manifest_path": str(self.manifest_path),
            "model_config_path": str(self.model_config_path),
            "checkpoint_dir": str(self.checkpoint_dir),
            "report_dir": str(self.report_dir),
            "epochs": self.epochs,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "log_every_steps": self.log_every_steps,
            "num_workers": self.num_workers,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_min_epochs": self.early_stopping_min_epochs,
            "cutoffs": list(self.cutoffs),
            "filter_seen_items": self.filter_seen_items,
        }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_loaders(
    config: TrainingConfig,
    model_config: GPTRecBaseConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    store = GPTRecEventStore.from_parquet(config.events_path, seed=config.seed)
    train_dataset = GPTRecTrainDataset(
        store,
        max_sequence_length=model_config.max_sequence_length,
        mode=config.mode,
        pad_token_id=model_config.pad_token_id,
    )
    validation_dataset = GPTRecEvalDataset(
        store,
        split="validation",
        max_sequence_length=model_config.max_sequence_length,
        mode=config.mode,
        pad_token_id=model_config.pad_token_id,
    )
    test_dataset = GPTRecEvalDataset(
        store,
        split="test",
        max_sequence_length=model_config.max_sequence_length,
        mode=config.mode,
        pad_token_id=model_config.pad_token_id,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, validation_loader, test_loader


@torch.no_grad()
def evaluate(
    model: GPTRecBase,
    loader: DataLoader,
    *,
    device: torch.device,
    cutoffs: tuple[int, ...],
    filter_seen_items: bool,
) -> dict[str, float]:
    model.eval()
    max_k = max(cutoffs)
    totals = {f"recall@{k}": 0.0 for k in cutoffs}
    totals.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    loss_sum = 0.0
    example_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target_id"].to(device)
        logits = model.next_item_logits(input_ids, attention_mask)

        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
        example_count += int(targets.size(0))

        ranking_logits = logits.clone()
        target_scores = ranking_logits.gather(1, targets.unsqueeze(1))
        ranking_logits[:, 0] = -torch.inf
        if filter_seen_items:
            ranking_logits.scatter_(1, input_ids, -torch.inf)
            ranking_logits.scatter_(1, targets.unsqueeze(1), target_scores)

        available_k = min(max_k, ranking_logits.size(1) - 1)
        top_items = ranking_logits.topk(available_k, dim=1).indices
        hits = top_items.eq(targets.unsqueeze(1))

        for cutoff in cutoffs:
            k = min(cutoff, available_k)
            hits_at_k = hits[:, :k]
            totals[f"recall@{cutoff}"] += float(hits_at_k.any(dim=1).sum().cpu())
            discounts = torch.log2(
                torch.arange(2, k + 2, device=device, dtype=torch.float32)
            )
            ndcg = (hits_at_k.float() / discounts).sum(dim=1)
            totals[f"ndcg@{cutoff}"] += float(ndcg.sum().cpu())

    if example_count == 0:
        raise ValueError("evaluation dataset contains no eligible targets")
    metrics = {name: value / example_count for name, value in totals.items()}
    metrics["loss"] = loss_sum / example_count
    metrics["examples"] = float(example_count)
    return metrics


def save_history_plots(history: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["validation_recall@10"], marker="o", label="Recall@10")
    axes[1].plot(epochs, history["validation_ndcg@10"], marker="o", label="NDCG@10")
    axes[1].set(title="Validation ranking quality", xlabel="Epoch", ylabel="Metric")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(report_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run_training(config: TrainingConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device()

    with config.manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    vocab_size = int(manifest["vocabulary"]["vocab_size_with_pad"])
    model_config = GPTRecBaseConfig.from_yaml(
        config.model_config_path,
        vocab_size=vocab_size,
    )
    train_loader, validation_loader, test_loader = create_loaders(config, model_config)

    model = GPTRecBase(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = math.floor(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    history: dict[str, Any] = {
        "epochs": [],
        "train_loss": [],
        "validation_loss": [],
        "validation_recall@10": [],
        "validation_ndcg@10": [],
        "step_log": [],
    }
    best_ndcg = -math.inf
    best_epoch = 0
    global_step = 0
    epochs_without_improvement = 0
    stopped_early = False
    best_path = config.checkpoint_dir / "best.pt"

    print(f"experiment={config.name}")
    print(f"device={device.type}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"train_batches={len(train_loader):,}")
    print(f"validation_examples={len(validation_loader.dataset):,}")
    print(f"test_examples={len(test_loader.dataset):,}")
    print(f"epochs={config.epochs}")
    print(f"total_steps={total_steps:,}")
    print(f"warmup_steps={warmup_steps:,}")
    print(
        "early_stopping="
        f"patience={config.early_stopping_patience},"
        f"min_delta={config.early_stopping_min_delta},"
        f"min_epochs={config.early_stopping_min_epochs}"
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_target_count = 0
        epoch_started_at = time.perf_counter()

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch}, batch={batch_index}")
            loss.backward()
            gradient_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            target_count = int(batch["labels"][:, 1:].ne(-100).sum().item())
            epoch_loss_sum += float(loss.detach().cpu()) * target_count
            epoch_target_count += target_count

            if global_step == 1 or global_step % config.log_every_steps == 0:
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                }
                history["step_log"].append(record)
                print(
                    f"epoch={epoch} step={global_step}/{total_steps} "
                    f"loss={record['loss']:.4f} lr={record['learning_rate']:.7f}"
                )

        train_loss = epoch_loss_sum / epoch_target_count
        validation_metrics = evaluate(
            model,
            validation_loader,
            device=device,
            cutoffs=config.cutoffs,
            filter_seen_items=config.filter_seen_items,
        )
        epoch_minutes = (time.perf_counter() - epoch_started_at) / 60

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_recall@10"].append(validation_metrics["recall@10"])
        history["validation_ndcg@10"].append(validation_metrics["ndcg@10"])

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config.to_dict(),
            "training_config": config.to_dict(),
            "validation_metrics": validation_metrics,
        }
        epoch_path = config.checkpoint_dir / f"epoch_{epoch:02d}.pt"
        torch.save(checkpoint, epoch_path)
        improved = (
            validation_metrics["ndcg@10"]
            > best_ndcg + config.early_stopping_min_delta
        )
        if improved:
            best_ndcg = validation_metrics["ndcg@10"]
            best_epoch = epoch
            epochs_without_improvement = 0
            shutil.copy2(epoch_path, best_path)
        else:
            epochs_without_improvement += 1

        save_json(config.report_dir / "training_history.json", history)
        save_history_plots(history, config.report_dir)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_recall@10={validation_metrics['recall@10']:.6f} "
            f"val_ndcg@10={validation_metrics['ndcg@10']:.6f} "
            f"minutes={epoch_minutes:.1f}"
        )

        if (
            epoch >= config.early_stopping_min_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"early_stopping at epoch={epoch}; "
                f"best_epoch={best_epoch} best_val_ndcg@10={best_ndcg:.6f}"
            )
            break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        cutoffs=config.cutoffs,
        filter_seen_items=config.filter_seen_items,
    )
    result = {
        "experiment": config.name,
        "epochs_completed": len(history["epochs"]),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_ndcg@10": best_ndcg,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_path),
    }
    save_json(config.report_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

