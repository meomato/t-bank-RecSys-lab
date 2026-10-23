"""Training, calibration and evaluation for the KION reward model."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from gptrec.data.reward_dataset import KIONRewardDataset, RewardEventStore
from gptrec.models.reward_model import KIONRewardModel, RewardModelConfig


@dataclass(frozen=True, slots=True)
class RewardTrainingConfig:
    events_path: Path
    item_features_path: Path
    backbone_checkpoint: Path
    model_config_path: Path
    checkpoint_dir: Path
    report_dir: Path
    seeds: tuple[int, ...]
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    patience: int
    log_every_steps: int
    num_workers: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> RewardTrainingConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        paths = document["paths"]
        training = document["training"]
        config = cls(
            events_path=Path(paths["events"]),
            item_features_path=Path(paths["item_features"]),
            backbone_checkpoint=Path(paths["backbone_checkpoint"]),
            model_config_path=Path(paths["model_config"]),
            checkpoint_dir=Path(paths["checkpoint_dir"]),
            report_dir=Path(paths["report_dir"]),
            seeds=tuple(int(seed) for seed in training["seeds"]),
            epochs=int(training["epochs"]),
            train_batch_size=int(training["train_batch_size"]),
            eval_batch_size=int(training["eval_batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            max_grad_norm=float(training["max_grad_norm"]),
            patience=int(training["patience"]),
            log_every_steps=int(training.get("log_every_steps", 100)),
            num_workers=int(training.get("num_workers", 0)),
        )
        if (
            not config.seeds
            or config.epochs < 1
            or config.patience < 1
            or config.log_every_steps < 1
        ):
            raise ValueError(
                "seeds, epochs, patience and log_every_steps must be positive"
            )
        return config

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "events_path",
            "item_features_path",
            "backbone_checkpoint",
            "model_config_path",
            "checkpoint_dir",
            "report_dir",
        ):
            value[key] = str(value[key])
        value["seeds"] = list(value["seeds"])
        return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _log(message: str) -> None:
    print(message, flush=True)


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _forward(model: KIONRewardModel, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["action_id"],
        batch["content_type_id"],
        batch["item_numeric_features"],
        compute_choice=False,
    )


def _ece(probabilities: Tensor, labels: Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    total = probabilities.numel()
    error = torch.zeros((), device=probabilities.device)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities > lower) & (probabilities <= upper)
        if mask.any():
            error += mask.float().mean() * torch.abs(
                probabilities[mask].mean() - labels[mask].float().mean()
            )
    return float(error.cpu()) if total else math.nan


@torch.no_grad()
def evaluate_reward_model(
    model: KIONRewardModel,
    loader: DataLoader,
    *,
    device: torch.device,
    completion_temperature: float = 1.0,
    return_temperature: float = 1.0,
) -> dict[str, float]:
    model.eval()
    completion_logits: list[Tensor] = []
    completion_labels: list[Tensor] = []
    duration_means: list[Tensor] = []
    duration_targets: list[Tensor] = []
    return_probabilities: list[Tensor] = []
    return_labels: list[Tensor] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        output = _forward(model, batch)
        valid = batch["completion_class"].ge(0)
        completion_logits.append(output["completion_logits"][valid].cpu())
        completion_labels.append(batch["completion_class"][valid].cpu())
        duration_means.append(output["duration_utility"].cpu())
        duration_targets.append(batch["duration_utility"].cpu())
        return_probabilities.append(
            (output["return_logit"] / return_temperature).sigmoid().cpu()
        )
        return_labels.append(batch["return_7d"].cpu())

    logits = torch.cat(completion_logits) / completion_temperature
    completion = torch.cat(completion_labels)
    completion_prob = logits.softmax(dim=-1)
    duration_mean = torch.cat(duration_means)
    duration_target = torch.cat(duration_targets)
    return_probability = torch.cat(return_probabilities)
    return_target = torch.cat(return_labels)
    completion_confidence, completion_prediction = completion_prob.max(dim=-1)
    return {
        "examples": float(len(duration_target)),
        "completion_nll": float(F.cross_entropy(logits, completion)),
        "completion_accuracy": float(completion_prediction.eq(completion).float().mean()),
        "completion_ece": _ece(
            completion_confidence, completion_prediction.eq(completion).float()
        ),
        "duration_mae": float(torch.abs(duration_mean - duration_target).mean()),
        "duration_rmse": float(
            torch.sqrt(torch.square(duration_mean - duration_target).mean())
        ),
        "return_log_loss": float(
            F.binary_cross_entropy(return_probability, return_target)
        ),
        "return_brier": float(torch.square(return_probability - return_target).mean()),
        "return_ece": _ece(return_probability, return_target),
    }


@torch.no_grad()
def _collect_calibration_logits(
    model: KIONRewardModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    model.eval()
    completion_logits: list[Tensor] = []
    completion_labels: list[Tensor] = []
    return_logits: list[Tensor] = []
    return_labels: list[Tensor] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        output = _forward(model, batch)
        valid = batch["completion_class"].ge(0)
        completion_logits.append(output["completion_logits"][valid])
        completion_labels.append(batch["completion_class"][valid])
        return_logits.append(output["return_logit"])
        return_labels.append(batch["return_7d"])
    return (
        torch.cat(completion_logits),
        torch.cat(completion_labels),
        torch.cat(return_logits),
        torch.cat(return_labels),
    )


def _fit_temperature(logits: Tensor, labels: Tensor, *, binary: bool) -> float:
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        if binary:
            loss = F.binary_cross_entropy_with_logits(logits / temperature, labels.float())
        else:
            loss = F.cross_entropy(logits / temperature, labels.long())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())


def fit_temperatures(
    model: KIONRewardModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    completion_logits, completion_labels, return_logits, return_labels = (
        _collect_calibration_logits(model, loader, device)
    )
    return {
        "completion": _fit_temperature(
            completion_logits, completion_labels, binary=False
        ),
        "return": _fit_temperature(return_logits, return_labels, binary=True),
    }


def _make_loaders(
    config: RewardTrainingConfig,
    *,
    seed: int,
    max_examples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    store = RewardEventStore.from_parquet(config.events_path, config.item_features_path)
    datasets = {
        split: KIONRewardDataset(store, split=split, max_examples=max_examples)
        for split in ("train", "calibration", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            datasets["train"],
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
        ),
        DataLoader(
            datasets["calibration"],
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        DataLoader(
            datasets["test"],
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    )


def train_reward_member(
    config: RewardTrainingConfig,
    *,
    seed: int,
    max_examples: int | None = None,
    max_steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    _seed_everything(seed)
    device = _choose_device()
    _log(f"member_setup seed={seed} device={device.type} stage=load_data_and_model")
    model_config = RewardModelConfig.from_yaml(config.model_config_path)
    train_loader, calibration_loader, test_loader = _make_loaders(
        config, seed=seed, max_examples=max_examples
    )
    model = KIONRewardModel.from_gptrec_checkpoint(
        config.backbone_checkpoint, model_config, map_location=device
    ).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    member_dir = config.checkpoint_dir / f"seed_{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    best_path = member_dir / "best.pt"
    best_objective = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    train_batches = len(train_loader)
    if max_steps_per_epoch is not None:
        train_batches = min(train_batches, max_steps_per_epoch)
    _log(
        f"member_start seed={seed} device={device.type} "
        f"train_examples={len(train_loader.dataset):,} train_batches={train_batches:,} "
        f"calibration_examples={len(calibration_loader.dataset):,} "
        f"test_examples={len(test_loader.dataset):,}"
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        if model.config.freeze_backbone:
            model.backbone.eval()
        started_at = time.perf_counter()
        total_loss = 0.0
        batches = 0
        _log(
            f"stage=train seed={seed} epoch={epoch}/{config.epochs} "
            f"batches={train_batches:,}"
        )
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss, _ = model.loss(
                output,
                batch["completion_class"],
                batch["duration_utility"],
                batch["return_7d"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite reward loss at epoch={epoch}, step={step}")
            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            if (
                step == 1
                or step % config.log_every_steps == 0
                or step == train_batches
            ):
                elapsed_seconds = time.perf_counter() - started_at
                batches_per_second = step / max(elapsed_seconds, 1e-9)
                eta_seconds = (train_batches - step) / max(batches_per_second, 1e-9)
                _log(
                    f"stage=train seed={seed} epoch={epoch}/{config.epochs} "
                    f"batch={step:,}/{train_batches:,} "
                    f"progress={100 * step / train_batches:5.1f}% "
                    f"loss_avg={total_loss / batches:.4f} "
                    f"elapsed={_format_duration(elapsed_seconds)} "
                    f"eta={_format_duration(eta_seconds)}"
                )
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
        _log(
            f"stage=calibration seed={seed} epoch={epoch}/{config.epochs} "
            f"batches={len(calibration_loader):,}"
        )
        metrics = evaluate_reward_model(model, calibration_loader, device=device)
        objective = (
            metrics["completion_nll"]
            + metrics["duration_mae"]
            + metrics["return_log_loss"]
        )
        record = {
            "epoch": float(epoch),
            "train_loss": total_loss / batches,
            "calibration_objective": objective,
            "minutes": (time.perf_counter() - started_at) / 60,
        }
        history.append(record)
        _log(
            f"seed={seed} epoch={epoch} train_loss={record['train_loss']:.4f} "
            f"calibration_objective={objective:.4f} "
            f"minutes={record['minutes']:.1f}"
        )
        if objective < best_objective:
            best_objective = objective
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "training_config": config.to_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "calibration_metrics": metrics,
                },
                best_path,
            )
            _log(
                f"checkpoint_saved seed={seed} epoch={epoch} "
                f"objective={objective:.4f} path={best_path}"
            )
        else:
            epochs_without_improvement += 1
            _log(
                f"no_improvement seed={seed} epoch={epoch} "
                f"count={epochs_without_improvement}/{config.patience}"
            )
            if epochs_without_improvement >= config.patience:
                _log(f"early_stopping seed={seed} epoch={epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _log(
        f"stage=temperature_fit seed={seed} "
        f"batches={len(calibration_loader):,}"
    )
    temperatures = fit_temperatures(model, calibration_loader, device)
    _log(
        f"temperatures seed={seed} completion={temperatures['completion']:.4f} "
        f"return={temperatures['return']:.4f}"
    )
    _log(f"stage=test seed={seed} batches={len(test_loader):,}")
    test_metrics = evaluate_reward_model(
        model,
        test_loader,
        device=device,
        completion_temperature=temperatures["completion"],
        return_temperature=temperatures["return"],
    )
    checkpoint["temperatures"] = temperatures
    checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, best_path)
    _log(f"member_complete seed={seed} checkpoint={best_path}")
    return {
        "seed": seed,
        "checkpoint": str(best_path),
        "history": history,
        "temperatures": temperatures,
        "test_metrics": test_metrics,
    }


def run_reward_training(config: RewardTrainingConfig) -> dict[str, Any]:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    _log(
        f"ensemble_start members={len(config.seeds)} seeds={list(config.seeds)} "
        f"epochs={config.epochs}"
    )
    members: list[dict[str, Any]] = []
    for member_index, seed in enumerate(config.seeds, start=1):
        _log(f"ensemble_member={member_index}/{len(config.seeds)} seed={seed}")
        members.append(train_reward_member(config, seed=seed))
    metric_names = members[0]["test_metrics"]
    aggregate = {
        name: {
            "mean": float(np.mean([member["test_metrics"][name] for member in members])),
            "std": float(np.std([member["test_metrics"][name] for member in members])),
        }
        for name in metric_names
    }
    result = {"members": members, "aggregate_test_metrics": aggregate}
    with (config.report_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    _log(f"ensemble_complete report={config.report_dir / 'result.json'}")
    return result
