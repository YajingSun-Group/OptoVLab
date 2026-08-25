from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .metrics import interval_metrics, pinball_loss, regression_metrics
from .model import build_oled_gat, quantile_hybrid_loss


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    history: pd.DataFrame


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(
    graphs: list[Data],
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    subset = [graphs[index] for index in indices]
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        output = model(batch)
        predictions.append(output.float().cpu().numpy())
        observed.append(batch.y_raw.float().cpu().numpy().reshape(-1))
        sample_indices.append(batch.sample_row.cpu().numpy().reshape(-1))
    normalized = np.concatenate(predictions)
    raw = normalized * target_std + target_mean
    raw = np.clip(raw, 0.0, 60.0)
    return (
        np.concatenate(sample_indices),
        np.concatenate(observed),
        raw,
    )


def _prediction_metrics(observed: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    quantiles = predictions[:, 1:]
    return {
        "mean_head": regression_metrics(observed, predictions[:, 0]),
        "median_head": regression_metrics(observed, quantiles[:, 1]),
        "interval_10_90": interval_metrics(
            observed,
            quantiles[:, 0],
            quantiles[:, 2],
        ),
        "pinball": {
            "q10": pinball_loss(observed, quantiles[:, 0], 0.1),
            "q50": pinball_loss(observed, quantiles[:, 1], 0.5),
            "q90": pinball_loss(observed, quantiles[:, 2], 0.9),
        },
    }


def train_oled_gat(
    graphs: list[Data],
    manifest: pd.DataFrame,
    vocabulary: dict[str, Any],
    config: dict[str, Any],
    *,
    output_dir: Path,
    evaluate_test: bool = False,
) -> TrainingResult:
    set_random_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OLED-GAT training")
    device = torch.device("cuda")
    model_config = config["model"]
    training_config = config["training"]
    target_stats = vocabulary["numeric_stats"]
    split_indices = {
        split: manifest.index[manifest["split"].eq(split)].tolist()
        for split in ("train", "validation", "test")
    }
    train_loader = _loader(
        graphs,
        split_indices["train"],
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
    )
    validation_loader = _loader(
        graphs,
        split_indices["validation"],
        batch_size=int(training_config["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )
    evaluation_split = "test" if evaluate_test else "validation"
    evaluation_loader = _loader(
        graphs,
        split_indices[evaluation_split],
        batch_size=int(training_config["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )

    model = build_oled_gat(vocabulary, model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.55,
        patience=10,
        min_lr=2e-5,
    )
    use_amp = bool(training_config["mixed_precision"])
    scaler = GradScaler("cuda", enabled=use_amp)
    quantiles = tuple(float(value) for value in model_config["quantiles"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_rows: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_epoch = -1
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        epoch_started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                output = model(batch)
                sample_weights = (
                    batch.sample_weight
                    if bool(training_config.get("use_sample_weights", True))
                    else torch.ones_like(batch.sample_weight)
                )
                loss = quantile_hybrid_loss(
                    output,
                    batch.y,
                    sample_weights,
                    quantiles=quantiles,
                    mean_mse_weight=float(training_config["mean_mse_weight"]),
                    quantile_loss_weight=float(
                        training_config["quantile_loss_weight"]
                    ),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training_config["gradient_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        _, validation_observed, validation_quantiles = predict(
            model,
            validation_loader,
            device=device,
            target_mean=float(target_stats["target_mean"]),
            target_std=float(target_stats["target_std"]),
        )
        validation_metrics = _prediction_metrics(
            validation_observed,
            validation_quantiles,
        )
        validation_rmse = validation_metrics["mean_head"]["rmse"]
        scheduler.step(validation_rmse)
        improved = validation_rmse < best_rmse - 1e-4
        if improved:
            best_rmse = validation_rmse
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "vocabulary": vocabulary,
                    "seed": int(config["seed"]),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_rmse": validation_rmse,
                "validation_mae": validation_metrics["mean_head"]["mae"],
                "validation_r2": validation_metrics["mean_head"]["r2"],
                "validation_coverage": validation_metrics["interval_10_90"][
                    "coverage"
                ],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
                "improved": improved,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
                f"val_rmse={validation_rmse:.4f} "
                f"val_r2={validation_metrics['mean_head']['r2']:.4f} "
                f"coverage={validation_metrics['interval_10_90']['coverage']:.3f}"
            )
        if stale_epochs >= int(training_config["early_stopping_patience"]):
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    sample_rows, observed, model_predictions = predict(
        model,
        evaluation_loader,
        device=device,
        target_mean=float(target_stats["target_mean"]),
        target_std=float(target_stats["target_std"]),
    )
    predictions = manifest.iloc[sample_rows][
        ["id", "paper_id", "doi", "eqe_max", "split"]
    ].copy()
    predictions["mean"] = model_predictions[:, 0]
    predictions["q10"] = model_predictions[:, 1]
    predictions["q50"] = model_predictions[:, 2]
    predictions["q90"] = model_predictions[:, 3]
    metrics = {
        "evaluation_split": evaluation_split,
        "device_count": int(len(predictions)),
        "paper_count": int(predictions["paper_id"].nunique()),
        "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        **_prediction_metrics(observed, model_predictions),
    }
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "training_history.csv", index=False)
    predictions.to_csv(
        output_dir
        / (
            "test_predictions.csv"
            if evaluate_test
            else "validation_predictions.csv"
        ),
        index=False,
    )
    metrics_path = output_dir / (
        "test_metrics.json" if evaluate_test else "validation_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TrainingResult(
        metrics=metrics,
        predictions=predictions,
        history=history,
    )


def fit_fixed_oled_gat(
    graphs: list[Data],
    manifest: pd.DataFrame,
    vocabulary: dict[str, Any],
    config: dict[str, Any],
    *,
    output_dir: Path,
    epochs: int,
) -> TrainingResult:
    """Refit a selected architecture on train plus validation for fixed epochs."""
    set_random_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OLED-GAT training")
    device = torch.device("cuda")
    model_config = config["model"]
    training_config = config["training"]
    target_stats = vocabulary["numeric_stats"]
    train_indices = manifest.index[manifest["split"].eq("train")].tolist()
    test_indices = manifest.index[manifest["split"].eq("test")].tolist()
    train_loader = _loader(
        graphs,
        train_indices,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
    )
    test_loader = _loader(
        graphs,
        test_indices,
        batch_size=int(training_config["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )
    model = build_oled_gat(vocabulary, model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    milestones = sorted(
        {
            max(1, int(round(epochs * 0.35))),
            max(2, int(round(epochs * 0.65))),
        }
    )
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.55)
    use_amp = bool(training_config["mixed_precision"])
    scaler = GradScaler("cuda", enabled=use_amp)
    quantiles = tuple(float(value) for value in model_config["quantiles"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        epoch_started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                output = model(batch)
                sample_weights = (
                    batch.sample_weight
                    if bool(training_config.get("use_sample_weights", True))
                    else torch.ones_like(batch.sample_weight)
                )
                loss = quantile_hybrid_loss(
                    output,
                    batch.y,
                    sample_weights,
                    quantiles=quantiles,
                    mean_mse_weight=float(training_config["mean_mse_weight"]),
                    quantile_loss_weight=float(
                        training_config["quantile_loss_weight"]
                    ),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training_config["gradient_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history_rows.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"refit_epoch={epoch:03d} loss={row['train_loss']:.4f} "
                f"lr={row['learning_rate']:.6g}"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "vocabulary": vocabulary,
            "seed": int(config["seed"]),
            "epoch": epochs,
            "training_protocol": "fixed_train_plus_validation_refit",
        },
        output_dir / "best_model.pt",
    )
    sample_rows, observed, model_predictions = predict(
        model,
        test_loader,
        device=device,
        target_mean=float(target_stats["target_mean"]),
        target_std=float(target_stats["target_std"]),
    )
    predictions = manifest.iloc[sample_rows][
        ["id", "paper_id", "doi", "eqe_max", "split"]
    ].copy()
    predictions["mean"] = model_predictions[:, 0]
    predictions["q10"] = model_predictions[:, 1]
    predictions["q50"] = model_predictions[:, 2]
    predictions["q90"] = model_predictions[:, 3]
    metrics = {
        "evaluation_split": "test",
        "training_protocol": "fixed_train_plus_validation_refit",
        "device_count": int(len(predictions)),
        "paper_count": int(predictions["paper_id"].nunique()),
        "epochs": epochs,
        "training_seconds": time.perf_counter() - started,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        **_prediction_metrics(observed, model_predictions),
    }
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "training_history.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TrainingResult(
        metrics=metrics,
        predictions=predictions,
        history=history,
    )
