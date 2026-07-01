from __future__ import annotations

import json
from pathlib import Path
import sys
import copy
import random
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Make src importable when running this script directly
sys.path.append(str(SRC_DIR))

from models.llt import LLT, count_trainable_parameters

# Setup
SEED = 42

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "meeting_room_full_windows_30.npz"

K_VALUES = [1, 5, 15, 25, 35]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_SPLITS_DIR = PROJECT_ROOT / "outputs" / "splits"

SUMMARY_CSV_PATH = (
    OUTPUT_LOGS_DIR
    / "fingerprint_classification_ordered_k_results.csv"
)


def get_split_path(k: int) -> Path:
    return OUTPUT_SPLITS_DIR / f"meeting_room_ordered_k{k}_split.npz"


def get_checkpoint_path(k: int) -> Path:
    return CHECKPOINT_DIR / f"llt_meeting_room_ordered_k{k}_best.pt"


def get_metrics_path(k: int) -> Path:
    return OUTPUT_LOGS_DIR / f"llt_meeting_room_ordered_k{k}_metrics.json"


BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
GRAD_CLIP_NORM = 1.0

NUM_CLASSES = 176

# LLT architecture
PATCH_SIZE = (5, 5)
EMBED_DIM = 32
DEPTH = 2
NUM_HEADS = 2
MLP_RATIO = 2.0
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reproducibility

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load processed CSI windows, class labels and spatial positions.

    Expected arrays inside the .npz file:
        x_windows: [num_samples, 3, 30, 30]
        y_labels: [num_samples]
        grid_positions: [num_samples, 2]
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path)

    x_windows = data["x_windows"]
    y_labels = data["y_labels"]
    grid_positions = data["grid_positions"]

    x_tensor = torch.tensor(x_windows, dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)
    positions_tensor = torch.tensor(grid_positions, dtype=torch.float32)

    return x_tensor, y_tensor, positions_tensor


def load_split_indices(split_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/validation/test split indices from an existing split file.

    This function accepts common key names to stay compatible with previous
    training scripts.
    """

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_data = np.load(split_path)

    train_keys = ["train_idx", "train_indices", "idx_train"]
    val_keys = ["val_idx", "val_indices", "idx_val"]
    test_keys = ["test_idx", "test_indices", "idx_test"]

    train_idx = get_first_existing_array(split_data, train_keys, "train")
    val_idx = get_first_existing_array(split_data, val_keys, "validation")
    test_idx = get_first_existing_array(split_data, test_keys, "test")

    return train_idx, val_idx, test_idx


def get_first_existing_array(
    npz_file: np.lib.npyio.NpzFile,
    possible_keys: list[str],
    split_name: str,
) -> np.ndarray:
    """Return the first existing array among possible key names."""

    for key in possible_keys:
        if key in npz_file:
            return npz_file[key]

    available_keys = list(npz_file.keys())
    raise KeyError(
        f"Could not find {split_name} indices in split file. "
        f"Expected one of {possible_keys}, found {available_keys}."
    )


def build_loaders(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    positions_tensor: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation and test DataLoaders."""

    train_dataset = TensorDataset(
        x_tensor[train_idx],
        y_tensor[train_idx],
        positions_tensor[train_idx],
    )

    val_dataset = TensorDataset(
        x_tensor[val_idx],
        y_tensor[val_idx],
        positions_tensor[val_idx],
    )

    test_dataset = TensorDataset(
        x_tensor[test_idx],
        y_tensor[test_idx],
        positions_tensor[test_idx],
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def build_class_positions(
    y_tensor: torch.Tensor,
    positions_tensor: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Build a class-to-position table.

    class_positions[c] contains the spatial coordinates of reference point c.
    """

    class_positions = torch.zeros(num_classes, 2, dtype=torch.float32)
    class_seen = torch.zeros(num_classes, dtype=torch.bool)

    for label, position in zip(y_tensor, positions_tensor):
        class_id = int(label.item())

        if not class_seen[class_id]:
            class_positions[class_id] = position
            class_seen[class_id] = True

    missing_classes = torch.where(~class_seen)[0]

    if len(missing_classes) > 0:
        raise ValueError(f"Missing spatial positions for classes: {missing_classes.tolist()}")

    return class_positions


# =============================================================================
# METRICS
# =============================================================================

def compute_spatial_error(
    predictions: torch.Tensor,
    true_positions: torch.Tensor,
    class_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Euclidean localization error in grid-coordinate space.

    predictions: [batch_size]
    true_positions: [batch_size, 2]
    class_positions: [num_classes, 2]
    """

    predicted_positions = class_positions[predictions]
    distances = torch.linalg.norm(predicted_positions - true_positions, dim=1)

    return distances


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch, _ in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        if GRAD_CLIP_NORM is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()

        batch_size = x_batch.size(0)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == y_batch).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_positions: torch.Tensor,
) -> dict[str, float]:
    """Evaluate the model on validation or test data."""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_spatial_errors = []

    class_positions = class_positions.to(device)

    for x_batch, y_batch, positions_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        positions_batch = positions_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        batch_size = x_batch.size(0)
        predictions = logits.argmax(dim=1)

        spatial_errors = compute_spatial_error(
            predictions=predictions,
            true_positions=positions_batch,
            class_positions=class_positions,
        )

        total_loss += loss.item() * batch_size
        total_correct += (predictions == y_batch).sum().item()
        total_samples += batch_size

        all_spatial_errors.append(spatial_errors.detach().cpu())

    all_spatial_errors = torch.cat(all_spatial_errors, dim=0)

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "mean_spatial_error": all_spatial_errors.mean().item(),
        "median_spatial_error": torch.quantile(
            all_spatial_errors,
            0.5,
        ).item(),
        "rmse_spatial_error": torch.sqrt(
            torch.mean(all_spatial_errors ** 2)
        ).item(),
    }

    return metrics


def update_summary_csv(
    summary_row: dict,
    summary_csv_path: Path,
) -> None:
    """
    Save or update the final row for one model and one K value.
    """

    fieldnames = [
        "model",
        "k",
        "dataset",
        "experiment",
        "split_file",
        "best_model_file",
        "num_classes",
        "train_samples",
        "val_samples",
        "test_samples",
        "accuracy",
        "mean_grid_error",
        "median_grid_error",
        "rmse_grid_error",
        "trainable_parameters",
        "best_epoch",
        "epochs_ran",
        "early_stopped",
        "training_time_seconds",
        "test_time_seconds",
    ]

    rows = []

    if summary_csv_path.exists():
        with open(
            summary_csv_path,
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            reader = csv.DictReader(input_file)

            for row in reader:
                same_model = row.get("model") == summary_row["model"]
                same_k = str(row.get("k")) == str(summary_row["k"])

                if not (same_model and same_k):
                    rows.append(row)

    rows.append(summary_row)

    with open(
        summary_csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# Main
def train_single_k(
    k: int,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    positions_tensor: torch.Tensor,
) -> None:
    set_seed(SEED)

    split_path = get_split_path(k)
    checkpoint_path = get_checkpoint_path(k)
    metrics_path = get_metrics_path(k)

    print()
    print("=" * 80)
    print(f"LLT ORDERED-K CLASSIFIER TRAINING | K={k}")
    print("=" * 80)
    print("device:", DEVICE)
    print("dataset file:", DATASET_PATH)
    print("split file:", split_path)

    train_idx, val_idx, test_idx = load_split_indices(split_path)
    train_mean = float(x_tensor[train_idx].mean())
    train_std = float(x_tensor[train_idx].std())

    x_tensor_normalized = (x_tensor - train_mean) / (train_std + 1e-8)

    print("\nSPLIT")
    print("Loaded ordered-K split from:", split_path)
    print("k train windows per class:", k)
    print("train samples:", len(train_idx))
    print("val samples:", len(val_idx))
    print("test samples:", len(test_idx))
    print("train mean:", f"{train_mean:.6f}")
    print("train std:", f"{train_std:.6f}")

    train_loader, val_loader, test_loader = build_loaders(
        x_tensor=x_tensor_normalized,
        y_tensor=y_tensor,
        positions_tensor=positions_tensor,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=BATCH_SIZE,
    )

    class_positions = build_class_positions(
        y_tensor=y_tensor,
        positions_tensor=positions_tensor,
        num_classes=NUM_CLASSES,
    )

    model = LLT(
        in_channels=3,
        image_size=(30, 30),
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    print("\nMODEL")
    print("model name: LLT")
    print("patch_size:", PATCH_SIZE)
    print("embed_dim:", EMBED_DIM)
    print("depth:", DEPTH)
    print("num_heads:", NUM_HEADS)
    print("mlp_ratio:", MLP_RATIO)
    print("dropout:", DROPOUT)
    print("trainable parameters:", count_trainable_parameters(model))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1
    best_val_metrics = None
    bad_epochs = 0
    early_stopped = False
    history = []

    print("\nTRAINING")
    print("epochs:", EPOCHS)
    print("batch_size:", BATCH_SIZE)
    print("learning_rate:", LEARNING_RATE)
    print("weight_decay:", WEIGHT_DECAY)
    print("patience:", PATIENCE)

    training_start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            class_positions=class_positions,
        )

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_mean_spatial_error": val_metrics["mean_spatial_error"],
            "val_median_spatial_error": val_metrics["median_spatial_error"],
            "val_rmse_spatial_error": val_metrics["rmse_spatial_error"],
            "learning_rate": current_lr,
        }

        history.append(epoch_metrics)
        improved = val_metrics["loss"] < best_val_loss

        if improved:
            best_val_loss = val_metrics["loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(val_metrics)
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "config": {
                        "patch_size": PATCH_SIZE,
                        "embed_dim": EMBED_DIM,
                        "depth": DEPTH,
                        "num_heads": NUM_HEADS,
                        "mlp_ratio": MLP_RATIO,
                        "dropout": DROPOUT,
                        "num_classes": NUM_CLASSES,
                    },
                    "k": k,
                    "experiment": f"ordered_k{k}_seed42",
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_metrics["accuracy"],
                    "best_val_mean_spatial_error": best_val_metrics["mean_spatial_error"],
                    "best_val_median_spatial_error": best_val_metrics["median_spatial_error"],
                    "best_val_rmse_spatial_error": best_val_metrics["rmse_spatial_error"],
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "split_file": str(split_path),
                    "dataset_file": str(DATASET_PATH),
                },
                checkpoint_path,
            )

            best_marker = "*"
        else:
            bad_epochs += 1
            best_marker = ""

        print(
            f"K={k} "
            f"epoch {epoch:03d}/{EPOCHS} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_mean_err={val_metrics['mean_spatial_error']:.4f} "
            f"val_median_err={val_metrics['median_spatial_error']:.4f} "
            f"val_rmse_err={val_metrics['rmse_spatial_error']:.4f} "
            f"bad={bad_epochs}/{PATIENCE} {best_marker}"
        )

        if bad_epochs >= PATIENCE:
            early_stopped = True
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    training_time_seconds = time.perf_counter() - training_start_time

    if best_state_dict is None:
        raise RuntimeError(f"No best model state was saved during training for K={k}.")

    model.load_state_dict(best_state_dict)

    print("\nBEST CHECKPOINT")
    print("saved to:", checkpoint_path)
    print("best val loss:", best_val_loss)

    test_start_time = time.perf_counter()

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        class_positions=class_positions,
    )

    test_time_seconds = time.perf_counter() - test_start_time

    if best_val_metrics is None:
        raise RuntimeError("Best validation metrics were not saved.")

    metrics_output = {
        "model_name": "LLT",
        "experiment": f"ordered_k{k}_seed42",
        "k": k,
        "dataset_file": str(DATASET_PATH),
        "best_model_file": str(checkpoint_path),
        "split_file": str(split_path),
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "epochs_ran": len(history),
        "early_stopped": early_stopped,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "mlp_ratio": MLP_RATIO,
        "dropout": DROPOUT,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "train_windows_per_class": k,
        "train_mean": train_mean,
        "train_std": train_std,
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_val_mean_spatial_error": best_val_metrics["mean_spatial_error"],
        "best_val_median_spatial_error": best_val_metrics["median_spatial_error"],
        "best_val_rmse_spatial_error": best_val_metrics["rmse_spatial_error"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_mean_spatial_error": test_metrics["mean_spatial_error"],
        "test_median_spatial_error": test_metrics["median_spatial_error"],
        "test_rmse_spatial_error": test_metrics["rmse_spatial_error"],
        "training_time_seconds": training_time_seconds,
        "test_time_seconds": test_time_seconds,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as output_file:
        json.dump(metrics_output, output_file, indent=4)

    summary_row = {
        "model": "LLT",
        "k": k,
        "dataset": "meeting_room_full_windows_30",
        "experiment": f"ordered_k{k}_seed42",
        "split_file": str(split_path),
        "best_model_file": str(checkpoint_path),
        "num_classes": NUM_CLASSES,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "accuracy": test_metrics["accuracy"],
        "mean_grid_error": test_metrics["mean_spatial_error"],
        "median_grid_error": test_metrics["median_spatial_error"],
        "rmse_grid_error": test_metrics["rmse_spatial_error"],
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "early_stopped": early_stopped,
        "training_time_seconds": training_time_seconds,
        "test_time_seconds": test_time_seconds,
    }

    update_summary_csv(
        summary_row=summary_row,
        summary_csv_path=SUMMARY_CSV_PATH,
    )

    print("\nTEST")
    print("K:", k)
    print("test_loss:", f"{test_metrics['loss']:.4f}")
    print("test_accuracy:", f"{test_metrics['accuracy']:.4f}")
    print("test_mean_spatial_error:", f"{test_metrics['mean_spatial_error']:.4f}")
    print("test_median_spatial_error:", f"{test_metrics['median_spatial_error']:.4f}")
    print("test_rmse_spatial_error:", f"{test_metrics['rmse_spatial_error']:.4f}")
    print("training_time_seconds:", f"{training_time_seconds:.2f}")
    print("test_time_seconds:", f"{test_time_seconds:.2f}")
    print("best model saved to:", checkpoint_path)
    print("metrics saved to:", metrics_path)
    print("summary CSV saved to:", SUMMARY_CSV_PATH)


def main() -> None:
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("LLT ORDERED-K CLASSIFIER TRAINING")
    print("device:", DEVICE)
    print("dataset file:", DATASET_PATH)
    print("k values:", K_VALUES)

    x_tensor, y_tensor, positions_tensor = load_dataset(DATASET_PATH)

    print("\nDATASET")
    print("x_windows shape:", tuple(x_tensor.shape))
    print("y_labels shape:", tuple(y_tensor.shape))
    print("grid_positions shape:", tuple(positions_tensor.shape))
    print("num_classes:", int(y_tensor.max().item() + 1))

    for k in K_VALUES:
        train_single_k(
            k=k,
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            positions_tensor=positions_tensor,
        )

    print()
    print("ORDERED-K LLT EXPERIMENT COMPLETED")
    print("summary CSV saved to:", SUMMARY_CSV_PATH)


if __name__ == "__main__":
    main()