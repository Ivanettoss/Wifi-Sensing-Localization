from __future__ import annotations

from pathlib import Path
import sys
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Make src importable when running this script directly
sys.path.append(str(SRC_DIR))

from models.llt import LLT, count_trainable_parameters


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "meeting_room_full_windows_30.npz"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "splits" / "meeting_room_full_train_val_test_split_seed42.npz"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "llt_classifier_best_2h.pt"

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


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATA LOADING
# =============================================================================

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

def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """Compute top-k accuracy for a batch."""

    topk_predictions = logits.topk(k, dim=1).indices
    correct = topk_predictions.eq(labels.view(-1, 1)).any(dim=1)

    return correct.float().mean().item()


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
    total_top5 = 0.0
    total_spatial_error = 0.0
    total_samples = 0

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
        total_top5 += compute_topk_accuracy(logits, y_batch, k=5) * batch_size
        total_spatial_error += spatial_errors.sum().item()
        total_samples += batch_size

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "top5_accuracy": total_top5 / total_samples,
        "mean_spatial_error": total_spatial_error / total_samples,
    }

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("LLT CLASSIFIER TRAINING")
    print("device:", DEVICE)
    print("dataset file:", DATASET_PATH)
    print("split file:", SPLIT_PATH)

    x_tensor, y_tensor, positions_tensor = load_dataset(DATASET_PATH)

    print("\nDATASET")
    print("x_windows shape:", tuple(x_tensor.shape))
    print("y_labels shape:", tuple(y_tensor.shape))
    print("grid_positions shape:", tuple(positions_tensor.shape))
    print("num_classes:", int(y_tensor.max().item() + 1))

    train_idx, val_idx, test_idx = load_split_indices(SPLIT_PATH)

    print("\nSPLIT")
    print("Loaded existing split from:", SPLIT_PATH)
    print("train samples:", len(train_idx))
    print("val samples:", len(val_idx))
    print("test samples:", len(test_idx))

    train_loader, val_loader, test_loader = build_loaders(
        x_tensor=x_tensor,
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
    bad_epochs = 0

    print("\nTRAINING")
    print("epochs:", EPOCHS)
    print("batch_size:", BATCH_SIZE)
    print("learning_rate:", LEARNING_RATE)
    print("weight_decay:", WEIGHT_DECAY)
    print("patience:", PATIENCE)

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

        improved = val_metrics["loss"] < best_val_loss

        if improved:
            best_val_loss = val_metrics["loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
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
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                CHECKPOINT_PATH,
            )

            best_marker = "*"
        else:
            bad_epochs += 1
            best_marker = ""

        print(
            f"epoch {epoch:03d}/{EPOCHS} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_top5={val_metrics['top5_accuracy']:.4f} "
            f"val_spatial_err={val_metrics['mean_spatial_error']:.4f} "
            f"bad={bad_epochs}/{PATIENCE} {best_marker}"
        )

        if bad_epochs >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state_dict is None:
        raise RuntimeError("No best model state was saved during training.")

    model.load_state_dict(best_state_dict)

    print("\nBEST CHECKPOINT")
    print("saved to:", CHECKPOINT_PATH)
    print("best val loss:", best_val_loss)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        class_positions=class_positions,
    )

    print("\nTEST")
    print("test_loss:", f"{test_metrics['loss']:.4f}")
    print("test_accuracy:", f"{test_metrics['accuracy']:.4f}")
    print("test_top5_accuracy:", f"{test_metrics['top5_accuracy']:.4f}")
    print("test_mean_spatial_error:", f"{test_metrics['mean_spatial_error']:.4f}")


if __name__ == "__main__":
    main()