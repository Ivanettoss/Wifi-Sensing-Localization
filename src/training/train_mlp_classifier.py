import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.mlp import MLPClassifier, count_trainable_parameters


DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "meeting_room_1_100_windows_30.npz"
)

OUTPUT_MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

BEST_MODEL_FILE = OUTPUT_MODELS_DIR / "mlp_classifier_best.pt"
METRICS_FILE = OUTPUT_LOGS_DIR / "mlp_classifier_metrics.json"

RANDOM_SEED = 42

NUM_CLASSES = 100
WINDOWS_PER_CLASS = 50
TRAIN_WINDOWS_PER_CLASS = 40
TEST_WINDOWS_PER_CLASS = 10

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3


class CSIWindowDataset(Dataset):
    """
    PyTorch dataset for CSI temporal windows.
    """

    def __init__(
        self,
        x_windows: np.ndarray,
        y_labels: np.ndarray,
        grid_positions: np.ndarray,
        indices: np.ndarray,
        mean_value: float,
        std_value: float,
    ) -> None:
        self.x_windows = x_windows
        self.y_labels = y_labels
        self.grid_positions = grid_positions
        self.indices = indices
        self.mean_value = mean_value
        self.std_value = std_value

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        item_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset_index = self.indices[item_index]

        x_window = self.x_windows[dataset_index]
        y_label = self.y_labels[dataset_index]
        grid_position = self.grid_positions[dataset_index]

        x_window = (x_window - self.mean_value) / (self.std_value + 1e-8)

        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        grid_tensor = torch.tensor(grid_position, dtype=torch.long)

        return x_tensor, y_tensor, grid_tensor


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_split_indices(y_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a balanced train/test split.

    For each class:
        40 windows are used for training
        10 windows are used for testing
    """

    train_indices = []
    test_indices = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(y_labels == class_id)[0]

        if len(class_indices) != WINDOWS_PER_CLASS:
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} windows, "
                f"expected {WINDOWS_PER_CLASS}"
            )

        np.random.shuffle(class_indices)

        train_indices.extend(class_indices[:TRAIN_WINDOWS_PER_CLASS])
        test_indices.extend(class_indices[TRAIN_WINDOWS_PER_CLASS:])

    train_indices = np.array(train_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return train_indices, test_indices


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch, _ in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == y_batch).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


def compute_grid_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_grid_positions: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute mean and RMSE grid localization error.

    The predicted class is mapped to its grid position.
    """

    predictions = torch.argmax(logits, dim=1)

    predicted_positions = class_grid_positions[predictions]
    true_positions = class_grid_positions[labels]

    errors = torch.sqrt(
        torch.sum(
            (predicted_positions.float() - true_positions.float()) ** 2,
            dim=1,
        )
    )

    mean_error = errors.mean().item()
    rmse_error = torch.sqrt(torch.mean(errors ** 2)).item()

    return mean_error, rmse_error


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    class_grid_positions: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    Evaluate the model.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            batch_size = y_batch.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (torch.argmax(logits, dim=1) == y_batch).sum().item()
            total_samples += batch_size

            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mean_grid_error, rmse_grid_error = compute_grid_error(
        logits=all_logits,
        labels=all_labels,
        class_grid_positions=class_grid_positions.cpu(),
    )

    return epoch_loss, epoch_accuracy, mean_grid_error, rmse_grid_error


def build_class_grid_positions(
    y_labels: np.ndarray,
    grid_positions: np.ndarray,
) -> np.ndarray:
    """
    Build the mapping from class id to grid position.
    """

    class_grid_positions = np.zeros((NUM_CLASSES, 2), dtype=np.int64)

    for class_id in range(NUM_CLASSES):
        class_positions = grid_positions[y_labels == class_id]

        if len(class_positions) == 0:
            raise ValueError(f"No grid position found for class {class_id}")

        class_grid_positions[class_id] = class_positions[0]

    return class_grid_positions


def main() -> None:
    set_random_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("MLP CLASSIFIER TRAINING")
    print(f"device: {device}")
    print(f"dataset file: {DATASET_FILE}")
    print()

    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

    dataset = np.load(DATASET_FILE)

    x_windows = dataset["x_windows"]
    y_labels = dataset["y_labels"]
    grid_positions = dataset["grid_positions"]

    print("DATASET")
    print(f"x_windows shape: {x_windows.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"grid_positions shape: {grid_positions.shape}")
    print()

    train_indices, test_indices = build_split_indices(y_labels)

    train_mean = float(x_windows[train_indices].mean())
    train_std = float(x_windows[train_indices].std())

    print("SPLIT")
    print(f"train samples: {len(train_indices)}")
    print(f"test samples: {len(test_indices)}")
    print(f"train mean: {train_mean:.6f}")
    print(f"train std: {train_std:.6f}")
    print()

    train_dataset = CSIWindowDataset(
        x_windows=x_windows,
        y_labels=y_labels,
        grid_positions=grid_positions,
        indices=train_indices,
        mean_value=train_mean,
        std_value=train_std,
    )

    test_dataset = CSIWindowDataset(
        x_windows=x_windows,
        y_labels=y_labels,
        grid_positions=grid_positions,
        indices=test_indices,
        mean_value=train_mean,
        std_value=train_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = x_windows.shape[1] * x_windows.shape[2] * x_windows.shape[3]

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim_1=512,
        hidden_dim_2=256,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    class_grid_positions_np = build_class_grid_positions(
        y_labels=y_labels,
        grid_positions=grid_positions,
    )

    class_grid_positions = torch.tensor(
        class_grid_positions_np,
        dtype=torch.long,
        device=device,
    )

    print("MODEL")
    print(model)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    print()

    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    best_test_accuracy = -1.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy, mean_grid_error, rmse_grid_error = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            class_grid_positions=class_grid_positions,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "mean_grid_error": mean_grid_error,
            "rmse_grid_error": rmse_grid_error,
        }

        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"test loss: {test_loss:.4f} | "
            f"test acc: {test_accuracy:.4f} | "
            f"mean grid error: {mean_grid_error:.4f} | "
            f"rmse grid error: {rmse_grid_error:.4f}"
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": NUM_CLASSES,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "class_grid_positions": class_grid_positions_np,
                    "best_test_accuracy": best_test_accuracy,
                },
                BEST_MODEL_FILE,
            )

    metrics_output = {
        "dataset_file": str(DATASET_FILE),
        "best_model_file": str(BEST_MODEL_FILE),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "train_mean": train_mean,
        "train_std": train_std,
        "best_test_accuracy": best_test_accuracy,
        "history": history,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as output_file:
        json.dump(metrics_output, output_file, indent=4)

    print()
    print("TRAINING COMPLETED")
    print(f"best test accuracy: {best_test_accuracy:.4f}")
    print(f"best model saved to: {BEST_MODEL_FILE}")
    print(f"metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()