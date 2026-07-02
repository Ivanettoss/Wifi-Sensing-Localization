#Baseline MLP classifier training script with k samples ordered by acquisition 

import csv
import copy
import json
import random
import sys
import time
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


# Choose the dataset to train on:
# "meeting_room" or "lab"
DATASET_NAME = "lab"

DATASET_CONFIGS = {
    "meeting_room": {
        "dataset_file": "meeting_room_full_windows_30.npz",
        "split_prefix": "meeting_room_ordered",
        "dataset_label": "meeting_room_full_windows_30",
        "experiment_prefix": "ordered_k",
        "model_prefix": "meeting_room_ordered",
        "summary_csv": "fingerprint_classification_ordered_k_results.csv",
    },
    "lab": {
        "dataset_file": "lab_full_windows_30.npz",
        "split_prefix": "lab_ordered",
        "dataset_label": "lab_full_windows_30",
        "experiment_prefix": "lab_ordered_k",
        "model_prefix": "lab_ordered",
        "summary_csv": "fingerprint_classification_lab_ordered_k_results.csv",
    },
}

if DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(
        f"Invalid DATASET_NAME: {DATASET_NAME}. "
        f"Choose one of: {list(DATASET_CONFIGS.keys())}"
    )

CONFIG = DATASET_CONFIGS[DATASET_NAME]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / CONFIG["dataset_file"]
)

OUTPUT_MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_SPLITS_DIR = PROJECT_ROOT / "outputs" / "splits"

K_VALUES = [1, 5, 15, 25, 35]

SUMMARY_CSV_FILE = (
    OUTPUT_LOGS_DIR
    / CONFIG["summary_csv"]
)


def get_experiment_name(k: int) -> str:
    return f"{CONFIG['experiment_prefix']}{k}_seed42"


def get_split_file(k: int) -> Path:
    return OUTPUT_SPLITS_DIR / f"{CONFIG['split_prefix']}_k{k}_split.npz"


def get_best_model_file(k: int) -> Path:
    return OUTPUT_MODELS_DIR / f"mlp_{CONFIG['model_prefix']}_k{k}_best.pt"


def get_metrics_file(k: int) -> Path:
    return OUTPUT_LOGS_DIR / f"mlp_{CONFIG['model_prefix']}_k{k}_metrics.json"

RANDOM_SEED = 42

WINDOWS_PER_CLASS = 50
TRAIN_WINDOWS_PER_CLASS = 35
VAL_WINDOWS_PER_CLASS = 5
TEST_WINDOWS_PER_CLASS = 10

BATCH_SIZE = 64
MAX_EPOCHS = 40
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-6


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


def validate_labels(
    y_labels: np.ndarray,
    num_classes: int,
) -> None:
    """
    Check that labels are consecutive integers from 0 to num_classes - 1.
    """

    unique_labels = np.unique(y_labels)
    expected_labels = np.arange(num_classes)

    if not np.array_equal(unique_labels, expected_labels):
        raise ValueError(
            "Labels are not consecutive integers from 0 to num_classes - 1"
        )


def load_split_indices(
    split_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ordered-K train/validation/test split indices.

    The standard split keys are:
        train_indices / val_indices / test_indices
    """

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    split_data = np.load(split_file)

    required_keys = {
        "train_indices",
        "val_indices",
        "test_indices",
    }

    if not required_keys.issubset(set(split_data.files)):
        raise ValueError(
            f"Invalid ordered-K split file: {split_file}. "
            f"Available keys: {split_data.files}"
        )

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    test_indices = split_data["test_indices"]

    print(f"Loaded ordered-K split from: {split_file}")

    return train_indices, val_indices, test_indices


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
) -> tuple[float, float, float]:
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
    median_error = torch.quantile(errors, 0.5).item()
    rmse_error = torch.sqrt(torch.mean(errors ** 2)).item()

    return mean_error, median_error, rmse_error


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    class_grid_positions: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
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
            total_correct += (
                torch.argmax(logits, dim=1) == y_batch
            ).sum().item()
            total_samples += batch_size

            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mean_grid_error, median_grid_error, rmse_grid_error = compute_grid_error(
        logits=all_logits,
        labels=all_labels,
        class_grid_positions=class_grid_positions.cpu(),
    )

    return epoch_loss, epoch_accuracy, mean_grid_error, median_grid_error, rmse_grid_error


def build_class_grid_positions(
    y_labels: np.ndarray,
    grid_positions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Build the mapping from class id to grid position.
    """

    class_grid_positions = np.zeros((num_classes, 2), dtype=np.int64)

    for class_id in range(num_classes):
        class_positions = grid_positions[y_labels == class_id]

        if len(class_positions) == 0:
            raise ValueError(f"No grid position found for class {class_id}")

        class_grid_positions[class_id] = class_positions[0]

    return class_grid_positions



def update_summary_csv(
    summary_row: dict,
    summary_csv_file: Path,
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

    if summary_csv_file.exists():
        with open(
            summary_csv_file,
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
        summary_csv_file,
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


# MAIN

def train_single_k(
    k: int,
    x_windows: np.ndarray,
    y_labels: np.ndarray,
    grid_positions: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> None:
    """
    Train and evaluate the MLP classifier for one ordered-K split.
    """

    set_random_seed(RANDOM_SEED)

    split_file = get_split_file(k)
    best_model_file = get_best_model_file(k)
    metrics_file = get_metrics_file(k)

    print()
    print("=" * 80)
    print(f"MLP ORDERED-K CLASSIFIER TRAINING | K={k}")
    print("=" * 80)
    print(f"dataset name: {DATASET_NAME}")
    print(f"experiment: {get_experiment_name(k)}")
    print(f"device: {device}")
    print(f"dataset file: {DATASET_FILE}")
    print(f"split file: {split_file}")
    print()

    train_indices, val_indices, test_indices = load_split_indices(
        split_file=split_file,
    )

    train_mean = float(x_windows[train_indices].mean())
    train_std = float(x_windows[train_indices].std())

    print("SPLIT")
    print(f"k train windows per class: {k}")
    print(f"train samples: {len(train_indices)}")
    print(f"validation samples: {len(val_indices)}")
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

    val_dataset = CSIWindowDataset(
        x_windows=x_windows,
        y_labels=y_labels,
        grid_positions=grid_positions,
        indices=val_indices,
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
        num_classes=num_classes,
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
        num_classes=num_classes,
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

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state_dict = None

    best_val_accuracy = None
    best_val_mean_grid_error = None
    best_val_median_grid_error = None
    best_val_rmse_grid_error = None

    epochs_without_improvement = 0
    early_stopped = False

    history = []

    training_start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            val_loss,
            val_accuracy,
            val_mean_grid_error,
            val_median_grid_error,
            val_rmse_grid_error,
        ) = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            class_grid_positions=class_grid_positions,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_mean_grid_error": val_mean_grid_error,
            "val_median_grid_error": val_median_grid_error,
            "val_rmse_grid_error": val_rmse_grid_error,
        }

        history.append(epoch_metrics)

        print(
            f"K={k} | "
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_accuracy:.4f} | "
            f"val mean grid error: {val_mean_grid_error:.4f} | "
            f"val median grid error: {val_median_grid_error:.4f} | "
            f"val rmse grid error: {val_rmse_grid_error:.4f}"
        )

        improved = val_loss < (best_val_loss - MIN_DELTA)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())

            best_val_accuracy = val_accuracy
            best_val_mean_grid_error = val_mean_grid_error
            best_val_median_grid_error = val_median_grid_error
            best_val_rmse_grid_error = val_rmse_grid_error

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": best_model_state_dict,
                    "dataset_name": DATASET_NAME,
                    "experiment": get_experiment_name(k),
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "k": k,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "class_grid_positions": class_grid_positions_np,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_mean_grid_error": best_val_mean_grid_error,
                    "best_val_median_grid_error": best_val_median_grid_error,
                    "best_val_rmse_grid_error": best_val_rmse_grid_error,
                    "split_file": str(split_file),
                    "dataset_file": str(DATASET_FILE),
                },
                best_model_file,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            print()
            print(
                f"K={k} early stopping triggered: "
                f"validation loss did not improve for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    training_time_seconds = time.perf_counter() - training_start_time

    if best_model_state_dict is None:
        raise RuntimeError(f"No best model was saved during training for K={k}.")

    model.load_state_dict(best_model_state_dict)

    test_start_time = time.perf_counter()

    (
        test_loss,
        test_accuracy,
        test_mean_grid_error,
        test_median_grid_error,
        test_rmse_grid_error,
    ) = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        class_grid_positions=class_grid_positions,
        device=device,
    )

    test_time_seconds = time.perf_counter() - test_start_time

    metrics_output = {
        "model_name": "MLP",
        "dataset_name": DATASET_NAME,
        "experiment": get_experiment_name(k),
        "k": k,
        "dataset_file": str(DATASET_FILE),
        "best_model_file": str(best_model_file),
        "split_file": str(split_file),
        "num_classes": num_classes,
        "max_epochs": MAX_EPOCHS,
        "epochs_ran": len(history),
        "early_stopped": early_stopped,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_delta": MIN_DELTA,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE,
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "test_samples": int(len(test_indices)),
        "train_windows_per_class": k,
        "val_windows_per_class": VAL_WINDOWS_PER_CLASS,
        "test_windows_per_class": TEST_WINDOWS_PER_CLASS,
        "train_mean": train_mean,
        "train_std": train_std,
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_val_mean_grid_error": best_val_mean_grid_error,
        "best_val_median_grid_error": best_val_median_grid_error,
        "best_val_rmse_grid_error": best_val_rmse_grid_error,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_mean_grid_error": test_mean_grid_error,
        "test_median_grid_error": test_median_grid_error,
        "test_rmse_grid_error": test_rmse_grid_error,
        "training_time_seconds": training_time_seconds,
        "test_time_seconds": test_time_seconds,
        "history": history,
    }

    with open(metrics_file, "w", encoding="utf-8") as output_file:
        json.dump(metrics_output, output_file, indent=4)

    summary_row = {
        "model": "MLP",
        "k": k,
        "dataset": CONFIG["dataset_label"],
        "experiment": get_experiment_name(k),
        "split_file": str(split_file),
        "best_model_file": str(best_model_file),
        "num_classes": num_classes,
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "test_samples": int(len(test_indices)),
        "accuracy": test_accuracy,
        "mean_grid_error": test_mean_grid_error,
        "median_grid_error": test_median_grid_error,
        "rmse_grid_error": test_rmse_grid_error,
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "early_stopped": early_stopped,
        "training_time_seconds": training_time_seconds,
        "test_time_seconds": test_time_seconds,
    }

    update_summary_csv(
        summary_row=summary_row,
        summary_csv_file=SUMMARY_CSV_FILE,
    )

    print()
    print(f"K={k} TRAINING COMPLETED")
    print(f"epochs ran: {len(history)}")
    print(f"early stopped: {early_stopped}")
    print(f"best epoch: {best_epoch}")
    print(f"best validation loss: {best_val_loss:.4f}")
    print(f"best validation accuracy: {best_val_accuracy:.4f}")
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.4f}")
    print(f"test mean grid error: {test_mean_grid_error:.4f}")
    print(f"test median grid error: {test_median_grid_error:.4f}")
    print(f"test rmse grid error: {test_rmse_grid_error:.4f}")
    print(f"training time seconds: {training_time_seconds:.2f}")
    print(f"test time seconds: {test_time_seconds:.2f}")
    print(f"best model saved to: {best_model_file}")
    print(f"metrics saved to: {metrics_file}")
    print(f"summary CSV saved to: {SUMMARY_CSV_FILE}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("MLP ORDERED-K CLASSIFIER TRAINING")
    print(f"dataset name: {DATASET_NAME}")
    print(f"summary csv: {SUMMARY_CSV_FILE}")
    print(f"device: {device}")
    print(f"dataset file: {DATASET_FILE}")
    print(f"k values: {K_VALUES}")

    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

    dataset = np.load(DATASET_FILE)

    x_windows = dataset["x_windows"]
    y_labels = dataset["y_labels"]
    grid_positions = dataset["grid_positions"]

    num_classes = int(len(np.unique(y_labels)))

    validate_labels(
        y_labels=y_labels,
        num_classes=num_classes,
    )

    print("DATASET")
    print(f"x_windows shape: {x_windows.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"grid_positions shape: {grid_positions.shape}")
    print(f"num_classes: {num_classes}")
    print()

    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    for k in K_VALUES:
        train_single_k(
            k=k,
            x_windows=x_windows,
            y_labels=y_labels,
            grid_positions=grid_positions,
            num_classes=num_classes,
            device=device,
        )

    print()
    print("ORDERED-K MLP EXPERIMENT COMPLETED")
    print(f"summary CSV saved to: {SUMMARY_CSV_FILE}")


if __name__ == "__main__":
    main()