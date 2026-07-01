from __future__ import annotations

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lab_full_windows_30.npz"
)

OUTPUT_SPLITS_DIR = PROJECT_ROOT / "outputs" / "splits"

FULL_RANDOM_SPLIT_FILE = (
    OUTPUT_SPLITS_DIR
    / "lab_full_train_val_test_split_seed42.npz"
)

RANDOM_SEED = 42

WINDOWS_PER_CLASS = 50
TRAIN_WINDOWS_PER_CLASS = 35
VAL_WINDOWS_PER_CLASS = 5
TEST_WINDOWS_PER_CLASS = 10

ORDERED_K_VALUES = [1, 5, 15, 25, 35]

ORDERED_VAL_START = 35
ORDERED_VAL_END = 40
ORDERED_TEST_START = 40
ORDERED_TEST_END = 50


def validate_dataset(
    y_labels: np.ndarray,
    num_classes: int,
) -> None:
    """
    Validate labels and number of samples per class.
    """

    unique_labels = np.unique(y_labels)
    expected_labels = np.arange(num_classes)

    if not np.array_equal(unique_labels, expected_labels):
        raise ValueError(
            "Labels are not consecutive integers from 0 to num_classes - 1."
        )

    for class_id in range(num_classes):
        class_indices = np.where(y_labels == class_id)[0]

        if len(class_indices) != WINDOWS_PER_CLASS:
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} samples, "
                f"expected {WINDOWS_PER_CLASS}."
            )


def build_full_random_split(
    y_labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a random train/validation/test split for each class.

    For each reference point:
        train: 35 random windows
        validation: 5 random windows
        test: 10 random windows
    """

    rng = np.random.default_rng(RANDOM_SEED)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in range(num_classes):
        class_indices = np.where(y_labels == class_id)[0]
        class_indices = np.array(class_indices, dtype=np.int64)

        rng.shuffle(class_indices)

        train_class_indices = class_indices[:TRAIN_WINDOWS_PER_CLASS]
        val_class_indices = class_indices[
            TRAIN_WINDOWS_PER_CLASS:
            TRAIN_WINDOWS_PER_CLASS + VAL_WINDOWS_PER_CLASS
        ]
        test_class_indices = class_indices[
            TRAIN_WINDOWS_PER_CLASS + VAL_WINDOWS_PER_CLASS:
            TRAIN_WINDOWS_PER_CLASS + VAL_WINDOWS_PER_CLASS + TEST_WINDOWS_PER_CLASS
        ]

        train_indices.append(train_class_indices)
        val_indices.append(val_class_indices)
        test_indices.append(test_class_indices)

    train_indices = np.concatenate(train_indices).astype(np.int64)
    val_indices = np.concatenate(val_indices).astype(np.int64)
    test_indices = np.concatenate(test_indices).astype(np.int64)

    return train_indices, val_indices, test_indices


def build_ordered_k_split(
    y_labels: np.ndarray,
    num_classes: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an ordered-K split for each class.

    For each reference point:
        train: first K windows
        validation: windows 35-39
        test: windows 40-49
    """

    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in range(num_classes):
        class_indices = np.where(y_labels == class_id)[0]
        class_indices = np.sort(class_indices).astype(np.int64)

        train_class_indices = class_indices[:k]
        val_class_indices = class_indices[ORDERED_VAL_START:ORDERED_VAL_END]
        test_class_indices = class_indices[ORDERED_TEST_START:ORDERED_TEST_END]

        train_indices.append(train_class_indices)
        val_indices.append(val_class_indices)
        test_indices.append(test_class_indices)

    train_indices = np.concatenate(train_indices).astype(np.int64)
    val_indices = np.concatenate(val_indices).astype(np.int64)
    test_indices = np.concatenate(test_indices).astype(np.int64)

    return train_indices, val_indices, test_indices


def save_split(
    output_file: Path,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    split_type: str,
    num_classes: int,
    shots_per_class: int | None = None,
) -> None:
    """
    Save split indices and metadata.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        train_idx=train_indices,
        val_idx=val_indices,
        test_idx=test_indices,
        split_type=split_type,
        num_classes=num_classes,
        windows_per_class=WINDOWS_PER_CLASS,
        train_windows_per_class=(
            shots_per_class
            if shots_per_class is not None
            else TRAIN_WINDOWS_PER_CLASS
        ),
        val_windows_per_class=VAL_WINDOWS_PER_CLASS,
        test_windows_per_class=TEST_WINDOWS_PER_CLASS,
        shots_per_class=(
            -1
            if shots_per_class is None
            else shots_per_class
        ),
        random_seed=RANDOM_SEED,
        ordered_val_start=ORDERED_VAL_START,
        ordered_val_end=ORDERED_VAL_END,
        ordered_test_start=ORDERED_TEST_START,
        ordered_test_end=ORDERED_TEST_END,
        dataset_file=str(DATASET_FILE),
    )


def main() -> None:
    print("BUILD LAB SPLITS")
    print(f"dataset file: {DATASET_FILE}")
    print(f"output directory: {OUTPUT_SPLITS_DIR}")
    print()

    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

    dataset = np.load(DATASET_FILE)

    y_labels = dataset["y_labels"]
    num_classes = int(len(np.unique(y_labels)))

    print("DATASET")
    print(f"num samples: {len(y_labels)}")
    print(f"num classes: {num_classes}")
    print(f"windows per class: {WINDOWS_PER_CLASS}")
    print()

    validate_dataset(
        y_labels=y_labels,
        num_classes=num_classes,
    )

    print("BUILDING FULL RANDOM SPLIT")

    train_idx, val_idx, test_idx = build_full_random_split(
        y_labels=y_labels,
        num_classes=num_classes,
    )

    save_split(
        output_file=FULL_RANDOM_SPLIT_FILE,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_type="full_random_seed42",
        num_classes=num_classes,
        shots_per_class=None,
    )

    print(f"saved: {FULL_RANDOM_SPLIT_FILE}")
    print(f"train samples: {len(train_idx)}")
    print(f"val samples: {len(val_idx)}")
    print(f"test samples: {len(test_idx)}")
    print()

    print("BUILDING ORDERED-K SPLITS")

    for k in ORDERED_K_VALUES:
        train_idx, val_idx, test_idx = build_ordered_k_split(
            y_labels=y_labels,
            num_classes=num_classes,
            k=k,
        )

        output_file = (
            OUTPUT_SPLITS_DIR
            / f"lab_ordered_k{k}_split.npz"
        )

        save_split(
            output_file=output_file,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            split_type=f"ordered_k{k}_seed42",
            num_classes=num_classes,
            shots_per_class=k,
        )

        print(f"K={k}")
        print(f"saved: {output_file}")
        print(f"train samples: {len(train_idx)}")
        print(f"val samples: {len(val_idx)}")
        print(f"test samples: {len(test_idx)}")
        print()

    print("LAB SPLITS COMPLETED")


if __name__ == "__main__":
    main()