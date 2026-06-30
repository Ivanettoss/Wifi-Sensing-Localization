#Second experiment:  k values in training, ordered by acquisition 
# K values used : 1,5,15,25,35(full)
# The file will produce a csv summary and the following splits:
''' 
outputs\splits\meeting_room_ordered_k1_split.npz
outputs\splits\meeting_room_ordered_k5_split.npz
outputs\splits\meeting_room_ordered_k15_split.npz
outputs\splits\meeting_room_ordered_k25_split.npz
outputs\splits\meeting_room_ordered_k35_split.npz
'''

from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np



# Build paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "meeting_room_full_windows_30.npz"

SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SUMMARY_CSV_PATH = REPORT_DIR / "ordered_k_splits_summary.csv"



# K split settings : fixed validation and test windows, variable training windows

K_VALUES = [1, 5, 15, 25, 35]

VAL_WINDOW_START = 35
VAL_WINDOW_END = 39

TEST_WINDOW_START = 40
TEST_WINDOW_END = 49

EXPECTED_WINDOWS_PER_CLASS = 50
EXPECTED_NUM_CLASSES = 176


# Utilities

def load_dataset(dataset_path: Path) -> dict[str, np.ndarray]:

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path)

    required_keys = [
        "x_windows",
        "y_labels",
        "grid_positions",
        "window_indices",
        "window_start_packets",
        "window_end_packets",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key missing from dataset: {key}")

    return {
        "x_windows": data["x_windows"],
        "y_labels": data["y_labels"],
        "grid_positions": data["grid_positions"],
        "window_indices": data["window_indices"],
        "window_start_packets": data["window_start_packets"],
        "window_end_packets": data["window_end_packets"],
    }


def check_dataset_structure(
    y_labels: np.ndarray,
    window_indices: np.ndarray,
) -> None:
    
    #Check that each class has the expected number of ordered windows
    unique_labels = np.unique(y_labels)

    if len(unique_labels) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} classes, found {len(unique_labels)}"
        )

    for label in unique_labels:
        label_indices = np.where(y_labels == label)[0]
        label_window_indices = window_indices[label_indices]

        if len(label_indices) != EXPECTED_WINDOWS_PER_CLASS:
            raise ValueError(
                f"Label {label} has {len(label_indices)} samples, "
                f"expected {EXPECTED_WINDOWS_PER_CLASS}"
            )

        expected_window_indices = np.arange(EXPECTED_WINDOWS_PER_CLASS)

        if not np.array_equal(np.sort(label_window_indices), expected_window_indices):
            raise ValueError(
                f"Label {label} does not contain window indices 0..49 exactly once"
            )


def get_indices_for_window_range(
    y_labels: np.ndarray,
    window_indices: np.ndarray,
    label: int,
    start_window: int,
    end_window: int,
) -> np.ndarray:
    
    mask = (
        (y_labels == label)
        & (window_indices >= start_window)
        & (window_indices <= end_window)
    )

    selected_indices = np.where(mask)[0]

    order = np.argsort(window_indices[selected_indices])
    selected_indices = selected_indices[order]

    return selected_indices


''' build an ordered window, For each reference point:
        train      = windows 0 ... K-1
        validation = windows 35 ... 39
        test       = windows 40 ... 49 '''

def build_ordered_k_split(
    y_labels: np.ndarray,
    window_indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if k < 1:
        raise ValueError("k must be >= 1")

    if k > VAL_WINDOW_START:
        raise ValueError(
            f"k={k} is too large. Respect K <= {VAL_WINDOW_START} "
            f"to avoid overlap with validation windows."
        )

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    unique_labels = np.unique(y_labels)

    for label in unique_labels:
        label = int(label)

        train_for_label = get_indices_for_window_range(
            y_labels=y_labels,
            window_indices=window_indices,
            label=label,
            start_window=0,
            end_window=k - 1,
        )

        val_for_label = get_indices_for_window_range(
            y_labels=y_labels,
            window_indices=window_indices,
            label=label,
            start_window=VAL_WINDOW_START,
            end_window=VAL_WINDOW_END,
        )

        test_for_label = get_indices_for_window_range(
            y_labels=y_labels,
            window_indices=window_indices,
            label=label,
            start_window=TEST_WINDOW_START,
            end_window=TEST_WINDOW_END,
        )

        if len(train_for_label) != k:
            raise ValueError(
                f"Label {label} has {len(train_for_label)} train samples, expected {k}"
            )

        if len(val_for_label) != 5:
            raise ValueError(
                f"Label {label} has {len(val_for_label)} validation samples, expected 5"
            )

        if len(test_for_label) != 10:
            raise ValueError(
                f"Label {label} has {len(test_for_label)} test samples, expected 10"
            )

        train_indices.extend(train_for_label.tolist())
        val_indices.extend(val_for_label.tolist())
        test_indices.extend(test_for_label.tolist())

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
    )

# check if the split is valid in terms of overlapping and samples 
def validate_split(
    y_labels: np.ndarray,
    window_indices: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    k: int,
) -> None:

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    if train_set.intersection(val_set):
        raise ValueError(f"K={k}: train/validation overlap found")

    if train_set.intersection(test_set):
        raise ValueError(f"K={k}: train/test overlap found")

    if val_set.intersection(test_set):
        raise ValueError(f"K={k}: validation/test overlap found")

    unique_labels = np.unique(y_labels)

    for label in unique_labels:
        label = int(label)

        train_for_label = train_idx[y_labels[train_idx] == label]
        val_for_label = val_idx[y_labels[val_idx] == label]
        test_for_label = test_idx[y_labels[test_idx] == label]

        if len(train_for_label) != k:
            raise ValueError(
                f"K={k}, label={label}: train count is {len(train_for_label)}, expected {k}"
            )

        if len(val_for_label) != 5:
            raise ValueError(
                f"K={k}, label={label}: validation count is {len(val_for_label)}, expected 5"
            )

        if len(test_for_label) != 10:
            raise ValueError(
                f"K={k}, label={label}: test count is {len(test_for_label)}, expected 10"
            )

        train_windows = np.sort(window_indices[train_for_label])
        val_windows = np.sort(window_indices[val_for_label])
        test_windows = np.sort(window_indices[test_for_label])

        expected_train_windows = np.arange(k)
        expected_val_windows = np.arange(VAL_WINDOW_START, VAL_WINDOW_END + 1)
        expected_test_windows = np.arange(TEST_WINDOW_START, TEST_WINDOW_END + 1)

        if not np.array_equal(train_windows, expected_train_windows):
            raise ValueError(
                f"K={k}, label={label}: unexpected train windows {train_windows}"
            )

        if not np.array_equal(val_windows, expected_val_windows):
            raise ValueError(
                f"K={k}, label={label}: unexpected validation windows {val_windows}"
            )

        if not np.array_equal(test_windows, expected_test_windows):
            raise ValueError(
                f"K={k}, label={label}: unexpected test windows {test_windows}"
            )



 # Save split indices and metadata to a .npz file
def save_split(
    output_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    k: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        shots_per_class=np.array(k, dtype=np.int64),
        split_type=np.array("ordered_k_acquisition_split"),
        train_window_start=np.array(0, dtype=np.int64),
        train_window_end=np.array(k - 1, dtype=np.int64),
        val_window_start=np.array(VAL_WINDOW_START, dtype=np.int64),
        val_window_end=np.array(VAL_WINDOW_END, dtype=np.int64),
        test_window_start=np.array(TEST_WINDOW_START, dtype=np.int64),
        test_window_end=np.array(TEST_WINDOW_END, dtype=np.int64),
    )

# Save a summary CSV file with split info
def save_summary_csv(rows: list[dict[str, int | str]]) -> None:

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split_file",
        "k",
        "train_samples",
        "val_samples",
        "test_samples",
        "train_per_class",
        "val_per_class",
        "test_per_class",
        "train_window_range",
        "val_window_range",
        "test_window_range",
    ]

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)



# MAIN

def main() -> None:
    print("BUILD ORDERED K SPLITS")
    print("project root:", PROJECT_ROOT)
    print("dataset:", DATASET_PATH)

    dataset = load_dataset(DATASET_PATH)

    y_labels = dataset["y_labels"]
    window_indices = dataset["window_indices"]
    window_start_packets = dataset["window_start_packets"]
    window_end_packets = dataset["window_end_packets"]

    print()
    print("DATASET")
    print("num samples:", len(y_labels))
    print("num classes:", len(np.unique(y_labels)))
    print("samples per class:", len(y_labels) // len(np.unique(y_labels)))
    print("window_indices min:", int(window_indices.min()))
    print("window_indices max:", int(window_indices.max()))
    print("window_start_packets min:", int(window_start_packets.min()))
    print("window_start_packets max:", int(window_start_packets.max()))
    print("window_end_packets min:", int(window_end_packets.min()))
    print("window_end_packets max:", int(window_end_packets.max()))

    check_dataset_structure(
        y_labels=y_labels,
        window_indices=window_indices,
    )

    summary_rows: list[dict[str, int | str]] = []

    print()
    print("GENERATING SPLITS")

    for k in K_VALUES:
        train_idx, val_idx, test_idx = build_ordered_k_split(
            y_labels=y_labels,
            window_indices=window_indices,
            k=k,
        )

        validate_split(
            y_labels=y_labels,
            window_indices=window_indices,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            k=k,
        )

        output_path = SPLIT_DIR / f"meeting_room_ordered_k{k}_split.npz"

        save_split(
            output_path=output_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            k=k,
        )

        print()
        print(f"K={k}")
        print("saved to:", output_path)
        print("train samples:", len(train_idx))
        print("validation samples:", len(val_idx))
        print("test samples:", len(test_idx))
        print(f"train windows: 0-{k - 1}")
        print(f"validation windows: {VAL_WINDOW_START}-{VAL_WINDOW_END}")
        print(f"test windows: {TEST_WINDOW_START}-{TEST_WINDOW_END}")

        summary_rows.append(
            {
                "split_file": str(output_path),
                "k": k,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
                "train_per_class": k,
                "val_per_class": 5,
                "test_per_class": 10,
                "train_window_range": f"0-{k - 1}",
                "val_window_range": f"{VAL_WINDOW_START}-{VAL_WINDOW_END}",
                "test_window_range": f"{TEST_WINDOW_START}-{TEST_WINDOW_END}",
            }
        )

    save_summary_csv(summary_rows)

    print()
    print("SUMMARY CSV")
    print("saved to:", SUMMARY_CSV_PATH)

    print()
    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("ERROR:", error)
        sys.exit(1)