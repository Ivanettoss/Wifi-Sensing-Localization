#Testing script to check the integrity of the previously built meeting_room_1_100_raw.npz dataset.

from pathlib import Path
from collections import Counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "meeting_room_1_100_raw.npz"
)

EXPECTED_X_SHAPE = (100, 3, 30, 1500)
EXPECTED_Y_SHAPE = (100,)


def check_condition(condition: bool, error_message: str) -> None:
    """
    Raise a clear error message if a validation check fails.
    """

    if not condition:
        raise ValueError(error_message)


def main() -> None:
    print("INPUT FILE")
    print(INPUT_FILE)
    print()

    check_condition(
        INPUT_FILE.exists(),
        f"Processed dataset file not found: {INPUT_FILE}",
    )

    dataset = np.load(INPUT_FILE)

    available_keys = list(dataset.keys())

    print("AVAILABLE KEYS")
    print(available_keys)
    print()

    required_keys = ["x_data", "y_labels", "file_names", "selected_keys"]

    for key in required_keys:
        check_condition(
            key in available_keys,
            f"Missing required key: {key}",
        )

    x_data = dataset["x_data"]
    y_labels = dataset["y_labels"]
    file_names = dataset["file_names"]
    selected_keys = dataset["selected_keys"]

    print("BASIC SHAPES")
    print(f"x_data shape: {x_data.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"file_names shape: {file_names.shape}")
    print(f"selected_keys shape: {selected_keys.shape}")
    print()

    check_condition(
        x_data.shape == EXPECTED_X_SHAPE,
        f"Invalid x_data shape: {x_data.shape}, expected {EXPECTED_X_SHAPE}",
    )

    check_condition(
        y_labels.shape == EXPECTED_Y_SHAPE,
        f"Invalid y_labels shape: {y_labels.shape}, expected {EXPECTED_Y_SHAPE}",
    )

    check_condition(
        len(file_names) == EXPECTED_X_SHAPE[0],
        f"Invalid number of file names: {len(file_names)}",
    )

    check_condition(
        len(selected_keys) == EXPECTED_X_SHAPE[0],
        f"Invalid number of selected keys: {len(selected_keys)}",
    )

    print("DATA TYPES")
    print(f"x_data dtype: {x_data.dtype}")
    print(f"y_labels dtype: {y_labels.dtype}")
    print(f"file_names dtype: {file_names.dtype}")
    print(f"selected_keys dtype: {selected_keys.dtype}")
    print()

    check_condition(
        x_data.dtype == np.float32,
        f"Invalid x_data dtype: {x_data.dtype}, expected float32",
    )

    check_condition(
        np.issubdtype(y_labels.dtype, np.integer),
        f"Invalid y_labels dtype: {y_labels.dtype}, expected integer dtype",
    )

    print("NUMERIC CHECKS")
    print(f"finite values: {np.isfinite(x_data).all()}")
    print(f"NaN count: {np.isnan(x_data).sum()}")
    print(f"+inf count: {np.isposinf(x_data).sum()}")
    print(f"-inf count: {np.isneginf(x_data).sum()}")
    print()

    check_condition(
        np.isfinite(x_data).all(),
        "x_data contains NaN or infinite values",
    )

    print("VALUE STATISTICS")
    print(f"min: {x_data.min():.6f}")
    print(f"max: {x_data.max():.6f}")
    print(f"mean: {x_data.mean():.6f}")
    print(f"std: {x_data.std():.6f}")
    print()

    print("LABEL CHECKS")
    unique_labels = np.unique(y_labels)

    print(f"number of unique labels: {len(unique_labels)}")
    print(f"first labels: {unique_labels[:10]}")
    print(f"last labels: {unique_labels[-10:]}")
    print()

    expected_labels = np.arange(EXPECTED_X_SHAPE[0])

    check_condition(
        np.array_equal(unique_labels, expected_labels),
        "Labels are not consecutive from 0 to 99",
    )

    label_counts = Counter(y_labels.tolist())

    print("LABEL COUNTS CHECK")
    print(f"minimum samples per label: {min(label_counts.values())}")
    print(f"maximum samples per label: {max(label_counts.values())}")
    print()

    check_condition(
        min(label_counts.values()) == 1 and max(label_counts.values()) == 1,
        "Each label should appear exactly once in the raw dataset",
    )

    print("FILE NAME CHECK")
    print(f"first file: {file_names[0]}")
    print(f"last file: {file_names[-1]}")
    print()

    print("SELECTED MATLAB KEYS")
    key_counts = Counter(selected_keys.tolist())

    for key, count in key_counts.items():
        print(f"{key}: {count}")

    print()

    print("SAMPLE VARIANCE CHECK")
    flattened_data = x_data.reshape(x_data.shape[0], -1)
    sample_std_values = flattened_data.std(axis=1)

    zero_variance_indices = np.where(sample_std_values == 0)[0]

    print(f"zero-variance samples: {len(zero_variance_indices)}")

    check_condition(
        len(zero_variance_indices) == 0,
        f"Found zero-variance samples at indices: {zero_variance_indices}",
    )

    print()
    print("DATASET VERIFICATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()