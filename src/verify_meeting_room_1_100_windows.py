from pathlib import Path
from collections import Counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "meeting_room_1_100_windows_30.npz"
)

EXPECTED_X_SHAPE = (5000, 3, 30, 30)
EXPECTED_Y_SHAPE = (5000,)
EXPECTED_GRID_SHAPE = (5000, 2)

EXPECTED_NUM_CLASSES = 100
EXPECTED_WINDOWS_PER_CLASS = 50


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
        f"Windowed dataset file not found: {INPUT_FILE}",
    )

    dataset = np.load(INPUT_FILE)

    available_keys = list(dataset.keys())

    print("AVAILABLE KEYS")
    print(available_keys)
    print()

    required_keys = [
        "x_windows",
        "y_labels",
        "grid_positions",
        "source_file_names",
        "source_sample_indices",
        "window_indices",
        "window_start_packets",
        "window_end_packets",
        "window_size",
        "stride",
    ]

    for key in required_keys:
        check_condition(
            key in available_keys,
            f"Missing required key: {key}",
        )

    x_windows = dataset["x_windows"]
    y_labels = dataset["y_labels"]
    grid_positions = dataset["grid_positions"]
    source_file_names = dataset["source_file_names"]
    source_sample_indices = dataset["source_sample_indices"]
    window_indices = dataset["window_indices"]
    window_start_packets = dataset["window_start_packets"]
    window_end_packets = dataset["window_end_packets"]
    window_size = int(dataset["window_size"])
    stride = int(dataset["stride"])

    print("BASIC SHAPES")
    print(f"x_windows shape: {x_windows.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"grid_positions shape: {grid_positions.shape}")
    print(f"source_file_names shape: {source_file_names.shape}")
    print(f"source_sample_indices shape: {source_sample_indices.shape}")
    print(f"window_indices shape: {window_indices.shape}")
    print(f"window_start_packets shape: {window_start_packets.shape}")
    print(f"window_end_packets shape: {window_end_packets.shape}")
    print()

    check_condition(
        x_windows.shape == EXPECTED_X_SHAPE,
        f"Invalid x_windows shape: {x_windows.shape}, expected {EXPECTED_X_SHAPE}",
    )

    check_condition(
        y_labels.shape == EXPECTED_Y_SHAPE,
        f"Invalid y_labels shape: {y_labels.shape}, expected {EXPECTED_Y_SHAPE}",
    )

    check_condition(
        grid_positions.shape == EXPECTED_GRID_SHAPE,
        f"Invalid grid_positions shape: {grid_positions.shape}, expected {EXPECTED_GRID_SHAPE}",
    )

    print("WINDOW CONFIGURATION")
    print(f"window size: {window_size}")
    print(f"stride: {stride}")
    print()

    check_condition(
        window_size == 30,
        f"Invalid window size: {window_size}, expected 30",
    )

    check_condition(
        stride == 30,
        f"Invalid stride: {stride}, expected 30",
    )

    print("DATA TYPES")
    print(f"x_windows dtype: {x_windows.dtype}")
    print(f"y_labels dtype: {y_labels.dtype}")
    print(f"grid_positions dtype: {grid_positions.dtype}")
    print()

    check_condition(
        x_windows.dtype == np.float32,
        f"Invalid x_windows dtype: {x_windows.dtype}, expected float32",
    )

    check_condition(
        np.issubdtype(y_labels.dtype, np.integer),
        f"Invalid y_labels dtype: {y_labels.dtype}, expected integer dtype",
    )

    check_condition(
        np.issubdtype(grid_positions.dtype, np.integer),
        f"Invalid grid_positions dtype: {grid_positions.dtype}, expected integer dtype",
    )

    print("NUMERIC CHECKS")
    print(f"finite values: {np.isfinite(x_windows).all()}")
    print(f"NaN count: {np.isnan(x_windows).sum()}")
    print(f"+inf count: {np.isposinf(x_windows).sum()}")
    print(f"-inf count: {np.isneginf(x_windows).sum()}")
    print()

    check_condition(
        np.isfinite(x_windows).all(),
        "x_windows contains NaN or infinite values",
    )

    print("VALUE STATISTICS")
    print(f"min: {x_windows.min():.6f}")
    print(f"max: {x_windows.max():.6f}")
    print(f"mean: {x_windows.mean():.6f}")
    print(f"std: {x_windows.std():.6f}")
    print()

    print("LABEL CHECKS")
    unique_labels = np.unique(y_labels)
    print(f"number of unique labels: {len(unique_labels)}")
    print(f"first labels: {unique_labels[:10]}")
    print(f"last labels: {unique_labels[-10:]}")
    print()

    check_condition(
        len(unique_labels) == EXPECTED_NUM_CLASSES,
        f"Invalid number of classes: {len(unique_labels)}, expected {EXPECTED_NUM_CLASSES}",
    )

    check_condition(
        np.array_equal(unique_labels, np.arange(EXPECTED_NUM_CLASSES)),
        "Labels are not consecutive from 0 to 99",
    )

    label_counts = Counter(y_labels.tolist())

    print("WINDOWS PER LABEL CHECK")
    print(f"minimum windows per label: {min(label_counts.values())}")
    print(f"maximum windows per label: {max(label_counts.values())}")
    print()

    check_condition(
        min(label_counts.values()) == EXPECTED_WINDOWS_PER_CLASS
        and max(label_counts.values()) == EXPECTED_WINDOWS_PER_CLASS,
        "Each label should have exactly 50 windows",
    )

    print("SOURCE FILE CHECK")
    print(f"first source file: {source_file_names[0]}")
    print(f"last source file: {source_file_names[-1]}")
    print(f"first source sample index: {source_sample_indices[0]}")
    print(f"last source sample index: {source_sample_indices[-1]}")
    print()

    check_condition(
        source_sample_indices[0] == 0,
        f"Invalid first source sample index: {source_sample_indices[0]}, expected 0",
    )

    check_condition(
        source_sample_indices[-1] == 99,
        f"Invalid last source sample index: {source_sample_indices[-1]}, expected 99",
    )

    print("WINDOW INDEX CHECK")
    print(f"first 10 window indices: {window_indices[:10]}")
    print(f"last 10 window indices: {window_indices[-10:]}")
    print()

    expected_window_indices = np.tile(
        np.arange(EXPECTED_WINDOWS_PER_CLASS),
        EXPECTED_NUM_CLASSES,
    )

    check_condition(
        np.array_equal(window_indices, expected_window_indices),
        "Window indices are not repeated correctly from 0 to 49 for each sample",
    )

    print("WINDOW PACKET RANGE CHECK")
    print(f"first window packet range: {window_start_packets[0]}:{window_end_packets[0]}")
    print(f"last window packet range: {window_start_packets[-1]}:{window_end_packets[-1]}")
    print()

    check_condition(
        window_start_packets[0] == 0 and window_end_packets[0] == 30,
        "Invalid first window packet range",
    )

    check_condition(
        window_start_packets[-1] == 1470 and window_end_packets[-1] == 1500,
        "Invalid last window packet range",
    )

    print("GRID POSITION CHECK")
    print(f"first grid position: {grid_positions[0]}")
    print(f"last grid position: {grid_positions[-1]}")
    print()

    check_condition(
        np.array_equal(grid_positions[0], np.array([1, 1])),
        f"Invalid first grid position: {grid_positions[0]}, expected [1, 1]",
    )

    check_condition(
        np.array_equal(grid_positions[-1], np.array([10, 1])),
        f"Invalid last grid position: {grid_positions[-1]}, expected [10, 1]",
    )

    print("SAMPLE VARIANCE CHECK")
    flattened_windows = x_windows.reshape(x_windows.shape[0], -1)
    window_std_values = flattened_windows.std(axis=1)

    zero_variance_indices = np.where(window_std_values == 0)[0]

    print(f"zero-variance windows: {len(zero_variance_indices)}")

    check_condition(
        len(zero_variance_indices) == 0,
        f"Found zero-variance windows at indices: {zero_variance_indices}",
    )

    print()
    print("WINDOWED DATASET VERIFICATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()