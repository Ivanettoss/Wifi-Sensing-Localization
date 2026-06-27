from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "meeting_room_1_100_raw.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "meeting_room_1_100_windows_30.npz"

WINDOW_SIZE = 30
STRIDE = 30

EXPECTED_INPUT_SHAPE = (100, 3, 30, 1500)
EXPECTED_WINDOW_SHAPE = (3, 30, 30)


def check_condition(condition: bool, error_message: str) -> None:
    """
    Raise a clear error if a validation check fails.
    """

    if not condition:
        raise ValueError(error_message)


def build_temporal_windows(
    x_data: np.ndarray,
    y_labels: np.ndarray,
    grid_positions: np.ndarray,
    file_names: np.ndarray,
) -> dict:
    """
    Build non-overlapping temporal windows from raw CSI samples.

    Each raw CSI sample has shape:
        (antennas, subcarriers, time_packets)

    With WINDOW_SIZE = 30 and STRIDE = 30:
        (3, 30, 1500) -> 50 windows of shape (3, 30, 30)
    """

    num_samples, num_antennas, num_subcarriers, num_time_packets = x_data.shape

    check_condition(
        num_time_packets >= WINDOW_SIZE,
        f"Time dimension is smaller than window size: "
        f"{num_time_packets} < {WINDOW_SIZE}",
    )

    window_start_indices = list(
        range(0, num_time_packets - WINDOW_SIZE + 1, STRIDE)
    )

    windows = []
    window_labels = []
    window_grid_positions = []
    source_file_names = []
    source_sample_indices = []
    window_indices = []
    window_start_packets = []
    window_end_packets = []

    for sample_index in range(num_samples):
        csi_sample = x_data[sample_index]
        label = y_labels[sample_index]
        grid_position = grid_positions[sample_index]
        source_file_name = file_names[sample_index]

        for window_index, start_packet in enumerate(window_start_indices):
            end_packet = start_packet + WINDOW_SIZE

            csi_window = csi_sample[:, :, start_packet:end_packet]

            check_condition(
                csi_window.shape == EXPECTED_WINDOW_SHAPE,
                f"Invalid window shape: {csi_window.shape}, "
                f"expected {EXPECTED_WINDOW_SHAPE}",
            )

            windows.append(csi_window)
            window_labels.append(label)
            window_grid_positions.append(grid_position)
            source_file_names.append(source_file_name)
            source_sample_indices.append(sample_index)
            window_indices.append(window_index)
            window_start_packets.append(start_packet)
            window_end_packets.append(end_packet)

    x_windows = np.stack(windows, axis=0).astype(np.float32)
    y_window_labels = np.array(window_labels, dtype=np.int64)
    window_grid_positions = np.array(window_grid_positions, dtype=np.int64)
    source_file_names = np.array(source_file_names)
    source_sample_indices = np.array(source_sample_indices, dtype=np.int64)
    window_indices = np.array(window_indices, dtype=np.int64)
    window_start_packets = np.array(window_start_packets, dtype=np.int64)
    window_end_packets = np.array(window_end_packets, dtype=np.int64)

    return {
        "x_windows": x_windows,
        "y_labels": y_window_labels,
        "grid_positions": window_grid_positions,
        "source_file_names": source_file_names,
        "source_sample_indices": source_sample_indices,
        "window_indices": window_indices,
        "window_start_packets": window_start_packets,
        "window_end_packets": window_end_packets,
    }


def main() -> None:
    print("INPUT FILE")
    print(INPUT_FILE)
    print()

    check_condition(
        INPUT_FILE.exists(),
        f"Input dataset file not found: {INPUT_FILE}",
    )

    dataset = np.load(INPUT_FILE)

    required_keys = [
        "x_data",
        "y_labels",
        "file_names",
        "selected_keys",
        "grid_positions",
    ]

    available_keys = list(dataset.keys())

    print("AVAILABLE KEYS")
    print(available_keys)
    print()

    for key in required_keys:
        check_condition(
            key in available_keys,
            f"Missing required key in raw dataset: {key}",
        )

    x_data = dataset["x_data"]
    y_labels = dataset["y_labels"]
    file_names = dataset["file_names"]
    grid_positions = dataset["grid_positions"]

    print("RAW DATASET")
    print(f"x_data shape: {x_data.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"file_names shape: {file_names.shape}")
    print(f"grid_positions shape: {grid_positions.shape}")
    print()

    check_condition(
        x_data.shape == EXPECTED_INPUT_SHAPE,
        f"Invalid x_data shape: {x_data.shape}, "
        f"expected {EXPECTED_INPUT_SHAPE}",
    )

    check_condition(
        np.isfinite(x_data).all(),
        "x_data contains NaN or infinite values",
    )

    windowed_dataset = build_temporal_windows(
        x_data=x_data,
        y_labels=y_labels,
        grid_positions=grid_positions,
        file_names=file_names,
    )

    x_windows = windowed_dataset["x_windows"]
    y_window_labels = windowed_dataset["y_labels"]
    window_grid_positions = windowed_dataset["grid_positions"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        **windowed_dataset,
        window_size=np.array(WINDOW_SIZE, dtype=np.int64),
        stride=np.array(STRIDE, dtype=np.int64),
    )

    print("WINDOWED DATASET CREATED")
    print(f"x_windows shape: {x_windows.shape}")
    print(f"y_labels shape: {y_window_labels.shape}")
    print(f"grid_positions shape: {window_grid_positions.shape}")
    print(f"window size: {WINDOW_SIZE}")
    print(f"stride: {STRIDE}")
    print(f"output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()