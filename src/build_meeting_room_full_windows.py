from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "meeting_room_full_raw.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "meeting_room_full_windows_30.npz"
)

WINDOW_SIZE = 30
STRIDE = 30

EXPECTED_SAMPLE_SHAPE = (3, 30, 1500)


def compute_number_of_windows(
    num_packets: int,
    window_size: int,
    stride: int,
) -> int:
    """
    Compute the number of temporal windows for one CSI sample.
    """

    if num_packets < window_size:
        raise ValueError(
            f"Number of packets {num_packets} is smaller than "
            f"window size {window_size}"
        )

    return ((num_packets - window_size) // stride) + 1


def main() -> None:
    print("BUILD MEETING ROOM FULL WINDOWED DATASET")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    raw_dataset = np.load(INPUT_FILE)

    x_data = raw_dataset["x_data"]
    y_labels = raw_dataset["y_labels"]
    file_names = raw_dataset["file_names"]
    grid_positions = raw_dataset["grid_positions"]

    print("RAW DATASET")
    print(f"x_data shape: {x_data.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"file_names shape: {file_names.shape}")
    print(f"grid_positions shape: {grid_positions.shape}")
    print()

    if x_data.ndim != 4:
        raise ValueError(f"Invalid x_data ndim: {x_data.ndim}, expected 4")

    if x_data.shape[1:] != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(
            f"Invalid sample shape: {x_data.shape[1:]}, "
            f"expected {EXPECTED_SAMPLE_SHAPE}"
        )

    if not np.isfinite(x_data).all():
        raise ValueError("NaN or infinite values found in x_data")

    num_samples = x_data.shape[0]
    num_antennas = x_data.shape[1]
    num_subcarriers = x_data.shape[2]
    num_packets = x_data.shape[3]

    windows_per_sample = compute_number_of_windows(
        num_packets=num_packets,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    print("WINDOWING CONFIGURATION")
    print(f"num_samples: {num_samples}")
    print(f"num_antennas: {num_antennas}")
    print(f"num_subcarriers: {num_subcarriers}")
    print(f"num_packets: {num_packets}")
    print(f"windows per sample: {windows_per_sample}")
    print()

    x_windows = []
    window_labels = []
    window_grid_positions = []
    source_file_names = []
    source_sample_indices = []
    window_indices = []
    window_start_packets = []
    window_end_packets = []

    for sample_index in range(num_samples):
        sample = x_data[sample_index]
        label = y_labels[sample_index]
        file_name = file_names[sample_index]
        grid_position = grid_positions[sample_index]

        for window_index in range(windows_per_sample):
            start_packet = window_index * STRIDE
            end_packet = start_packet + WINDOW_SIZE

            window = sample[:, :, start_packet:end_packet]

            if window.shape != (num_antennas, num_subcarriers, WINDOW_SIZE):
                raise ValueError(
                    f"Invalid window shape for sample {sample_index}, "
                    f"window {window_index}: {window.shape}"
                )

            x_windows.append(window)
            window_labels.append(label)
            window_grid_positions.append(grid_position)
            source_file_names.append(file_name)
            source_sample_indices.append(sample_index)
            window_indices.append(window_index)
            window_start_packets.append(start_packet)
            window_end_packets.append(end_packet)

    x_windows = np.stack(x_windows, axis=0).astype(np.float32)
    window_labels = np.array(window_labels, dtype=np.int64)
    window_grid_positions = np.array(window_grid_positions, dtype=np.int64)
    source_file_names = np.array(source_file_names)
    source_sample_indices = np.array(source_sample_indices, dtype=np.int64)
    window_indices = np.array(window_indices, dtype=np.int64)
    window_start_packets = np.array(window_start_packets, dtype=np.int64)
    window_end_packets = np.array(window_end_packets, dtype=np.int64)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        x_windows=x_windows,
        y_labels=window_labels,
        grid_positions=window_grid_positions,
        source_file_names=source_file_names,
        source_sample_indices=source_sample_indices,
        window_indices=window_indices,
        window_start_packets=window_start_packets,
        window_end_packets=window_end_packets,
        window_size=np.array(WINDOW_SIZE, dtype=np.int64),
        stride=np.array(STRIDE, dtype=np.int64),
    )

    print("WINDOWED DATASET CREATED")
    print(f"x_windows shape: {x_windows.shape}")
    print(f"y_labels shape: {window_labels.shape}")
    print(f"grid_positions shape: {window_grid_positions.shape}")
    print(f"source_file_names shape: {source_file_names.shape}")
    print(f"source_sample_indices shape: {source_sample_indices.shape}")
    print(f"window_indices shape: {window_indices.shape}")
    print()

    print("VALUE STATISTICS")
    print(f"min: {x_windows.min():.6f}")
    print(f"max: {x_windows.max():.6f}")
    print(f"mean: {x_windows.mean():.6f}")
    print(f"std: {x_windows.std():.6f}")
    print()

    unique_labels, label_counts = np.unique(window_labels, return_counts=True)

    print("LABEL STATISTICS")
    print(f"number of classes: {len(unique_labels)}")
    print(f"min windows per class: {label_counts.min()}")
    print(f"max windows per class: {label_counts.max()}")
    print()

    print("FIRST WINDOWS")
    for index in range(min(5, len(x_windows))):
        print(
            f"  [{index:03d}] "
            f"source={source_file_names[index]} "
            f"label={window_labels[index]} "
            f"grid={window_grid_positions[index].tolist()} "
            f"window={window_indices[index]} "
            f"packets={window_start_packets[index]}:{window_end_packets[index]}"
        )

    print()

    print("LAST WINDOWS")
    start_index = max(0, len(x_windows) - 5)
    for index in range(start_index, len(x_windows)):
        print(
            f"  [{index:03d}] "
            f"source={source_file_names[index]} "
            f"label={window_labels[index]} "
            f"grid={window_grid_positions[index].tolist()} "
            f"window={window_indices[index]} "
            f"packets={window_start_packets[index]}:{window_end_packets[index]}"
        )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()