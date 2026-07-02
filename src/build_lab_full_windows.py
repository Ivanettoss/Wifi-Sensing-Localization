from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAB_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CSI-dataset-for-indoor-localization"
    / "Lab Dataset"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lab_full_windows_30.npz"
)

WINDOW_SIZE = 30
WINDOWS_PER_CLASS = 50
EXPECTED_TIME_SAMPLES = WINDOW_SIZE * WINDOWS_PER_CLASS


def find_lab_mat_files(raw_dir: Path) -> list[Path]:
    """
    Find all Lab Dataset .mat files, excluding the imaginary part.
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"Lab raw directory not found: {raw_dir}")

    mat_files = []

    for mat_file in raw_dir.rglob("*.mat"):
        path_parts_lower = [part.lower() for part in mat_file.parts]

        if "imaginary_part" in path_parts_lower:
            continue

        mat_files.append(mat_file)

    mat_files = sorted(
        mat_files,
        key=lambda path: parse_grid_position(path),
    )

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in: {raw_dir}")

    return mat_files


def parse_grid_position(mat_file: Path) -> tuple[int, int]:
    """
    Parse the grid position from a filename such as coordinate715.mat.

    The dataset convention is:
        coordinate715 -> position [7, 15]
        coordinate1001 -> position [10, 1]
    """

    match = re.search(r"coordinate\s*([0-9]+)", mat_file.stem.lower())

    if match is None:
        raise ValueError(f"Could not parse coordinate from filename: {mat_file.name}")

    coordinate_digits = match.group(1)

    if len(coordinate_digits) < 3:
        raise ValueError(f"Invalid coordinate format in filename: {mat_file.name}")

    x_position = int(coordinate_digits[:-2])
    y_position = int(coordinate_digits[-2:])

    return x_position, y_position

def replace_non_finite_values(
    csi_array: np.ndarray,
    mat_file: Path,
) -> np.ndarray:
    """
    Replace NaN and infinite values with finite values.

    Negative infinity is replaced with the minimum finite value in the file.
    Positive infinity is replaced with the maximum finite value in the file.
    NaN values are replaced with the mean finite value in the file.
    """

    finite_mask = np.isfinite(csi_array)

    if finite_mask.all():
        return csi_array

    if not finite_mask.any():
        raise ValueError(f"All values are non-finite in file: {mat_file}")

    finite_values = csi_array[finite_mask]

    finite_min = float(finite_values.min())
    finite_max = float(finite_values.max())
    finite_mean = float(finite_values.mean())

    non_finite_count = int((~finite_mask).sum())

    print(
        f"Warning: replaced {non_finite_count} non-finite values "
        f"in {mat_file.name}"
    )

    csi_array = np.nan_to_num(
        csi_array,
        nan=finite_mean,
        posinf=finite_max,
        neginf=finite_min,
    )

    return csi_array.astype(np.float32)

def load_csi_array(mat_file: Path) -> np.ndarray:
    """
    Load the first valid 3D numeric CSI array from a .mat file.
    """

    mat_data = loadmat(mat_file)

    candidate_arrays = []

    for key, value in mat_data.items():
        if key.startswith("__"):
            continue

        if not isinstance(value, np.ndarray):
            continue

        if value.ndim != 3:
            continue

        if not np.issubdtype(value.dtype, np.number):
            continue

        candidate_arrays.append(value)

    if len(candidate_arrays) == 0:
        available_keys = list(mat_data.keys())
        raise ValueError(
            f"No valid 3D numeric CSI array found in {mat_file}. "
            f"Available keys: {available_keys}"
        )

    candidate_arrays = sorted(
        candidate_arrays,
        key=lambda array: array.size,
        reverse=True,
    )

    csi_array = candidate_arrays[0].astype(np.float32)
    csi_array = replace_non_finite_values(csi_array, mat_file)  #due to -inf values in lab dataset


    return standardize_csi_shape(csi_array, mat_file)


def standardize_csi_shape(
    csi_array: np.ndarray,
    mat_file: Path,
) -> np.ndarray:
    """
    Convert CSI array to shape [3, 30, time_samples].
    """

    shape = csi_array.shape

    channel_axes = [axis for axis, size in enumerate(shape) if size == 3]
    subcarrier_axes = [axis for axis, size in enumerate(shape) if size == 30]

    if len(channel_axes) == 0:
        raise ValueError(f"No antenna/channel axis of size 3 found in {mat_file}: {shape}")

    if len(subcarrier_axes) == 0:
        raise ValueError(f"No subcarrier axis of size 30 found in {mat_file}: {shape}")

    channel_axis = channel_axes[0]

    subcarrier_axis = None
    for axis in subcarrier_axes:
        if axis != channel_axis:
            subcarrier_axis = axis
            break

    if subcarrier_axis is None:
        raise ValueError(f"Could not identify subcarrier axis in {mat_file}: {shape}")

    time_axes = [
        axis for axis in range(csi_array.ndim)
        if axis not in {channel_axis, subcarrier_axis}
    ]

    if len(time_axes) != 1:
        raise ValueError(f"Could not identify time axis in {mat_file}: {shape}")

    time_axis = time_axes[0]

    csi_array = np.transpose(
        csi_array,
        axes=(channel_axis, subcarrier_axis, time_axis),
    )

    if csi_array.shape[2] < EXPECTED_TIME_SAMPLES:
        raise ValueError(
            f"Not enough time samples in {mat_file}. "
            f"Expected at least {EXPECTED_TIME_SAMPLES}, found {csi_array.shape[2]}"
        )

    csi_array = csi_array[:, :, :EXPECTED_TIME_SAMPLES]

    return csi_array


def build_windows_from_csi(csi_array: np.ndarray) -> np.ndarray:
    """
    Split one CSI recording into non-overlapping temporal windows.

    Input shape:
        [3, 30, 1500]

    Output shape:
        [50, 3, 30, 30]
    """

    windows = []

    for window_index in range(WINDOWS_PER_CLASS):
        start_index = window_index * WINDOW_SIZE
        end_index = start_index + WINDOW_SIZE

        window = csi_array[:, :, start_index:end_index]
        windows.append(window)

    return np.stack(windows, axis=0)


def main() -> None:
    print("BUILD LAB FULL WINDOWS DATASET")
    print(f"raw directory: {LAB_RAW_DIR}")
    print(f"output file: {OUTPUT_FILE}")
    print()

    mat_files = find_lab_mat_files(LAB_RAW_DIR)

    print("FOUND FILES")
    print(f"number of .mat files: {len(mat_files)}")
    print()

    all_windows = []
    all_labels = []
    all_grid_positions = []
    class_grid_positions = []
    source_files = []

    seen_positions = set()

    for class_id, mat_file in enumerate(mat_files):
        grid_position = parse_grid_position(mat_file)

        if grid_position in seen_positions:
            raise ValueError(f"Duplicate grid position found: {grid_position}")

        seen_positions.add(grid_position)

        csi_array = load_csi_array(mat_file)
        windows = build_windows_from_csi(csi_array)

        labels = np.full(
            shape=(WINDOWS_PER_CLASS,),
            fill_value=class_id,
            dtype=np.int64,
        )

        repeated_positions = np.repeat(
            np.array(grid_position, dtype=np.int64)[None, :],
            repeats=WINDOWS_PER_CLASS,
            axis=0,
        )

        all_windows.append(windows)
        all_labels.append(labels)
        all_grid_positions.append(repeated_positions)

        class_grid_positions.append(grid_position)
        source_files.append(str(mat_file.relative_to(PROJECT_ROOT)))

        if (class_id + 1) % 50 == 0 or class_id == len(mat_files) - 1:
            print(
                f"processed {class_id + 1}/{len(mat_files)} files "
                f"| last file: {mat_file.name} "
                f"| position: {grid_position} "
                f"| csi shape: {tuple(csi_array.shape)}"
            )

    x_windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    y_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    grid_positions = np.concatenate(all_grid_positions, axis=0).astype(np.int64)

    class_grid_positions = np.array(
        class_grid_positions,
        dtype=np.int64,
    )

    source_files = np.array(
        source_files,
        dtype=str,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        x_windows=x_windows,
        y_labels=y_labels,
        grid_positions=grid_positions,
        class_grid_positions=class_grid_positions,
        source_files=source_files,
        window_size=WINDOW_SIZE,
        windows_per_class=WINDOWS_PER_CLASS,
        raw_directory=str(LAB_RAW_DIR),
    )

    print()
    print("DATASET SAVED")
    print(f"output file: {OUTPUT_FILE}")
    print(f"x_windows shape: {x_windows.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"grid_positions shape: {grid_positions.shape}")
    print(f"class_grid_positions shape: {class_grid_positions.shape}")
    print(f"num_classes: {len(class_grid_positions)}")
    print(f"windows per class: {WINDOWS_PER_CLASS}")
    print(f"window size: {WINDOW_SIZE}")
    print()
    print("EXPECTED")
    print(f"expected windows: {len(class_grid_positions)} * {WINDOWS_PER_CLASS} = {len(class_grid_positions) * WINDOWS_PER_CLASS}")


if __name__ == "__main__":
    main()