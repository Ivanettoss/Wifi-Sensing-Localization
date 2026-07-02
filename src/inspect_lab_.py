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


def load_raw_csi_array(mat_file: Path) -> np.ndarray:
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
        axis
        for axis in range(csi_array.ndim)
        if axis not in {channel_axis, subcarrier_axis}
    ]

    if len(time_axes) != 1:
        raise ValueError(f"Could not identify time axis in {mat_file}: {shape}")

    time_axis = time_axes[0]

    csi_array = np.transpose(
        csi_array,
        axes=(channel_axis, subcarrier_axis, time_axis),
    )

    return csi_array


def print_counts(
    title: str,
    values: np.ndarray,
) -> None:
    """
    Print counts for unique integer values.
    """

    unique_values, counts = np.unique(values, return_counts=True)

    print()
    print(title)
    print("-" * len(title))

    for value, count in zip(unique_values, counts):
        print(f"{int(value)}: {int(count)}")


def main() -> None:
    print("LAB RAW NON-FINITE VALUE INSPECTION")
    print(f"raw directory: {LAB_RAW_DIR}")
    print()

    mat_files = find_lab_mat_files(LAB_RAW_DIR)

    print("FOUND FILES")
    print(f"number of .mat files: {len(mat_files)}")
    print()

    total_values = 0
    total_non_finite = 0
    total_nan = 0
    total_posinf = 0
    total_neginf = 0

    all_antenna_ids = []
    all_subcarrier_ids = []
    all_time_ids = []

    bad_file_summaries = []
    detailed_examples = []

    for file_index, mat_file in enumerate(mat_files):
        grid_position = parse_grid_position(mat_file)
        csi_array = load_raw_csi_array(mat_file)

        total_values += csi_array.size

        non_finite_mask = ~np.isfinite(csi_array)
        nan_count = int(np.isnan(csi_array).sum())
        posinf_count = int(np.isposinf(csi_array).sum())
        neginf_count = int(np.isneginf(csi_array).sum())
        non_finite_count = int(non_finite_mask.sum())

        total_nan += nan_count
        total_posinf += posinf_count
        total_neginf += neginf_count
        total_non_finite += non_finite_count

        if non_finite_count == 0:
            continue

        non_finite_positions = np.argwhere(non_finite_mask)

        antenna_ids = non_finite_positions[:, 0]
        subcarrier_ids = non_finite_positions[:, 1]
        time_ids = non_finite_positions[:, 2]

        all_antenna_ids.append(antenna_ids)
        all_subcarrier_ids.append(subcarrier_ids)
        all_time_ids.append(time_ids)

        bad_file_summaries.append(
            {
                "file_index": file_index,
                "file_name": mat_file.name,
                "relative_path": str(mat_file.relative_to(PROJECT_ROOT)),
                "grid_position": grid_position,
                "shape": tuple(csi_array.shape),
                "non_finite_count": non_finite_count,
                "nan_count": nan_count,
                "posinf_count": posinf_count,
                "neginf_count": neginf_count,
            }
        )

        for position in non_finite_positions[:10]:
            antenna_id = int(position[0])
            subcarrier_id = int(position[1])
            time_id = int(position[2])
            value = csi_array[antenna_id, subcarrier_id, time_id]

            detailed_examples.append(
                {
                    "file_name": mat_file.name,
                    "relative_path": str(mat_file.relative_to(PROJECT_ROOT)),
                    "grid_position": grid_position,
                    "value": value,
                    "antenna_id": antenna_id,
                    "subcarrier_id": subcarrier_id,
                    "time_id": time_id,
                }
            )

    print("GLOBAL SUMMARY")
    print("--------------")
    print(f"total values: {total_values}")
    print(f"non-finite values: {total_non_finite}")
    print(f"nan values: {total_nan}")
    print(f"+inf values: {total_posinf}")
    print(f"-inf values: {total_neginf}")

    if total_non_finite == 0:
        print()
        print("No non-finite values found in raw Lab .mat files.")
        return

    print()
    print("percentage of non-finite values:")
    print(f"{(total_non_finite / total_values) * 100:.8f}%")

    all_antenna_ids = np.concatenate(all_antenna_ids)
    all_subcarrier_ids = np.concatenate(all_subcarrier_ids)
    all_time_ids = np.concatenate(all_time_ids)

    print_counts("Counts by antenna/channel axis", all_antenna_ids)
    print_counts("Counts by subcarrier axis", all_subcarrier_ids)

    print()
    print("TIME POSITION SUMMARY")
    print("---------------------")
    print(f"min time index: {int(all_time_ids.min())}")
    print(f"max time index: {int(all_time_ids.max())}")

    print()
    print("FILES WITH NON-FINITE VALUES")
    print("----------------------------")

    bad_file_summaries = sorted(
        bad_file_summaries,
        key=lambda item: item["non_finite_count"],
        reverse=True,
    )

    for item in bad_file_summaries:
        print(
            f"file={item['file_name']} | "
            f"grid={item['grid_position']} | "
            f"shape={item['shape']} | "
            f"non_finite={item['non_finite_count']} | "
            f"nan={item['nan_count']} | "
            f"+inf={item['posinf_count']} | "
            f"-inf={item['neginf_count']} | "
            f"path={item['relative_path']}"
        )

    print()
    print("FIRST 50 NON-FINITE EXAMPLES")
    print("----------------------------")

    for example_index, example in enumerate(detailed_examples[:50], start=1):
        print(
            f"{example_index:02d}) "
            f"value={example['value']} | "
            f"file={example['file_name']} | "
            f"grid={example['grid_position']} | "
            f"antenna={example['antenna_id']} | "
            f"subcarrier={example['subcarrier_id']} | "
            f"time_index={example['time_id']} | "
            f"path={example['relative_path']}"
        )


if __name__ == "__main__":
    main()