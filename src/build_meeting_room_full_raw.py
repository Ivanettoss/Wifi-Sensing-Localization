import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CSI-dataset-for-indoor-localization"
    / "Meeting Room Dataset"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "meeting_room_full_raw.npz"
)

EXPECTED_SHAPE = (3, 30, 1500)

EXCLUDED_PATH_KEYWORDS = [
    "imaginary",
    "imag",
]


def should_exclude_file(file_path: Path) -> bool:
    """
    Check whether a .mat file should be excluded.

    For now, we exclude possible imaginary CSI folders because this baseline
    uses the same real/raw CSI component used in the first experiment.
    """

    path_text = str(file_path).lower()

    return any(keyword in path_text for keyword in EXCLUDED_PATH_KEYWORDS)


def parse_grid_position(file_name: str) -> tuple[int, int]:
    """
    Parse the grid position from a coordinate file name.

    Examples:
        coordinate101.mat  -> (1, 1)
        coordinate715.mat  -> (7, 15)
        coordinate1001.mat -> (10, 1)

    Convention:
        row = all digits except the last two
        column = last two digits
    """

    file_stem = Path(file_name).stem.lower()

    match = re.search(r"coordinate\s*(\d+)", file_stem)

    if match is None:
        raise ValueError(f"Invalid coordinate file name: {file_name}")

    coordinate_id = match.group(1)

    if len(coordinate_id) < 3:
        raise ValueError(f"Coordinate id is too short: {coordinate_id}")

    row = int(coordinate_id[:-2])
    column = int(coordinate_id[-2:])

    return row, column


def coordinate_sort_key(file_path: Path) -> tuple[int, int, str]:
    """
    Sort files according to their grid position.
    """

    row, column = parse_grid_position(file_path.name)

    return row, column, str(file_path)


def get_selected_key(mat_content: dict) -> str:
    """
    Select the key containing the CSI matrix.

    The dataset usually stores CSI data in the key 'myData'. If this key is not
    found, the function searches for the first numeric ndarray with the expected
    shape.
    """

    if "myData" in mat_content:
        return "myData"

    for key, value in mat_content.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and value.shape == EXPECTED_SHAPE:
            return key

    raise ValueError("No valid CSI matrix key found in .mat file")


def main() -> None:
    print("BUILD MEETING ROOM FULL RAW DATASET")
    print(f"Dataset folder: {DATASET_FOLDER}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    if not DATASET_FOLDER.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_FOLDER}")

    all_mat_files = list(DATASET_FOLDER.rglob("*.mat"))

    mat_files = [
        file_path
        for file_path in all_mat_files
        if not should_exclude_file(file_path)
    ]

    mat_files = sorted(mat_files, key=coordinate_sort_key)

    excluded_files = len(all_mat_files) - len(mat_files)

    print(f"Total .mat files found: {len(all_mat_files)}")
    print(f"Excluded .mat files: {excluded_files}")
    print(f"Used .mat files: {len(mat_files)}")
    print()

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No valid .mat files found in: {DATASET_FOLDER}")

    x_data = []
    y_labels = []
    file_names = []
    selected_keys = []
    grid_positions = []

    shape_counter = {}
    key_counter = {}
    seen_grid_positions = {}

    for label, mat_file in enumerate(mat_files):
        grid_position = parse_grid_position(mat_file.name)

        if grid_position in seen_grid_positions:
            previous_file = seen_grid_positions[grid_position]
            raise ValueError(
                f"Duplicate grid position {grid_position} found:\n"
                f"  previous: {previous_file}\n"
                f"  current: {mat_file}"
            )

        seen_grid_positions[grid_position] = mat_file

        mat_content = loadmat(mat_file)
        selected_key = get_selected_key(mat_content)

        csi_matrix = mat_content[selected_key]

        if csi_matrix.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Invalid shape for {mat_file.name}: "
                f"{csi_matrix.shape}, expected {EXPECTED_SHAPE}"
            )

        if not np.isfinite(csi_matrix).all():
            raise ValueError(f"NaN or infinite values found in {mat_file.name}")

        csi_matrix = csi_matrix.astype(np.float32)

        x_data.append(csi_matrix)
        y_labels.append(label)
        file_names.append(str(mat_file.relative_to(DATASET_FOLDER)))
        selected_keys.append(selected_key)
        grid_positions.append(grid_position)

        shape_counter[csi_matrix.shape] = shape_counter.get(csi_matrix.shape, 0) + 1
        key_counter[selected_key] = key_counter.get(selected_key, 0) + 1

    x_data = np.stack(x_data, axis=0)
    y_labels = np.array(y_labels, dtype=np.int64)
    file_names = np.array(file_names)
    selected_keys = np.array(selected_keys)
    grid_positions = np.array(grid_positions, dtype=np.int64)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        x_data=x_data,
        y_labels=y_labels,
        file_names=file_names,
        selected_keys=selected_keys,
        grid_positions=grid_positions,
    )

    print("RAW DATASET CREATED")
    print(f"x_data shape: {x_data.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"file_names shape: {file_names.shape}")
    print(f"selected_keys shape: {selected_keys.shape}")
    print(f"grid_positions shape: {grid_positions.shape}")
    print()

    print("VALUE STATISTICS")
    print(f"min: {x_data.min():.6f}")
    print(f"max: {x_data.max():.6f}")
    print(f"mean: {x_data.mean():.6f}")
    print(f"std: {x_data.std():.6f}")
    print()

    print("Shapes found:")
    for shape, count in shape_counter.items():
        print(f"  {shape}: {count}")

    print()

    print("Selected keys:")
    for key, count in key_counter.items():
        print(f"  {key}: {count}")

    print()

    print("First files:")
    for index in range(min(5, len(file_names))):
        print(
            f"  [{index:03d}] "
            f"{file_names[index]} "
            f"label={y_labels[index]} "
            f"grid={grid_positions[index].tolist()}"
        )

    print()

    print("Last files:")
    start_index = max(0, len(file_names) - 5)

    for index in range(start_index, len(file_names)):
        print(
            f"  [{index:03d}] "
            f"{file_names[index]} "
            f"label={y_labels[index]} "
            f"grid={grid_positions[index].tolist()}"
        )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()