#This is a test loader, we want to load only the first 100 files of the "Meeting Room Dataset" for initial testing.

from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CSI-dataset-for-indoor-localization"
    / "Meeting Room Dataset"
    / "coordinate 1-100"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "meeting_room_1_100_raw.npz"

EXPECTED_SHAPE = (3, 30, 1500)


def extract_csi_matrix(mat_content: dict, file_name: str) -> tuple[np.ndarray, str]:
    """
    Extract the CSI matrix from a MATLAB .mat file.
    Ignore MATLAB system keys
    and search for a numeric array with the expected CSI shape.
    """

    candidate_matrices = {}

    for key, value in mat_content.items():
        if key.startswith("__"):
            continue

        if not isinstance(value, np.ndarray):
            continue

        squeezed_value = np.squeeze(value)

        if squeezed_value.shape == EXPECTED_SHAPE:
            candidate_matrices[key] = squeezed_value

    if len(candidate_matrices) == 0:
        raise ValueError(
            f"No CSI matrix with shape {EXPECTED_SHAPE} found in {file_name}"
        )

    if len(candidate_matrices) > 1:
        print(
            f"Warning: multiple CSI candidates found in {file_name}: "
            f"{list(candidate_matrices.keys())}. Using the first one."
        )

    selected_key = list(candidate_matrices.keys())[0]
    selected_matrix = candidate_matrices[selected_key]

    return selected_matrix, selected_key


def main() -> None:
    print("DATASET DIRECTORY")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    mat_files = sorted(DATASET_DIR.glob("*.mat"))

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in: {DATASET_DIR}")

    print()
    print(f"Number of .mat files found: {len(mat_files)}")
    print()

    csi_samples = []
    labels = []
    file_names = []
    selected_keys = []

    for class_index, mat_path in enumerate(mat_files):
        mat_content = loadmat(mat_path)

        csi_matrix, selected_key = extract_csi_matrix(
            mat_content=mat_content,
            file_name=mat_path.name,
        )

        if csi_matrix.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Invalid shape for {mat_path.name}: "
                f"{csi_matrix.shape}, expected {EXPECTED_SHAPE}"
            )

        if not np.isfinite(csi_matrix).all():
            raise ValueError(f"NaN or infinite values found in {mat_path.name}")

        csi_matrix = csi_matrix.astype(np.float32)

        csi_samples.append(csi_matrix)
        labels.append(class_index)
        file_names.append(mat_path.name)
        selected_keys.append(selected_key)

        print(
            f"[{class_index:03d}] {mat_path.name} "
            f"shape={csi_matrix.shape} key={selected_key}"
        )

    x_data = np.stack(csi_samples, axis=0)
    y_labels = np.array(labels, dtype=np.int64)
    file_names = np.array(file_names)
    selected_keys = np.array(selected_keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        x_data=x_data,
        y_labels=y_labels,
        file_names=file_names,
        selected_keys=selected_keys,
    )

    print()
    print("DATASET CREATED")
    print(f"x_data shape: {x_data.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()