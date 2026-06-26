#This script audits the CSI dataset for indoor localization. 
# It checks the structure of .mat files, summarizes their contents, and compares coordinate and imaginary parts for consistency.
#Due to the dataset authors' not creation of a READMI file, I am trying to understand the dataset

from pathlib import Path
from collections import Counter, defaultdict
from scipy.io import loadmat
import numpy as np


DATA_ROOT = Path("data/raw/CSI-dataset-for-indoor-localization")


def get_mat_info(path: Path):
    data = loadmat(path)
    if "myData" not in data:
        return None

    x = np.asarray(data["myData"])

    return {
        "shape": x.shape,
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def list_mat_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.mat"))


def summarize_folder(folder: Path, label: str):
    files = list_mat_files(folder)
    shape_counter = Counter()

    print("=" * 90)
    print(label)
    print(folder)
    print("=" * 90)
    print(f"Number of .mat files: {len(files)}")

    if not files:
        print("No files found.")
        return

    stats_examples = []

    for f in files:
        info = get_mat_info(f)
        if info is None:
            shape_counter["NO_myData"] += 1
        else:
            shape_counter[info["shape"]] += 1
            if len(stats_examples) < 3:
                stats_examples.append((f.name, info))

    print("Shapes:")
    for shape, count in shape_counter.items():
        print(f"  {shape}: {count}")

    print("Examples:")
    for name, info in stats_examples:
        print(
            f"  {name}: shape={info['shape']}, "
            f"min={info['min']:.4f}, max={info['max']:.4f}, "
            f"mean={info['mean']:.4f}, std={info['std']:.4f}"
        )

    print()


def extract_numeric_id(path: Path) -> str:
    """
    Examples:
    coordinate1001.mat -> 1001
    imaginary1001.mat  -> 1001
    1001.mat           -> 1001
    """
    import re
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"No numeric id found in {path.name}")
    return numbers[0]


def compare_coordinate_and_imaginary(dataset_name: str, coordinate_folders: list[str]):
    dataset_root = DATA_ROOT / dataset_name
    imaginary_root = dataset_root / "imaginary_part"

    coord_files = []

    for folder_name in coordinate_folders:
        folder = dataset_root / folder_name
        coord_files.extend(list_mat_files(folder))

    imag_files = list_mat_files(imaginary_root)

    coord_map = {extract_numeric_id(f): f for f in coord_files}
    imag_map = {extract_numeric_id(f): f for f in imag_files}

    coord_ids = set(coord_map.keys())
    imag_ids = set(imag_map.keys())

    print("=" * 90)
    print(f"REAL / IMAGINARY MATCH CHECK — {dataset_name}")
    print("=" * 90)

    print(f"Coordinate files: {len(coord_files)}")
    print(f"Imaginary files:  {len(imag_files)}")

    missing_in_imag = sorted(coord_ids - imag_ids)
    extra_in_imag = sorted(imag_ids - coord_ids)

    print(f"Missing numeric IDs in imaginary_part: {len(missing_in_imag)}")
    print(f"Extra numeric IDs in imaginary_part:   {len(extra_in_imag)}")

    if missing_in_imag[:10]:
        print("First missing IDs:", missing_in_imag[:10])

    if extra_in_imag[:10]:
        print("First extra IDs:", extra_in_imag[:10])

    mismatched_shapes = []

    for numeric_id in sorted(coord_ids & imag_ids):
        real_info = get_mat_info(coord_map[numeric_id])
        imag_info = get_mat_info(imag_map[numeric_id])

        if real_info is None or imag_info is None:
            mismatched_shapes.append((numeric_id, "missing myData"))
            continue

        if real_info["shape"] != imag_info["shape"]:
            mismatched_shapes.append(
                (numeric_id, real_info["shape"], imag_info["shape"])
            )

    print(f"Shape mismatches: {len(mismatched_shapes)}")
    if mismatched_shapes[:10]:
        print("First mismatches:", mismatched_shapes[:10])

    print()


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")

    print("DATASET ROOT")
    print(DATA_ROOT.resolve())
    print()

    folders_to_check = [
        ("Meeting Room Dataset / coordinate 1-100", DATA_ROOT / "Meeting Room Dataset" / "coordinate 1-100"),
        ("Meeting Room Dataset / coordinate 101-176", DATA_ROOT / "Meeting Room Dataset" / "coordinate 101-176"),
        ("Meeting Room Dataset / imaginary_part", DATA_ROOT / "Meeting Room Dataset" / "imaginary_part"),

        ("Lab Dataset / coordinate 1-100", DATA_ROOT / "Lab Dataset" / "coordinate 1-100"),
        ("Lab Dataset / coordinate 101-200", DATA_ROOT / "Lab Dataset" / "coordinate 101-200"),
        ("Lab Dataset / coordinate 201-300", DATA_ROOT / "Lab Dataset" / "coordinate 201-300"),
        ("Lab Dataset / coordinate 301-317", DATA_ROOT / "Lab Dataset" / "coordinate 301-317"),
        ("Lab Dataset / imaginary_part", DATA_ROOT / "Lab Dataset" / "imaginary_part"),

        ("Conference Room / coordinate 1-100", DATA_ROOT / "Conference Room" / "coordinate 1-100"),
        ("Conference Room / coordinate 101-160", DATA_ROOT / "Conference Room" / "coordinate 101-160"),

        ("miniLab / coordinate 1-35", DATA_ROOT / "miniLab" / "coordinate 1-35"),
    ]

    for label, folder in folders_to_check:
        summarize_folder(folder, label)

    compare_coordinate_and_imaginary(
        "Meeting Room Dataset",
        ["coordinate 1-100", "coordinate 101-176"],
    )

    compare_coordinate_and_imaginary(
        "Lab Dataset",
        ["coordinate 1-100", "coordinate 101-200", "coordinate 201-300", "coordinate 301-317"],
    )

    print("=" * 90)
    print("PRACTICAL DECISION")
    print("=" * 90)
    print("Use first:")
    print("  Meeting Room Dataset coordinate files")
    print()
    print("Use later:")
    print("  Lab Dataset coordinate files")
    print("  imaginary_part if matching is correct")
    print()
    print("Do not use initially:")
    print("  Conference Room")
    print("  miniLab")
    print()


if __name__ == "__main__":
    main()