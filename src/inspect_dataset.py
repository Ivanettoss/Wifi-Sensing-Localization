# This script perform a brief inspection of the CSI dataset for indoor localization
#No readmi from the dataset author, sadly

from pathlib import Path
from collections import Counter, defaultdict
from scipy.io import loadmat
import numpy as np


DATA_ROOT = Path("data/raw/CSI-dataset-for-indoor-localization")


def describe_mat_file(path: Path):
    data = loadmat(path)
    variables = {}

    for key, value in data.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):
            variables[key] = {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "min": float(np.nanmin(value)) if np.issubdtype(value.dtype, np.number) else None,
                "max": float(np.nanmax(value)) if np.issubdtype(value.dtype, np.number) else None,
                "mean": float(np.nanmean(value)) if np.issubdtype(value.dtype, np.number) else None,
            }
        else:
            variables[key] = {
                "type": str(type(value))
            }

    return variables


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_ROOT}")

    mat_files = sorted(DATA_ROOT.rglob("*.mat"))

    print("=" * 80)
    print("DATASET ROOT")
    print(DATA_ROOT.resolve())
    print("=" * 80)
    print(f"Number of .mat files found: {len(mat_files)}")
    print()

    print("First 20 .mat files:")
    for path in mat_files[:20]:
        print(" -", path.relative_to(DATA_ROOT))
    print()

    shape_counter = Counter()
    variable_counter = Counter()
    folder_counter = Counter()

    print("=" * 80)
    print("FIRST FILES INSPECTION")
    print("=" * 80)

    for path in mat_files[:10]:
        print()
        print("FILE:", path.relative_to(DATA_ROOT))
        variables = describe_mat_file(path)

        for var_name, info in variables.items():
            variable_counter[var_name] += 1

            if "shape" in info:
                shape_counter[(var_name, info["shape"], info["dtype"])] += 1
                print(
                    f"  {var_name}: shape={info['shape']}, dtype={info['dtype']}, "
                    f"min={info['min']:.4f}, max={info['max']:.4f}, mean={info['mean']:.4f}"
                )
            else:
                print(f"  {var_name}: {info}")

    print()
    print("=" * 80)
    print("FULL SHAPE SUMMARY")
    print("=" * 80)

    for path in mat_files:
        folder_counter[str(path.parent.relative_to(DATA_ROOT))] += 1

        try:
            variables = describe_mat_file(path)
        except Exception as e:
            print(f"ERROR reading {path.relative_to(DATA_ROOT)}: {e}")
            continue

        for var_name, info in variables.items():
            variable_counter[var_name] += 1
            if "shape" in info:
                shape_counter[(var_name, info["shape"], info["dtype"])] += 1

    print()
    print("Files per folder:")
    for folder, count in folder_counter.most_common():
        print(f"  {folder}: {count}")

    print()
    print("Variables found:")
    for var_name, count in variable_counter.most_common():
        print(f"  {var_name}: {count}")

    print()
    print("Shapes found:")
    for (var_name, shape, dtype), count in shape_counter.most_common():
        print(f"  {var_name}: shape={shape}, dtype={dtype} -> {count} files")


if __name__ == "__main__":
    main()