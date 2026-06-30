from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "meeting_room_full_windows_30.npz"

SPLIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "splits"
    / "meeting_room_full_train_val_test_split_seed42.npz"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = REPORT_DIR / "processed_dataset_audit.txt"
LABEL_CSV_PATH = REPORT_DIR / "processed_dataset_label_summary.csv"
SPLIT_CSV_PATH = REPORT_DIR / "processed_dataset_split_summary.csv"


# =============================================================================
# REPORT UTILS
# =============================================================================

report_lines: list[str] = []


def log(message: str = "") -> None:
    """Print a message and store it for the final text report."""

    print(message)
    report_lines.append(message)


def section(title: str) -> None:
    """Print a formatted section title."""

    log()
    log("=" * 90)
    log(title)
    log("=" * 90)


def subsection(title: str) -> None:
    """Print a formatted subsection title."""

    log()
    log("-" * 90)
    log(title)
    log("-" * 90)


def format_bytes(num_bytes: int) -> str:
    """Format a byte count into a readable string."""

    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TB"


def save_report() -> None:
    """Save the collected report lines to a text file."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    log()
    log(f"Report saved to: {REPORT_PATH}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_npz_file(path: Path) -> np.lib.npyio.NpzFile:
    """Load a .npz file after checking that it exists."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return np.load(path)


def get_first_existing_array(
    npz_file: np.lib.npyio.NpzFile,
    possible_keys: list[str],
    split_name: str,
) -> np.ndarray:
    """Return the first existing array among possible split key names."""

    for key in possible_keys:
        if key in npz_file:
            return npz_file[key]

    available_keys = list(npz_file.keys())

    raise KeyError(
        f"Could not find {split_name} indices. "
        f"Expected one of {possible_keys}, found {available_keys}."
    )


def load_split_indices(split_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train, validation and test indices from an existing split file."""

    split_data = load_npz_file(split_path)

    train_idx = get_first_existing_array(
        split_data,
        ["train_idx", "train_indices", "idx_train"],
        "train",
    )

    val_idx = get_first_existing_array(
        split_data,
        ["val_idx", "val_indices", "idx_val"],
        "validation",
    )

    test_idx = get_first_existing_array(
        split_data,
        ["test_idx", "test_indices", "idx_test"],
        "test",
    )

    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)


# =============================================================================
# ARRAY DESCRIPTION
# =============================================================================

def describe_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """Print all arrays stored inside the .npz file."""

    section("NPZ CONTENT")

    keys = list(data.keys())

    log(f"Dataset file: {DATASET_PATH}")
    log(f"Available arrays: {keys}")

    for key in keys:
        array = data[key]

        log()
        log(f"{key}")
        log(f"  shape: {array.shape}")
        log(f"  dtype: {array.dtype}")
        log(f"  memory: {format_bytes(array.nbytes)}")


def describe_numeric_array(name: str, array: np.ndarray) -> None:
    """Print descriptive statistics for a numeric array."""

    subsection(f"NUMERIC STATS: {name}")

    finite_mask = np.isfinite(array)
    num_values = array.size
    num_finite = int(finite_mask.sum())
    num_nan = int(np.isnan(array).sum())
    num_inf = int(np.isinf(array).sum())

    log(f"total values: {num_values}")
    log(f"finite values: {num_finite}")
    log(f"nan values: {num_nan}")
    log(f"inf values: {num_inf}")

    if num_finite == 0:
        log("No finite values available for statistics.")
        return

    finite_values = array[finite_mask]

    log(f"min: {finite_values.min():.6f}")
    log(f"max: {finite_values.max():.6f}")
    log(f"mean: {finite_values.mean():.6f}")
    log(f"std: {finite_values.std():.6f}")

    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    values = np.percentile(finite_values, percentiles)

    log("percentiles:")
    for percentile, value in zip(percentiles, values):
        log(f"  p{percentile:>3}: {value:.6f}")


def describe_channel_stats(x_windows: np.ndarray) -> None:
    """Print statistics for each CSI channel/antenna."""

    subsection("CHANNEL / ANTENNA STATS")

    if x_windows.ndim != 4:
        log(f"Expected x_windows with 4 dimensions, found shape {x_windows.shape}")
        return

    num_channels = x_windows.shape[1]

    for channel_id in range(num_channels):
        channel_values = x_windows[:, channel_id, :, :]

        log()
        log(f"channel {channel_id}")
        log(f"  shape: {channel_values.shape}")
        log(f"  min: {channel_values.min():.6f}")
        log(f"  max: {channel_values.max():.6f}")
        log(f"  mean: {channel_values.mean():.6f}")
        log(f"  std: {channel_values.std():.6f}")


# =============================================================================
# LABEL ANALYSIS
# =============================================================================

def analyze_labels(y_labels: np.ndarray) -> dict[int, dict[str, int]]:
    """Analyze class labels and return a per-label summary."""

    section("LABEL ANALYSIS")

    unique_labels, counts = np.unique(y_labels, return_counts=True)

    num_samples = len(y_labels)
    num_classes = len(unique_labels)

    log(f"num samples: {num_samples}")
    log(f"num classes: {num_classes}")
    log(f"min label: {unique_labels.min()}")
    log(f"max label: {unique_labels.max()}")

    expected_labels = np.arange(unique_labels.min(), unique_labels.max() + 1)
    labels_are_contiguous = np.array_equal(unique_labels, expected_labels)

    log(f"labels are contiguous integers: {labels_are_contiguous}")

    log()
    log("samples per class:")
    log(f"  min: {counts.min()}")
    log(f"  max: {counts.max()}")
    log(f"  mean: {counts.mean():.2f}")
    log(f"  std: {counts.std():.2f}")

    balanced = counts.min() == counts.max()
    log(f"balanced class counts: {balanced}")

    label_summary: dict[int, dict[str, int]] = {}

    for label, count in zip(unique_labels, counts):
        indices = np.where(y_labels == label)[0]

        label_summary[int(label)] = {
            "count": int(count),
            "first_index": int(indices[0]),
            "last_index": int(indices[-1]),
        }

    log()
    log("first 10 labels:")
    log(str(y_labels[:10].tolist()))

    log()
    log("last 10 labels:")
    log(str(y_labels[-10:].tolist()))

    return label_summary


def analyze_label_blocks(y_labels: np.ndarray) -> np.ndarray:
    """
    Analyze whether labels are stored in contiguous blocks.

    Also returns the window rank inside each label according to the current
    order in the processed dataset.
    """

    section("ORDER / BLOCK ANALYSIS")

    unique_labels = np.unique(y_labels)

    all_contiguous = True
    constant_block_size = True
    block_sizes: list[int] = []

    window_rank_within_label = np.zeros(len(y_labels), dtype=int)

    for label in unique_labels:
        indices = np.where(y_labels == label)[0]
        block_sizes.append(len(indices))

        expected_indices = np.arange(indices[0], indices[0] + len(indices))
        is_contiguous = np.array_equal(indices, expected_indices)

        if not is_contiguous:
            all_contiguous = False

        for rank, sample_index in enumerate(indices):
            window_rank_within_label[sample_index] = rank

    if len(set(block_sizes)) != 1:
        constant_block_size = False

    log(f"labels stored in contiguous blocks: {all_contiguous}")
    log(f"constant number of samples per label: {constant_block_size}")

    if block_sizes:
        log(f"block size min: {min(block_sizes)}")
        log(f"block size max: {max(block_sizes)}")
        log(f"block size mean: {np.mean(block_sizes):.2f}")

    log()
    log("Important interpretation:")
    log(
        "If labels are contiguous blocks and the preprocessing script generated "
        "windows sequentially from packet order without shuffling, then the rank "
        "inside each label can be interpreted as the acquisition-window order."
    )
    log(
        "This script can verify structural ordering in the processed .npz file, "
        "but it cannot prove temporal meaning unless the preprocessing code also "
        "preserved packet order."
    )

    return window_rank_within_label


# =============================================================================
# POSITION ANALYSIS
# =============================================================================

def analyze_positions(
    y_labels: np.ndarray,
    grid_positions: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Analyze grid positions and label-to-position consistency."""

    section("GRID POSITION ANALYSIS")

    if grid_positions.ndim != 2 or grid_positions.shape[1] != 2:
        log(f"Expected grid_positions shape [num_samples, 2], found {grid_positions.shape}")
        return {}

    unique_positions = np.unique(grid_positions, axis=0)

    log(f"grid_positions shape: {grid_positions.shape}")
    log(f"unique positions: {len(unique_positions)}")

    log()
    log("coordinate ranges:")
    log(f"  x min: {grid_positions[:, 0].min()}")
    log(f"  x max: {grid_positions[:, 0].max()}")
    log(f"  y min: {grid_positions[:, 1].min()}")
    log(f"  y max: {grid_positions[:, 1].max()}")

    unique_labels = np.unique(y_labels)

    label_to_position: dict[int, tuple[float, float]] = {}
    inconsistent_labels: list[int] = []

    for label in unique_labels:
        indices = np.where(y_labels == label)[0]
        positions_for_label = grid_positions[indices]
        unique_positions_for_label = np.unique(positions_for_label, axis=0)

        if len(unique_positions_for_label) != 1:
            inconsistent_labels.append(int(label))
        else:
            position = unique_positions_for_label[0]
            label_to_position[int(label)] = (float(position[0]), float(position[1]))

    log()
    log(f"labels with exactly one unique position: {len(label_to_position)}")
    log(f"labels with inconsistent positions: {len(inconsistent_labels)}")

    if inconsistent_labels:
        log(f"inconsistent labels: {inconsistent_labels[:20]}")
    else:
        log("Each label maps to exactly one spatial position.")

    log()
    log("first 10 label-position mappings:")
    for label in sorted(label_to_position.keys())[:10]:
        log(f"  label {label:03d}: {label_to_position[label]}")

    return label_to_position


# =============================================================================
# SPLIT ANALYSIS
# =============================================================================

def check_split_integrity(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    num_samples: int,
) -> None:
    """Check overlap and coverage of the split indices."""

    section("SPLIT INTEGRITY")

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    all_indices = train_set.union(val_set).union(test_set)

    log(f"train samples: {len(train_idx)}")
    log(f"validation samples: {len(val_idx)}")
    log(f"test samples: {len(test_idx)}")
    log(f"total split samples: {len(all_indices)}")
    log(f"dataset samples: {num_samples}")

    log()
    log(f"train/validation overlap: {len(train_val_overlap)}")
    log(f"train/test overlap: {len(train_test_overlap)}")
    log(f"validation/test overlap: {len(val_test_overlap)}")

    split_covers_dataset = len(all_indices) == num_samples
    log(f"split covers all dataset samples: {split_covers_dataset}")

    out_of_range = [
        index
        for index in all_indices
        if index < 0 or index >= num_samples
    ]

    log(f"indices out of range: {len(out_of_range)}")


def analyze_split_distribution(
    y_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    window_rank_within_label: np.ndarray,
) -> list[dict[str, int | float]]:
    """Analyze class distribution and window ranks inside each split."""

    section("SPLIT DISTRIBUTION BY CLASS")

    split_infos = [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]

    unique_labels = np.unique(y_labels)

    rows: list[dict[str, int | float]] = []

    for label in unique_labels:
        label = int(label)

        row: dict[str, int | float] = {"label": label}

        for split_name, split_idx in split_infos:
            split_labels = y_labels[split_idx]
            selected = split_idx[split_labels == label]

            row[f"{split_name}_count"] = int(len(selected))

            if len(selected) > 0:
                ranks = window_rank_within_label[selected]

                row[f"{split_name}_rank_min"] = int(ranks.min())
                row[f"{split_name}_rank_max"] = int(ranks.max())
                row[f"{split_name}_rank_mean"] = float(ranks.mean())
            else:
                row[f"{split_name}_rank_min"] = -1
                row[f"{split_name}_rank_max"] = -1
                row[f"{split_name}_rank_mean"] = -1.0

        rows.append(row)

    for split_name, split_idx in split_infos:
        counts = []

        for label in unique_labels:
            split_labels = y_labels[split_idx]
            counts.append(int((split_labels == label).sum()))

        counts_array = np.array(counts)

        log()
        log(f"{split_name}")
        log(f"  samples: {len(split_idx)}")
        log(f"  per-class min: {counts_array.min()}")
        log(f"  per-class max: {counts_array.max()}")
        log(f"  per-class mean: {counts_array.mean():.2f}")
        log(f"  per-class std: {counts_array.std():.2f}")
        log(f"  balanced per class: {counts_array.min() == counts_array.max()}")

    subsection("CURRENT SPLIT WINDOW-RANK INTERPRETATION")

    log(
        "The rank columns show which within-label window positions are used by "
        "each split. If train/validation/test ranks are mixed across the full "
        "0..49 range, the split is random-like with respect to window order."
    )
    log(
        "If train ranks are early, validation ranks are middle, and test ranks "
        "are late, the split is block-like with respect to window order."
    )

    for label in list(unique_labels[:5]):
        label = int(label)
        row = rows[label]

        log()
        log(f"label {label:03d}")
        log(
            f"  train count={row['train_count']}, "
            f"rank range=[{row['train_rank_min']}, {row['train_rank_max']}], "
            f"rank mean={row['train_rank_mean']:.2f}"
        )
        log(
            f"  val count={row['validation_count']}, "
            f"rank range=[{row['validation_rank_min']}, {row['validation_rank_max']}], "
            f"rank mean={row['validation_rank_mean']:.2f}"
        )
        log(
            f"  test count={row['test_count']}, "
            f"rank range=[{row['test_rank_min']}, {row['test_rank_max']}], "
            f"rank mean={row['test_rank_mean']:.2f}"
        )

    return rows


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_label_summary_csv(
    label_summary: dict[int, dict[str, int]],
    label_to_position: dict[int, tuple[float, float]],
) -> None:
    """Save per-label summary to CSV."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LABEL_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "label",
            "count",
            "first_index",
            "last_index",
            "grid_x",
            "grid_y",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for label in sorted(label_summary.keys()):
            position = label_to_position.get(label, (None, None))

            writer.writerow(
                {
                    "label": label,
                    "count": label_summary[label]["count"],
                    "first_index": label_summary[label]["first_index"],
                    "last_index": label_summary[label]["last_index"],
                    "grid_x": position[0],
                    "grid_y": position[1],
                }
            )

    log(f"Label summary CSV saved to: {LABEL_CSV_PATH}")


def save_split_summary_csv(rows: list[dict[str, int | float]]) -> None:
    """Save per-label split summary to CSV."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(SPLIT_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    log(f"Split summary CSV saved to: {SPLIT_CSV_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    section("PROCESSED DATASET AUDIT")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Dataset path: {DATASET_PATH}")
    log(f"Split path: {SPLIT_PATH}")

    data = load_npz_file(DATASET_PATH)

    describe_npz_keys(data)

    required_keys = ["x_windows", "y_labels", "grid_positions"]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key missing from dataset: {key}")

    x_windows = data["x_windows"]
    y_labels = data["y_labels"]
    grid_positions = data["grid_positions"]

    section("CORE SHAPES")
    log(f"x_windows shape: {x_windows.shape}")
    log(f"y_labels shape: {y_labels.shape}")
    log(f"grid_positions shape: {grid_positions.shape}")

    if x_windows.ndim == 4:
        num_samples, num_channels, height, width = x_windows.shape

        log()
        log("Interpreted x_windows dimensions:")
        log(f"  num_samples: {num_samples}")
        log(f"  num_channels / antennas: {num_channels}")
        log(f"  matrix height: {height}")
        log(f"  matrix width: {width}")
        log()
        log("For this project, each sample is interpreted as:")
        log("  [antennas, CSI matrix dimension 1, CSI matrix dimension 2]")
        log("  expected shape: [3, 30, 30]")

    describe_numeric_array("x_windows", x_windows)
    describe_channel_stats(x_windows)

    label_summary = analyze_labels(y_labels)
    window_rank_within_label = analyze_label_blocks(y_labels)

    label_to_position = analyze_positions(
        y_labels=y_labels,
        grid_positions=grid_positions,
    )

    save_label_summary_csv(
        label_summary=label_summary,
        label_to_position=label_to_position,
    )

    if SPLIT_PATH.exists():
        train_idx, val_idx, test_idx = load_split_indices(SPLIT_PATH)

        check_split_integrity(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_samples=len(y_labels),
        )

        split_rows = analyze_split_distribution(
            y_labels=y_labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            window_rank_within_label=window_rank_within_label,
        )

        save_split_summary_csv(split_rows)
    else:
        section("SPLIT ANALYSIS")
        log("Split file not found. Split analysis skipped.")

    section("FINAL INTERPRETATION CHECKLIST")

    log("1. If x_windows has shape [8800, 3, 30, 30], the processed CSI windows are ready.")
    log("2. If y_labels has 176 balanced classes, each reference point has the same number of windows.")
    log("3. If each label maps to exactly one grid position, classification errors can be converted into spatial errors.")
    log("4. If labels are stored in contiguous blocks of 50, the processed file likely preserves reference-point grouping.")
    log("5. If preprocessing generated windows sequentially and did not shuffle them, the within-label rank can represent acquisition-window order.")
    log("6. If the existing split has mixed within-label ranks, it is random-like with respect to acquisition-window order.")
    log("7. An ordered-window split is valid only if preprocessing preserved packet/window order.")

    save_report()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)