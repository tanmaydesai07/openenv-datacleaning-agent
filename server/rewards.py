"""
Reward computation for the Data Cleaning environment.

Compares submitted cleaned file against expected output using multiple metrics.
"""

import csv
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
# Use a larger epsilon so serialized/rounded scores still stay strictly inside (0, 1).
MIN_SCORE = 1e-4
MAX_SCORE = 1.0 - 1e-4


def load_csv(file_path: str) -> Tuple[list, list]:
    """
    Load CSV file and return headers and rows.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (headers, rows)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
        return headers, rows
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        return [], []


def normalize_value(value: str) -> str:
    """Normalize a value for comparison (strip whitespace, lowercase)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def compute_row_score(submitted_rows: list, expected_rows: list) -> float:
    """
    Compute row-level score (# of correct rows / # expected rows).

    Args:
        submitted_rows: List of rows from submitted file
        expected_rows: List of rows from expected file

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected_rows:
        return 1.0 if not submitted_rows else 0.0

    # Create normalized row tuples for comparison
    expected_set = {
        tuple(normalize_value(row.get(k, "")) for k in sorted(row.keys()))
        for row in expected_rows
    }
    submitted_set = {
        tuple(normalize_value(row.get(k, "")) for k in sorted(row.keys()))
        for row in submitted_rows
    }

    # Count matching rows
    matches = len(expected_set & submitted_set)
    return matches / len(expected_rows)


def compute_column_score(submitted_headers: list, expected_headers: list) -> float:
    """
    Compute column-level score (correct columns / expected columns).

    Args:
        submitted_headers: List of headers from submitted file
        expected_headers: List of headers from expected file

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected_headers:
        return 1.0 if not submitted_headers else 0.0

    expected_set = {normalize_value(h) for h in expected_headers}
    submitted_set = {normalize_value(h) for h in submitted_headers}

    matches = len(expected_set & submitted_set)
    return matches / len(expected_headers)


def compute_cell_accuracy(submitted_rows: list, expected_rows: list) -> float:
    """
    Compute cell-level accuracy (correct cells / total cells).

    Compares only rows with matching keys (e.g., same ID).

    Args:
        submitted_rows: List of rows from submitted file
        expected_rows: List of rows from expected file

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected_rows:
        return 1.0 if not submitted_rows else 0.0

    total_cells = 0
    correct_cells = 0

    # Build a map of expected rows by their first column value (assumed to be ID)
    expected_map = {}
    for row in expected_rows:
        if not row:
            continue
        key = list(row.values())[0]  # Use first column as key
        expected_map[normalize_value(key)] = row

    for sub_row in submitted_rows:
        if not sub_row:
            continue
        key = list(sub_row.values())[0]
        norm_key = normalize_value(key)

        if norm_key not in expected_map:
            continue  # Skip rows not in expected

        exp_row = expected_map[norm_key]

        # Compare each cell
        for col in exp_row.keys():
            total_cells += 1
            exp_val = normalize_value(exp_row.get(col, ""))
            sub_val = normalize_value(sub_row.get(col, ""))

            if exp_val == sub_val:
                correct_cells += 1

    if total_cells == 0:
        return 0.0

    return correct_cells / total_cells


def compute_reward(
    submitted_file_path: Optional[str],
    expected_file_path: str,
    row_weight: float = 0.2,
    column_weight: float = 0.2,
    cell_weight: float = 0.6,
) -> float:
    """
    Compute reward based on submitted file vs expected file.

    Reward = row_weight * row_score + column_weight * column_score + cell_weight * cell_accuracy

    Args:
        submitted_file_path: Path to submitted cleaned file (None if not submitted)
        expected_file_path: Path to expected cleaned file
        row_weight: Weight for row-level score (default: 0.2)
        column_weight: Weight for column-level score (default: 0.2)
        cell_weight: Weight for cell-level accuracy (default: 0.6)

    Returns:
        Reward between 0.0 and 1.0
    """
    if not submitted_file_path:
        logger.info("No file submitted, returning minimum bounded reward")
        return MIN_SCORE

    # Load files
    expected_headers, expected_rows = load_csv(expected_file_path)
    submitted_headers, submitted_rows = load_csv(submitted_file_path)

    if not expected_headers or not expected_rows:
        logger.error(f"Failed to load expected file: {expected_file_path}")
        return MIN_SCORE

    if not submitted_headers or not submitted_rows:
        logger.warning(
            f"Failed to load submitted file or file is empty: {submitted_file_path}"
        )
        return MIN_SCORE

    # Compute component scores
    row_score = compute_row_score(submitted_rows, expected_rows)
    column_score = compute_column_score(submitted_headers, expected_headers)
    cell_accuracy = compute_cell_accuracy(submitted_rows, expected_rows)

    # Weighted combination
    reward = (
        row_weight * row_score
        + column_weight * column_score
        + cell_weight * cell_accuracy
    )

    bounded_reward = max(MIN_SCORE, min(MAX_SCORE, reward))

    logger.debug(
        f"Reward computation metrics: row={row_score:.6f}, col={column_score:.6f}, "
        f"cell={cell_accuracy:.6f}, raw_value={reward:.6f}, bounded_value={bounded_reward:.6f}"
    )

    return bounded_reward
