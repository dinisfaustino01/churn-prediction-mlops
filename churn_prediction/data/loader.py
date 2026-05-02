import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load a CSV file from disk into a pandas DataFrame.

    Dataset-agnostic: performs no column-specific transformations. Any cleaning
    or type coercion belongs in the preprocessing layer.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame with the raw contents of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    return df