import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def get_column_lists(
    config_path: str | Path,
) -> tuple[list[str], list[str], list[str], str]:
    """Load feature schema from YAML and return the column groupings.

    Args:
        config_path: Path to the YAML file containing the feature schema.

    Returns:
        Tuple of (numeric_cols, categorical_cols, cols_to_drop, target_col).
    """

    logger.debug("Loading feature schema from %s", config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    numeric_cols = config["numeric"]
    categorical_cols = config["categorical"]
    cols_to_drop = config["drop"]
    target_col = config["target"]

    logger.debug(
        "Schema loaded: %d numeric, %d categorical, %d dropped, target=%s",
        len(numeric_cols),
        len(categorical_cols),
        len(cols_to_drop),
        target_col,
    )

    return numeric_cols, categorical_cols, cols_to_drop, target_col