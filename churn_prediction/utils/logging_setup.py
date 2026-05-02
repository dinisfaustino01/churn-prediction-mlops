import logging
import logging.config

import yaml

from churn_prediction import PROJECT_ROOT


def setup_logging() -> None:
    """
    Load logging configuration from config/logging.yaml and apply it globally.

    Call this once at program start. After this, every module can use:
        logger = logging.getLogger(__name__)
    """

    config_path = PROJECT_ROOT / "config" / "logging.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
