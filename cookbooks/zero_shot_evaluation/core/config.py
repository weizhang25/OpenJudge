# -*- coding: utf-8 -*-
"""Configuration loading and parsing for zero-shot evaluation."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from loguru import logger

from cookbooks.zero_shot_evaluation.core.schema import ZeroShotConfig


def resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values.

    Supports ${VAR_NAME} format.

    Args:
        value: Configuration value (can be str, dict, or list)

    Returns:
        Value with environment variables resolved
    """
    if isinstance(value, str):
        pattern = r"\$\{(\w+)\}"
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.getenv(var_name, "")
            if not env_value:
                logger.warning(f"Environment variable {var_name} not set")
            value = value.replace(f"${{{var_name}}}", env_value)
        return value
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: Union[str, Path]) -> ZeroShotConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated ZeroShotConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Resolve environment variables
    resolved_config = resolve_env_vars(raw_config)

    # Validate and create config object
    config = ZeroShotConfig(**resolved_config)
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Task: {config.task.description}")
    logger.info(f"Target endpoints: {list(config.target_endpoints.keys())}")

    return config


def config_to_dict(config: ZeroShotConfig) -> Dict[str, Any]:
    """Convert ZeroShotConfig to dictionary (for serialization).

    Args:
        config: ZeroShotConfig object

    Returns:
        Dictionary representation
    """
    return config.model_dump()

