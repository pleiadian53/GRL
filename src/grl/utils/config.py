"""
Configuration utilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.
    
    Args:
        base: Base configuration
        override: Override values
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
