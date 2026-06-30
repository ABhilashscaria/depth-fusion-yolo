import yaml
import os
from pathlib import Path

def load_config(config_path: str = "config/default.yaml") -> dict:
    """
    Loads the YAML configuration file and returns it as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config
