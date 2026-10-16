from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_configs(config_dir: str = "configs") -> Dict[str, Dict[str, Any]]:
    base = Path(config_dir)
    cfg = {}
    for name in ["config", "model", "streaming", "monitoring"]:
        fp = base / f"{name}.yaml"
        cfg[name] = load_yaml(fp) if fp.exists() else {}
    return cfg