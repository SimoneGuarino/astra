import json
from pathlib import Path
from typing import Any, Dict


def _deep_get(data: Dict[str, Any], path: str, default: Any) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_runtime_config() -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "runtime_config.json"

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


_RUNTIME_CONFIG = load_runtime_config()


def get_config(path: str, default: Any) -> Any:
    return _deep_get(_RUNTIME_CONFIG, path, default)