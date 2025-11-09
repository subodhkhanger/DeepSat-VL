import yaml
from pathlib import Path


def load_config(path: str | Path) -> dict:
    p = Path(path)
    with p.open('r') as f:
        cfg = yaml.safe_load(f)
    return cfg

