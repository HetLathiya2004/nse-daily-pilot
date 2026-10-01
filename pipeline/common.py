"""Shared helpers: repo paths, config loading, logging, registry IO."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
SITE_DATA_DIR = ROOT / "site" / "data"
REGISTRY_PATH = ROOT / "model_registry.json"


def setup_logging(name: str = "pilot") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config() -> dict:
    return load_yaml(CONFIG_DIR / "pipeline.yaml")


def load_universe() -> list:
    cfg = load_yaml(CONFIG_DIR / "universe.yaml")
    syms = cfg.get("symbols") or []
    return [s.strip().upper() for s in syms if s and s.strip()]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"champion": None, "runs": []}


def save_registry(reg: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)
