from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import yaml

REQUIRED_TOP_LEVEL = {"project", "inputs", "outputs", "analysis"}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    missing = REQUIRED_TOP_LEVEL - set(cfg)
    if missing:
        raise ValueError(f"Configuration missing top-level keys: {sorted(missing)}")
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def deep_get(mapping: dict, dotted: str, default=None):
    cur = mapping
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def validate_config(cfg: dict) -> list[str]:
    errors: list[str] = []
    project = cfg.get("project", {})
    if not project.get("study_id"):
        errors.append("project.study_id is required")
    if not project.get("event_time_utc"):
        errors.append("project.event_time_utc is required")
    analysis = cfg.get("analysis", {})
    if float(analysis.get("analysis_scale_m", 0)) <= 0:
        errors.append("analysis.analysis_scale_m must be > 0")
    thresholds = analysis.get("rms_thresholds", [])
    if not thresholds or float(analysis.get("primary_rms_threshold", -1)) not in [float(x) for x in thresholds]:
        errors.append("analysis.primary_rms_threshold must occur in analysis.rms_thresholds")
    return errors
