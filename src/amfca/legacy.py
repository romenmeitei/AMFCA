from __future__ import annotations
from pathlib import Path
import importlib.util

STAGE_FILES = {
    "m01_concurrence": "02_m01_concurrence_quantification.py",
    "background_comparison": "03_m01_background_comparison.py",
    "patch_inventory": "04_candidate_patch_inventory.py",
    "s2_recovery": "05_recover_s2_composites.py",
    "directional": "06_patch_directional_characterization.py",
    "dated_scene_audit": "07_dated_scene_audit.py",
    "inventory_reconciliation": "08_scene_inventory_reconciliation.py",
    "top20_review": "09_top20_patch_review.py",
    "external_corroboration": "10_external_corroboration.py",
}


def frozen_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / "workflows" / "frozen_lhende_2026"


def load_stage(stage: str, repo_root: str | Path = "."):
    if stage not in STAGE_FILES:
        raise KeyError(f"Unknown frozen stage: {stage}")
    path = frozen_dir(repo_root) / STAGE_FILES[stage]
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f"amfca_frozen_{stage}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path
