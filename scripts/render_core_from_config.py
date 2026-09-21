#!/usr/bin/env python
from pathlib import Path
import argparse, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from amfca.config import load_config
from amfca.core_render import render_assignments

p=argparse.ArgumentParser(description="Render a configured copy of the frozen Earth Engine core workflow")
p.add_argument("--config",required=True); p.add_argument("--output",required=True)
a=p.parse_args(); cfg=load_config(a.config)
template=ROOT/"workflows/frozen_lhende_2026/01_core_gee_lhende_2026.py"
overrides=cfg.get("core_gee_overrides",{})
if not overrides: raise SystemExit("No core_gee_overrides found in config")
out=render_assignments(template,a.output,overrides)
print(out)
