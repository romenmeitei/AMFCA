#!/usr/bin/env python
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from amfca.config import load_config, validate_config
from amfca.pipeline import plan, run_config

p=argparse.ArgumentParser(description="AMFCA reproducibility runner")
p.add_argument("--config",required=True)
p.add_argument("--stages",nargs="*",default=None,help="Subset of Lhende stages in dependency order")
p.add_argument("--plan",action="store_true")
a=p.parse_args()
cfg=load_config(a.config); errs=validate_config(cfg)
if errs:
    print("CONFIG INVALID")
    for e in errs: print(" -",e)
    raise SystemExit(2)
if a.plan:
    print(json.dumps({"profile":cfg["project"].get("profile","lhende_exact"),"stages":plan(cfg)},indent=2)); raise SystemExit(0)
print(json.dumps(run_config(a.config,ROOT,stages=a.stages),indent=2,default=str))
