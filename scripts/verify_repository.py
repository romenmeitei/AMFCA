#!/usr/bin/env python
from pathlib import Path
import ast, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from amfca.config import load_config, validate_config

checks=[]
def check(name, ok, detail=""):
    checks.append((name,bool(ok),detail)); print(f"{name:32} {'PASS' if ok else 'FAIL'} {detail}")

required=["README.md","pyproject.toml","configs/lhende_2026.yaml","notebooks/AMFCA_Master_Reproducibility_Colab.ipynb","scripts/run_pipeline.py"]
check("Required repository files",all((ROOT/x).exists() for x in required))
try:
    cfg=load_config(ROOT/"configs/lhende_2026.yaml"); errs=validate_config(cfg); check("Lhende config",not errs,"; ".join(errs))
except Exception as e: check("Lhende config",False,str(e))
try:
    cfg=load_config(ROOT/"configs/template_new_event.yaml"); errs=validate_config(cfg); check("Generic template config",not errs,"; ".join(errs))
except Exception as e: check("Generic template config",False,str(e))
legacy=list((ROOT/"workflows/frozen_lhende_2026").glob("*.py")); syntax_ok=True
for p in legacy:
    try: ast.parse(p.read_text(encoding="utf-8"))
    except Exception: syntax_ok=False
check("Frozen workflow syntax",syntax_ok,f"{len(legacy)} scripts")
manifest=json.loads((ROOT/"validation/frozen_script_manifest.json").read_text())
hash_ok=True
for row in manifest:
    p=ROOT/row["path"]
    h=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
    hash_ok &= h==row["sha256"]
check("Frozen workflow checksums",hash_ok,f"{len(manifest)} files")
try:
    r=subprocess.run([sys.executable,"-m","pytest","-q","-p","no:cacheprovider",str(ROOT/"tests")],cwd=ROOT,text=True,capture_output=True)
    check("Offline pytest suite",r.returncode==0,r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-120:])
except Exception as e: check("Offline pytest suite",False,str(e))
print("\nOVERALL:","PASS" if all(x[1] for x in checks) else "FAIL")
raise SystemExit(0 if all(x[1] for x in checks) else 1)
