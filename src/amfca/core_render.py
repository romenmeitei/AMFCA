from __future__ import annotations
from pathlib import Path
import ast


def render_assignments(template_path, destination, overrides: dict):
    """Patch only module-level simple assignments in the frozen core script.

    This intentionally refuses to patch names that are not found exactly once at module level.
    """
    src=Path(template_path).read_text(encoding="utf-8")
    tree=ast.parse(src)
    spans={}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            spans.setdefault(n.targets[0].id,[]).append((n.lineno,n.end_lineno))
    missing=[k for k in overrides if len(spans.get(k,[]))!=1]
    if missing: raise ValueError(f"Cannot safely patch assignments: {missing}")
    lines=src.splitlines()
    replacements=[]
    for name,value in overrides.items():
        start,end=spans[name][0]
        replacements.append((start-1,end,f"{name} = {repr(value)}"))
    for start,end,text in sorted(replacements,reverse=True):
        lines[start:end]=[text]
    out=Path(destination); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(lines)+"\n",encoding="utf-8")
    ast.parse(out.read_text(encoding="utf-8"))
    return out
