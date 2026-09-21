from pathlib import Path
from amfca.core_render import render_assignments

def test_render_assignments(tmp_path):
    src=tmp_path/"x.py"; src.write_text("A = 1\nB = [1,2]\n")
    out=render_assignments(src,tmp_path/"y.py",{"A":2,"B":[3,4]})
    ns={}; exec(out.read_text(),ns)
    assert ns["A"]==2 and ns["B"]==[3,4]
