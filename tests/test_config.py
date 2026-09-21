from pathlib import Path
from amfca.config import load_config, validate_config
ROOT=Path(__file__).resolve().parents[1]

def test_configs_validate():
    for name in ["lhende_2026.yaml","template_new_event.yaml"]:
        cfg=load_config(ROOT/"configs"/name)
        assert validate_config(cfg)==[]
