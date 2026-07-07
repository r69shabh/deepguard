import pytest
import yaml
import tempfile
import pathlib
from deepguard.utils import load_config, get_nested

def test_load_config():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "cfg.yaml"
        with open(p, "w") as f:
            f.write("a: 1\nb:\n  c: 2")
        
        cfg = load_config(p)
        assert cfg["a"] == 1
        assert cfg["b"]["c"] == 2

def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist_404.yaml")
        
def test_get_nested():
    cfg = {"a": {"b": {"c": 3}}}
    assert get_nested(cfg, "a", "b", "c") == 3
    assert get_nested(cfg, "a", "b", "d") is None
    assert get_nested(cfg, "a", "b", "d", default=5) == 5
    assert get_nested(cfg, "x", default=10) == 10
