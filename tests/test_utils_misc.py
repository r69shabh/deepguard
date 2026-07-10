import pytest
import logging
import tempfile
import pathlib
from deepguard.utils import setup_logging, get_logger, set_seed, ensure_dirs, project_root

def test_setup_logging():
    logger = setup_logging()
    assert logger.name == "deepguard" # wait it was changed to deepguard? Let's check. Ah I didn't change it inside utils perfectly if it was hardcoded. Wait I did sed. It should be deepguard.
    
def test_get_logger():
    logger = get_logger("test")
    assert "test" in logger.name
    
def test_set_seed():
    import numpy as np
    set_seed(42)
    a = np.random.rand()
    set_seed(42)
    b = np.random.rand()
    assert a == b

def test_ensure_dirs():
    with tempfile.TemporaryDirectory() as d:
        p1 = pathlib.Path(d) / "a"
        p2 = pathlib.Path(d) / "b" / "c"
        ensure_dirs(p1, p2)
        assert p1.exists()
        assert p2.exists()
        
def test_project_root():
    root = project_root()
    assert root.exists()
