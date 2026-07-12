import pytest
import tempfile
import pathlib
import os

def test_pipeline_import():
    try:
        from pipelines import train, evaluate, detect
    except Exception as e:
        pytest.fail(f"Could not import pipelines: {e}")

def test_main_cli_help():
    assert os.system("python main.py -h") == 0

def test_main_cli_train_help():
    assert os.system("python main.py train -h") == 0

def test_main_cli_detect_help():
    assert os.system("python main.py detect -h") == 0

def test_main_cli_evaluate_help():
    assert os.system("python main.py evaluate -h") == 0
