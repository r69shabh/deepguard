import os
import subprocess
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _cli_help(*args: str) -> None:
    res = subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, f"CLI failed: {res.stderr}"


def test_pipeline_import():
    try:
        from pipelines import train, evaluate, detect, preprocess  # noqa: F401
    except Exception as e:
        pytest.fail(f"Could not import pipelines: {e}")


def test_main_cli_help():
    _cli_help("-h")


def test_main_cli_preprocess_help():
    _cli_help("preprocess", "-h")


def test_main_cli_train_help():
    _cli_help("train", "-h")


def test_main_cli_detect_help():
    _cli_help("detect", "-h")


def test_main_cli_evaluate_help():
    _cli_help("evaluate", "-h")
