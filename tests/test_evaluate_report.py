import pytest
import tempfile
import pathlib
from deepguard.evaluate import generate_report

def test_generate_report():
    metrics = {"precision": 0.5, "recall": 0.5, "f1": 0.5, "auc_roc": 0.5, "accuracy": 0.5, "tp": 1, "fp": 1, "tn": 1, "fn": 1, "fpr": 0.5, "fnr": 0.5}
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "report.txt"
        generate_report("TestModel", metrics, p)
        assert p.exists()
        content = p.read_text()
        assert "EVALUATION REPORT — TestModel" in content
        assert "F1-Score    : 0.5000" in content
