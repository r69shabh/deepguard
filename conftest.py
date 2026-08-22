"""
Pytest configuration: make the project root importable so tests can
`import deepguard` / `from pipelines import ...` regardless of how pytest
is invoked (plain `pytest` does not add the CWD to sys.path).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
