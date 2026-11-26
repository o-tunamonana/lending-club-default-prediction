"""Kept so the INSY 674 notebooks still run: `from preprocessing import load_split, build_preprocessor`.

The single source of truth is now src/lcdp/preprocessing.py, shared by the
training flow, the API and the notebooks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcdp.preprocessing import *  # noqa: E402, F401, F403
