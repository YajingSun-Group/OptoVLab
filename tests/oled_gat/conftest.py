from __future__ import annotations

import sys
from pathlib import Path


ANALYSIS_PROJECT = Path(__file__).resolve().parents[2] / "analysis" / "oled_gat"
sys.path.insert(0, str(ANALYSIS_PROJECT))
