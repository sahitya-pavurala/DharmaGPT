from __future__ import annotations

import sys
from pathlib import Path

DHARMAGPT_DIR = Path(__file__).resolve().parents[2] / "dharmagpt"
if str(DHARMAGPT_DIR) not in sys.path:
    sys.path.insert(0, str(DHARMAGPT_DIR))

