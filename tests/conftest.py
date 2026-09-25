"""Shared pytest configuration for the reproducibility tests.

Puts the repository root, ``src/`` and ``video/`` on the import path and
forces a headless matplotlib backend (the figure scripts call
``plt.show()``).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "video"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
