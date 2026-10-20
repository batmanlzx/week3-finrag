"""Make the bundled runtime libraries importable regardless of CWD."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for extra in (ROOT / ".pylibs", ROOT / ".vendor" / "jieba_pkg", ROOT):
    p = str(extra)
    if p not in sys.path:
        sys.path.insert(0, p)
