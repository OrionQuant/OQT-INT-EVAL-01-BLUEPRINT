import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
_DEPS = _ROOT / "_deps"
if _DEPS.exists():
    sys.path.insert(0, str(_DEPS))
