import json
from pathlib import Path

# Path to the JSON configuration
_CONFIG_DIR = Path(__file__).resolve().parent
_JSON_PATH = _CONFIG_DIR / "time_metadata.json"

if _JSON_PATH.exists():
    with open(_JSON_PATH, "r", encoding="utf-8") as _f:
        _DATA = json.load(_f)
else:
    _DATA = {}

BASELINE_START = _DATA.get("BASELINE_START", "2025-08-01")
BASELINE_END = _DATA.get("BASELINE_END", "2025-08-31")
PRE_START = _DATA.get("PRE_START", "2026-08-01")
PRE_END = _DATA.get("PRE_END", "2026-08-25")
POST_START = _DATA.get("POST_START", "2026-08-27")
POST_END = _DATA.get("POST_END", "2026-09-05")
RECOVERY_START = _DATA.get("RECOVERY_START", "2026-09-06")
RECOVERY_END = _DATA.get("RECOVERY_END", "2026-09-30")

TIME_WINDOWS = {
    "baseline": (BASELINE_START, BASELINE_END),
    "pre": (PRE_START, PRE_END),
    "post": (POST_START, POST_END),
    "recovery": (RECOVERY_START, RECOVERY_END),
}
