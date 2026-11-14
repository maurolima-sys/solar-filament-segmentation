"""Check Kaggle submission status. Reads credentials from .env."""
import os, time, sys
from pathlib import Path

def load_env(path=".env"):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

load_env()

# kaggle module reads KAGGLE_USERNAME + KAGGLE_KEY from env
key = os.environ.get("KAGGLE_API_TOKEN", "")
if not key:
    sys.exit("KAGGLE_API_TOKEN not found in .env")

# kaggle SDK expects KAGGLE_KEY
os.environ["KAGGLE_KEY"] = key
if "KAGGLE_USERNAME" not in os.environ:
    # derive from token prefix or prompt
    os.environ.setdefault("KAGGLE_USERNAME", "maurofranklimadelima")

from kaggle.api.kaggle_api_extended import KaggleApiExtended

api = KaggleApiExtended()
api.authenticate()

COMPETITION = "filament-segmentation-2026"
INTERVAL    = int(os.environ.get("CHECK_INTERVAL", "300"))   # seconds, default 5 min
MAX_CHECKS  = int(os.environ.get("MAX_CHECKS", "48"))        # 48 × 5 min = 4 h

print(f"Monitoring {COMPETITION} every {INTERVAL}s (max {MAX_CHECKS} checks). Ctrl+C to stop.\n")

for i in range(1, MAX_CHECKS + 1):
    subs = api.competitions_submissions_list(COMPETITION)
    print(f"[check {i}/{MAX_CHECKS}]")
    for s in subs[:5]:
        score = getattr(s, "publicScore", None) or getattr(s, "public_score", "pending")
        ref   = getattr(s, "ref", getattr(s, "id", "?"))
        desc  = getattr(s, "description", getattr(s, "submittedBy", ""))
        print(f"  ref={ref}  score={score}  desc={desc}")
    # stop early if latest submission has a score
    latest_score = getattr(subs[0], "publicScore", None) if subs else None
    if latest_score and latest_score != "None":
        print(f"\nScore available: {latest_score}")
        break
    if i < MAX_CHECKS:
        print(f"  (waiting {INTERVAL}s...)\n")
        time.sleep(INTERVAL)
