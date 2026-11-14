"""Poll the public score for one Kaggle competition submission.

Reads KAGGLE_API_TOKEN from the repository .env file when it is not already
present in the process environment. The token is passed to the Kaggle CLI but
is never printed.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
CLI = ROOT / ".work" / "tools" / "kaggle-cli-venv" / "Scripts" / "kaggle.exe"
COMPETITION = "filament-segmentation-2026"
TARGET_REF = "58326011"
INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))
MAX_CHECKS = int(os.environ.get("MAX_CHECKS", "48"))


def load_env(path=ENV_FILE):
    if not path.is_file():
        sys.exit(".env not found; set KAGGLE_API_TOKEN in the process environment.")
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '\"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def fetch_submissions():
    command = [
        str(CLI),
        "competitions",
        "submissions",
        COMPETITION,
        "--format",
        "json",
        "--page-size",
        "200",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # Avoid echoing CLI diagnostics that could contain sensitive context.
        raise RuntimeError(f"Kaggle CLI failed with exit code {result.returncode}.")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Kaggle CLI returned invalid JSON.") from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("submissions", "data", "results"):
            if isinstance(data.get(key), list):
                return data[key]
        if "ref" in data:
            return [data]
    raise RuntimeError("Kaggle CLI returned an unexpected submissions format.")


load_env()
if not os.environ.get("KAGGLE_API_TOKEN"):
    sys.exit("KAGGLE_API_TOKEN is missing from .env or the process environment.")
if not CLI.is_file():
    sys.exit(f"Kaggle CLI not found at {CLI}; install it in .work/tools first.")
if INTERVAL < 1 or MAX_CHECKS < 1:
    sys.exit("CHECK_INTERVAL and MAX_CHECKS must be positive integers.")

print(
    f"Checking {COMPETITION} every {INTERVAL}s (max {MAX_CHECKS} checks); "
    f"target submission {TARGET_REF}. Ctrl+C to stop.\n"
)

for check_number in range(1, MAX_CHECKS + 1):
    submissions = fetch_submissions()
    print(f"[check {check_number}/{MAX_CHECKS}]")
    for submission in submissions[:5]:
        print(
            "  ref={ref} status={status} publicScore={score} date={date}".format(
                ref=submission.get("ref", "?"),
                status=submission.get("status", "?"),
                score=submission.get("publicScore", "pending"),
                date=submission.get("date", ""),
            )
        )

    target = next(
        (row for row in submissions if str(row.get("ref", "")) == TARGET_REF),
        None,
    )
    if target is not None:
        score = target.get("publicScore")
        if score is not None and str(score).strip() not in ("", "pending", "None"):
            print(f"\nSubmission {TARGET_REF} status: {target.get('status', 'unknown')}")
            print(f"Public score: {score}")
            break
    if check_number < MAX_CHECKS:
        print(f"  (waiting {INTERVAL}s...)\n")
        time.sleep(INTERVAL)
