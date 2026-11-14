from pathlib import Path
import json, os, subprocess, sys

ROOT = Path(__file__).resolve().parent
kaggle_dir = Path.home() / ".kaggle"
cli = str(ROOT / ".work/tools/kaggle-cli-venv/Scripts/kaggle.exe")
env = {**os.environ, "KAGGLE_CONFIG_DIR": str(kaggle_dir)}

r = subprocess.run(
    [cli, "competitions", "submissions", "filament-segmentation-2026",
     "--format", "json", "--page-size", "20"],
    capture_output=True, text=True, env=env
)
print("exit:", r.returncode)
if r.returncode != 0:
    print("stderr:", r.stderr[:400])
    sys.exit(1)

subs = json.loads(r.stdout)
if not isinstance(subs, list):
    print("unexpected:", str(subs)[:400])
    sys.exit(1)

for s in subs:
    print(json.dumps({
        k: s.get(k) for k in
        ("ref", "status", "publicScore", "errorDescription", "date", "fileName")
    }, indent=2))
