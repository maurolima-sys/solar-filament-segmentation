"""Setup Kaggle credentials from .env and verify authentication."""
from pathlib import Path
import json, os, subprocess, sys

ROOT = Path(__file__).resolve().parent

def load_env():
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            env[k.strip()] = v.strip()
    return env

def setup_kaggle(env):
    token = env.get("KAGGLE_API_TOKEN", "")
    if not token:
        sys.exit("KAGGLE_API_TOKEN not found in .env")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kj = kaggle_dir / "kaggle.json"
    kj.write_text(json.dumps({"username": "maurofranklimadelima", "key": token}))
    kj.chmod(0o600)
    print(f"kaggle.json written ({len(token)} char token)")
    return kaggle_dir

def find_cli(root):
    candidates = [
        root / ".work/tools/kaggle-cli-venv/Scripts/kaggle.exe",
        Path(os.environ.get("VIRTUAL_ENV", "")) / "Scripts/kaggle.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "kaggle"  # fallback to PATH

def test_auth(cli, kaggle_dir):
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(kaggle_dir)}
    r = subprocess.run(
        [cli, "competitions", "submissions", "filament-segmentation-2026",
         "--format", "json", "--page-size", "5"],
        capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        print("Auth failed:", r.stderr[:200])
        return False
    try:
        subs = json.loads(r.stdout)
        if isinstance(subs, list):
            print(f"Auth OK — {len(subs)} recent submissions")
            for s in subs[:3]:
                ref   = s.get("ref", "?")
                score = s.get("publicScore", "pending")
                status= s.get("status", "?")
                print(f"  ref={ref}  status={status}  score={score}")
        else:
            print("Auth OK — response:", str(subs)[:200])
    except json.JSONDecodeError:
        print("Auth OK but unexpected response:", r.stdout[:200])
    return True

if __name__ == "__main__":
    env = load_env()
    kaggle_dir = setup_kaggle(env)
    cli = find_cli(ROOT)
    print(f"Using CLI: {cli}")
    test_auth(cli, kaggle_dir)
