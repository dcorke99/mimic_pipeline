"""Run the scripts selected in pipeline_config.py."""

from pathlib import Path
import subprocess
import sys

import pipeline_config as config

REPO_ROOT = Path(__file__).resolve().parent


def main():
    for script in config.PIPELINE_STAGES:
        print(f"[PIPELINE] Starting {script}", flush=True)
        result = subprocess.run([sys.executable, str(REPO_ROOT / script)], cwd=REPO_ROOT)
        if result.returncode:
            print(f"{script} failed. Pipeline stopped.", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    main()
