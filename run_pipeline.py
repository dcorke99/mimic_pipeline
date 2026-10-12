"""Run the stages selected in pipeline_config.py with the active Python."""

import argparse
import subprocess
import sys

import pipeline_config as run_config
from pipeline_runtime import REPO_ROOT, get_artefact_root, get_data_dir, get_validation_dir, validate_run_choices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show selected stages without running them")
    args = parser.parse_args(argv)
    # Validate all choices before any stage can write output.
    validate_run_choices()
    print(f"[CONFIG] {REPO_ROOT / 'pipeline_config.py'}", flush=True)
    print(f"[CONFIG] data: {get_data_dir()}", flush=True)
    print(f"[CONFIG] artefacts: {get_artefact_root()}", flush=True)
    print(f"[CONFIG] semi-synthetic panels: {get_validation_dir()}", flush=True)
    print(f"[CONFIG] fit models: {run_config.MODEL_TYPES}; learning curves: {run_config.RUN_LEARNING_CURVES}", flush=True)
    for stage in run_config.PIPELINE_STAGES:
        print(f"[PIPELINE] {'Selected' if args.dry_run else 'Starting'} {stage}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run([sys.executable, str(REPO_ROOT / stage)], cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"{stage} failed with exit code {result.returncode}. Pipeline stopped.", file=sys.stderr, flush=True)
            return result.returncode
    if not run_config.PIPELINE_STAGES:
        print("[PIPELINE] No stages selected.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
