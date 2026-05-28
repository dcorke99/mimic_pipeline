import json

from model_utils import (
    ACTION_COL,
    AT_RISK_CAUTI,
    AT_RISK_REINS,
    ID_COL,
    OUTDIR,
    PERIODS_COL,
    SPLIT_COL,
    STATE_COL,
    TIME_COL,
    Y_CAUTI,
    Y_REINS,
    insert_score_columns_before_age,
    json_ready,
)

import pandas as pd


PROPENSITY_PANEL = OUTDIR / "propensity_scored_panel.csv"
OUTCOME_PANEL = OUTDIR / "outcome_scored_panel.csv"
FINAL_PANEL = OUTDIR / "scored_panel.csv"

PROPENSITY_SCORE_COLS = ["p_remove_obs"]
OUTCOME_SCORE_COLS = [
    "p_cauti_if_keep",
    "p_cauti_if_remove",
    "p_cauti_if_out",
    "p_reins_if_remove",
    "p_reins_if_out",
]
ALL_SCORE_COLS = PROPENSITY_SCORE_COLS + OUTCOME_SCORE_COLS


def _read_panel(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing scored panel: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)

    propensity_df = _read_panel(PROPENSITY_PANEL)
    outcome_df = _read_panel(OUTCOME_PANEL)

    if len(propensity_df) != len(outcome_df):
        raise ValueError(
            "Cannot merge scored panels because row counts differ: "
            f"{len(propensity_df)} propensity rows vs {len(outcome_df)} outcome rows"
        )

    required_base_cols = [
        ID_COL,
        "stay_id",
        "hadm_id",
        STATE_COL,
        PERIODS_COL,
        TIME_COL,
        SPLIT_COL,
        ACTION_COL,
        Y_CAUTI,
        Y_REINS,
        AT_RISK_CAUTI,
        AT_RISK_REINS,
    ]
    missing_propensity = [c for c in required_base_cols + PROPENSITY_SCORE_COLS if c not in propensity_df.columns]
    missing_outcome = [c for c in required_base_cols + OUTCOME_SCORE_COLS if c not in outcome_df.columns]
    if missing_propensity:
        raise ValueError(f"Propensity panel is missing required columns: {missing_propensity}")
    if missing_outcome:
        raise ValueError(f"Outcome panel is missing required columns: {missing_outcome}")

    for col in required_base_cols:
        if not propensity_df[col].astype(str).equals(outcome_df[col].astype(str)):
            raise ValueError(f"Cannot merge scored panels because base column differs by row: {col}")

    final_df = propensity_df.drop(columns=[c for c in OUTCOME_SCORE_COLS if c in propensity_df.columns]).copy()
    for col in OUTCOME_SCORE_COLS:
        final_df[col] = outcome_df[col]

    final_df = insert_score_columns_before_age(final_df, ALL_SCORE_COLS)
    final_df.to_csv(FINAL_PANEL, index=False, float_format="%.6f")

    metrics = {
        "artifacts": {
            "propensity_scored_panel": str(PROPENSITY_PANEL),
            "outcome_scored_panel": str(OUTCOME_PANEL),
            "final_scored_panel": str(FINAL_PANEL),
        },
        "n_rows": int(len(final_df)),
        "scored_panel_schema": {
            "required_columns_present": all(col in final_df.columns for col in required_base_cols + ALL_SCORE_COLS),
            "missing_columns": [col for col in required_base_cols + ALL_SCORE_COLS if col not in final_df.columns],
        },
    }
    (OUTDIR / "merge_metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2),
        encoding="utf-8",
    )

    print("\n--- SUCCESS MERGE ---", flush=True)
    print(f"Merged scored panel saved: {FINAL_PANEL}", flush=True)


if __name__ == "__main__":
    main()
