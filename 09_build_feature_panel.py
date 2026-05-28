
"""
This script:

1. Standardises key identifier/state columns
2. Normalises the train/test split labels
3. Derives explicit model columns such as state_is_out
4. Coerces predictor columns to numeric
5. Coerces binary targets/flags to numeric 0/1
6. Saves a modeling-panel CSV and a feature-spec JSON used by Step 1

Outputs
-------
- data/modeling_panel.csv
- data/feature_spec.json
"""

from __future__ import annotations
from pathlib import Path
import json
import re
import pandas as pd


# Global configuration
INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
INFILE = INDIR / "filtered_panel.csv"
OUTFILE = INDIR / "modeling_panel.csv"
FEATURE_SPEC_FILE = OUTDIR / "feature_spec.json"
COVARIATE_DICT_FILE = OUTDIR / "covariate_dictionary.csv"
D_ITEMS_PATH = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
SPLIT_COL = "split"

ACTION_COL = "removed_in_period"
Y_CAUTI = "cauti_in_period"
Y_REINS = "reinsertion_in_period"
LAST_PERIOD_COL = "is_last_period_of_episode"
END_REASON_COL = "episode_end_reason"

POST_REMOVE_RISK_PERIODS = 2

def _validate_split(df: pd.DataFrame) -> None:
    # Standardise split labels and fail if unexpected values are present.
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    valid_splits = {"train", "test"}
    found_splits = set(df[SPLIT_COL].dropna().unique())
    invalid_splits = sorted(found_splits - valid_splits)
    if invalid_splits:
        raise ValueError(f"Unexpected split values in {SPLIT_COL}: {invalid_splits}")


def _base_feature_cols(df: pd.DataFrame) -> list[str]:
    # Keep demographic and itemid-derived predictors in a stable order.
    cols = [
        c for c in df.columns
        if c.startswith("itemid_") or c.startswith("sex_") or c.startswith("ethnicity_")
    ]
    cols.append("age")
    seen = set()
    out = []
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str], fill_missing_with_zero: bool) -> None:
    # Convert model inputs and binary targets to numeric values.
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = df[col].replace({
                "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0,
                "true": 1, "false": 0,
            })
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if fill_missing_with_zero:
            df[col] = df[col].fillna(0).astype(int)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    return obj


def _detect_covariate_itemids(columns: list[str]) -> pd.DataFrame:
    # Parse itemid summary columns so the dictionary can label retained itemids.
    pattern = re.compile(r"^itemid_(\d+)__([a-z0-9_]+)$", flags=re.IGNORECASE)
    itemids = set()
    for col in columns:
        match = pattern.match(str(col))
        if not match:
            continue
        itemids.add(int(match.group(1)))
    return pd.DataFrame({"itemid": sorted(itemids)})


def main() -> None:
    OUTDIR.mkdir(exist_ok=True, parents=True)

    # Load the filtered panel and normalise the core identifier columns.
    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()

    _validate_split(df)

    # Add the explicit out-state feature used by the downstream models.
    df["state_is_out"] = (df[STATE_COL] == "out").astype(int)

    base_feature_cols = _base_feature_cols(df)

    # Coerce the model feature columns and time counters.
    _coerce_numeric(df, base_feature_cols + [TIME_COL, PERIODS_COL, "state_is_out"], fill_missing_with_zero=False)

    # Coerce binary targets and flags to 0/1 integers.
    target_flag_cols = [ACTION_COL, Y_CAUTI, Y_REINS, LAST_PERIOD_COL]
    _coerce_numeric(df, target_flag_cols, fill_missing_with_zero=True)

    feature_cols = list(base_feature_cols)
    x_cols_remove = [TIME_COL, PERIODS_COL, *feature_cols]
    x_cols_cauti = [TIME_COL, PERIODS_COL, "state_is_out", *feature_cols]
    x_cols_reins = [PERIODS_COL, *feature_cols]

    required_feature_cols = sorted(set(x_cols_remove + x_cols_cauti + x_cols_reins))
    missing_required = [c for c in required_feature_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required Step 1 feature columns after preprocessing: {missing_required}")

    # Save the cleaned modeling panel before writing the metadata side files.
    df.to_csv(OUTFILE, index=False)

    period_hours = int(
        pd.to_numeric(df["interval_hours"], errors="coerce")
        .dropna()
        .mode()
        .iloc[0]
    )

    covariate_dict = _detect_covariate_itemids(df.columns.tolist())
    d_items_df = pd.read_csv(D_ITEMS_PATH, usecols=["itemid", "label"], low_memory=False).drop_duplicates("itemid")
    d_items_df["itemid"] = pd.to_numeric(d_items_df["itemid"], errors="coerce")
    d_items_df = d_items_df.dropna(subset=["itemid"]).copy()
    d_items_df["itemid"] = d_items_df["itemid"].astype(int)
    itemid_to_label = d_items_df.set_index("itemid")["label"].to_dict()

    # Build a readable mapping from retained itemids to MIMIC labels.
    covariate_dict["label"] = covariate_dict["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    covariate_dict.sort_values(["label", "itemid"]).to_csv(COVARIATE_DICT_FILE, index=False)

    # Save the feature specification consumed by Step 1.
    spec = {
        "id_col": ID_COL,
        "time_col": TIME_COL,
        "state_col": STATE_COL,
        "periods_col": PERIODS_COL,
        "split_col": SPLIT_COL,
        "action_col": ACTION_COL,
        "y_cauti": Y_CAUTI,
        "y_reins": Y_REINS,
        "last_period_col": LAST_PERIOD_COL,
        "end_reason_col": END_REASON_COL,
        "period_hours": period_hours,
        "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
        "base_feature_cols": base_feature_cols,
        "features": feature_cols,
        "x_cols_remove": x_cols_remove,
        "x_cols_cauti": x_cols_cauti,
        "x_cols_reins": x_cols_reins,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
    }
    FEATURE_SPEC_FILE.write_text(json.dumps(_json_ready(spec), indent=2), encoding="utf-8")

    print(f"[SAVE] modeling panel: {OUTFILE}")
    print(f"[SAVE] feature spec: {FEATURE_SPEC_FILE}")
    print(f"[SAVE] covariate dictionary: {COVARIATE_DICT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Base features: {len(base_feature_cols)}")
    print(f"Total features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
