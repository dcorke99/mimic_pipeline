
"""
This script:

1. Standardises key identifier/state columns
2. Normalises the train/test split labels
3. Derives explicit model columns such as state_is_out
4. Coerces predictor columns to numeric
5. Coerces binary targets/flags to numeric 0/1
6. Saves a feature-panel CSV and a feature-spec JSON used by Step 1

Outputs
-------
- data/feature_panel.csv
- data/feature_spec.json
"""

from __future__ import annotations
from pathlib import Path
import json
import re
import pandas as pd


# Global configuration
INDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
INFILE = INDIR / "filtered_panel.csv"
OUTFILE = INDIR / "feature_panel.csv"
FEATURE_SPEC_FILE = OUTDIR / "feature_spec.json"
COVARIATE_DICT_FILE = OUTDIR / "covariate_dictionary.csv"
D_ITEMS_PATH = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
DAYS_COL = "days_in_state"
SPLIT_COL = "split"

ACTION_COL = "removed_today"
Y_CAUTI = "cauti_today"
Y_REINS = "reinsertion_today"
LAST_DAY_COL = "is_last_day_of_episode"
END_REASON_COL = "episode_end_reason"

POST_REMOVE_RISK_DAYS = 2
KEEP_STATS = {"mean"}


def _validate_split(df: pd.DataFrame) -> None:
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()


def _base_feature_cols(df: pd.DataFrame) -> list[str]:
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
def _detect_covariate_cols(columns: list[str], keep_stats: set[str]) -> pd.DataFrame:
    pattern = re.compile(r"^itemid_(\d+)__([a-z0-9_]+)$", flags=re.IGNORECASE)
    rows = []
    for col in columns:
        match = pattern.match(str(col))
        if not match:
            continue
        itemid = int(match.group(1))
        stat = match.group(2).lower()
        if stat in keep_stats:
            rows.append({"col": col, "itemid": itemid, "stat": stat})
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()

    _validate_split(df)

    df["state_is_out"] = (df[STATE_COL] == "out").astype(int)

    base_feat = _base_feature_cols(df)

    _coerce_numeric(df, base_feat + [TIME_COL, DAYS_COL, "state_is_out"], fill_missing_with_zero=False)

    target_flag_cols = [ACTION_COL, Y_CAUTI, Y_REINS, LAST_DAY_COL]
    _coerce_numeric(df, target_flag_cols, fill_missing_with_zero=True)

    feat = list(base_feat)
    x_cols_remove = [TIME_COL, DAYS_COL, *feat]
    x_cols_cauti = [TIME_COL, DAYS_COL, "state_is_out", *feat]
    x_cols_reins = [DAYS_COL, *feat]

    required_feature_cols = sorted(set(x_cols_remove + x_cols_cauti + x_cols_reins))
    missing_required = [c for c in required_feature_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required Step 1 feature columns after preprocessing: {missing_required}")

    df.to_csv(OUTFILE, index=False)

    covariate_dict = _detect_covariate_cols(df.columns.tolist(), KEEP_STATS)
    d_items = pd.read_csv(D_ITEMS_PATH, usecols=["itemid", "label"], low_memory=False).drop_duplicates("itemid")
    d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")
    d_items = d_items.dropna(subset=["itemid"])
    d_items["itemid"] = d_items["itemid"].astype(int)
    itemid_to_label = d_items.set_index("itemid")["label"].to_dict()
    covariate_dict["label"] = covariate_dict["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    covariate_dict["description"] = covariate_dict["label"].astype(str) + " [" + covariate_dict["stat"].astype(str) + "]"
    covariate_dict.sort_values(["label", "itemid", "stat"]).drop(columns=["stat", "description"]).to_csv(
        COVARIATE_DICT_FILE, index=False
    )

    spec = {
        "id_col": ID_COL,
        "time_col": TIME_COL,
        "state_col": STATE_COL,
        "days_col": DAYS_COL,
        "split_col": SPLIT_COL,
        "action_col": ACTION_COL,
        "y_cauti": Y_CAUTI,
        "y_reins": Y_REINS,
        "last_day_col": LAST_DAY_COL,
        "end_reason_col": END_REASON_COL,
        "post_remove_risk_days": POST_REMOVE_RISK_DAYS,
        "base_feature_cols": base_feat,
        "features": feat,
        "x_cols_remove": x_cols_remove,
        "x_cols_cauti": x_cols_cauti,
        "x_cols_reins": x_cols_reins,
        "n_rows": int(len(df)),
        "n_features": int(len(feat)),
    }
    FEATURE_SPEC_FILE.write_text(json.dumps(_json_ready(spec), indent=2), encoding="utf-8")

    print(f"[SAVE] feature panel: {OUTFILE}")
    print(f"[SAVE] feature spec: {FEATURE_SPEC_FILE}")
    print(f"[SAVE] covariate dictionary: {COVARIATE_DICT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Base features: {len(base_feat)}")
    print(f"Total features: {len(feat)}")


if __name__ == "__main__":
    main()
