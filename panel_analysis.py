from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config
DATA_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\filtered_panel.csv")
RESULTS_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\panel_analysis")
D_ITEMS_PATH = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

STEP1_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\step1")
STEP1_TOP_MODEL_FEATURES_FILE = STEP1_DIR / "step1_top_model_features.csv"
STEP1_TOP_SHAP_FEATURES_FILE = STEP1_DIR / "step1_top_shap_features.csv"

DP = 3
KEEP_STATS = {"mean"}
MIN_N_PER_GROUP = 20
LATE_REMOVAL_DAY_THRESHOLD = 7
AGE_THRESHOLD = 60
TOP_N_COVARIATES = 20

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
DAYS_COL = "days_in_state"
INTERVAL_COL = "interval_hours"
ACTION_COL = "removed_today"
SPLIT_COL = "split"
Y_CAUTI = "cauti_today"
Y_REINS = "reinsertion_today"
LAST_DAY_COL = "is_last_day_of_episode"
END_REASON_COL = "episode_end_reason"
EPISODE_KEYS = ["stay_id", "inserted"]
POST_REMOVE_RISK_DAYS = 2
LATE_REMOVAL_COL = "late_removal_today"
AGE_GROUP_COL = "age_ge_threshold"

REMOVAL_FEATURE = "GCS - Verbal Response [mean]"
CAUTI_BINARY_FEATURE = "sex_M"
CAUTI_CONTINUOUS_FEATURE = "Anion gap [mean]"
REINSERTION_FEATURE = "Bladder Scan Estimate [mean]"


# Detect columns like itemid_<ID>__mean/min/max and return the matching columns plus metadata.
def detect_covariate_cols(columns: list[str], keep_stats: set[str]) -> tuple[list[str], pd.DataFrame]:
    pat = re.compile(r"^itemid_(\d+)__(mean|min|max)$", flags=re.IGNORECASE)
    cov_cols: list[str] = []
    meta: list[dict[str, object]] = []
    for c in columns:
        m = pat.match(str(c))
        if not m:
            continue
        itemid = int(m.group(1))
        stat = m.group(2).lower()
        if stat in keep_stats:
            cov_cols.append(c)
            meta.append({"col": c, "itemid": itemid, "stat": stat})
    return cov_cols, pd.DataFrame(meta)


# Load item labels for readable output tables.
def load_item_labels(d_items_path: Path) -> dict[int, str]:
    if not d_items_path.exists():
        return {}
    d_items = pd.read_csv(d_items_path, usecols=["itemid", "label"], low_memory=False).drop_duplicates("itemid")
    d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")
    d_items = d_items.dropna(subset=["itemid"])
    d_items["itemid"] = d_items["itemid"].astype(int)
    return d_items.set_index("itemid")["label"].to_dict()


# Normalise the split labels to avoid train/test mismatches.
def validate_split(df: pd.DataFrame) -> None:
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()


# Coerce a set of columns to numeric, handling TRUE/FALSE strings.
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
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


# Raise a clear error if required columns are missing from the panel.
def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# Load a Step 1 top-features table saved by 01_step1_transition_models.py.
def load_step1_top_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Step 1 feature table not found: {path}")

    df = pd.read_csv(path)
    required = {"model", "feature"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    out = df.copy()
    out["model"] = out["model"].astype(str).str.strip().str.lower()
    out["feature"] = out["feature"].astype(str).str.strip()

    sort_cols = [c for c in ["model", "rank", "feature"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


# Return the ordered feature names for one model from a Step 1 feature table.
def get_step1_feature_names(step1_features: pd.DataFrame, model_name: str) -> list[str]:
    tmp = step1_features[step1_features["model"] == model_name].copy()

    if "rank" in tmp.columns:
        tmp = tmp.sort_values(["rank", "feature"])
    else:
        tmp = tmp.sort_values(["feature"])

    feature_names: list[str] = []
    seen: set[str] = set()

    for feature in tmp["feature"].tolist():
        if feature not in seen:
            feature_names.append(feature)
            seen.add(feature)

    return feature_names


# Calculate Cliff's delta for two groups as a simple effect-size summary.
def cliffs_delta(x1: np.ndarray, x0: np.ndarray) -> float:
    if x1.size == 0 or x0.size == 0:
        return np.nan
    xy = np.concatenate([x1, x0])
    ranks = stats.rankdata(xy)
    rx = ranks[: x1.size].sum()
    u = rx - x1.size * (x1.size + 1) / 2
    delta = (2 * u) / (x1.size * x0.size) - 1
    return float(delta)


# Apply Benjamini-Hochberg correction to a list/series of p-values.
def p_adjust_bh(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    adjusted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = valid.iloc[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = min(prev, 1.0)
    out.loc[valid.index] = adjusted
    return out


# Build first-event CAUTI and reinsertion fitting flags used by the current panel logic.
def build_risk_sets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(EPISODE_KEYS + ["day_end"]).copy()
    y_cauti = pd.to_numeric(out[Y_CAUTI], errors="coerce").fillna(0)
    out["prior_cauti_count"] = out.groupby(EPISODE_KEYS)[Y_CAUTI].cumsum() - y_cauti
    out["cauti_risk_row"] = (
        (out[STATE_COL] == "in") |
        ((out[STATE_COL] == "out") & (out[DAYS_COL] <= POST_REMOVE_RISK_DAYS))
    ).astype(int)
    out["reinsertion_fit_row"] = (
        (out[STATE_COL] == "out") &
        ~(
            (out[LAST_DAY_COL] == 1) &
            (out[END_REASON_COL] == "icu_end") &
            (out[Y_REINS] == 0)
        )
    ).astype(int)
    out["removal_fit_row"] = (
        (out[STATE_COL] == "in") &
        (out["prior_cauti_count"] == 0)
    ).astype(int)
    return out


# Convert the current panel structure into tidy overview tables suitable for sharing.
def build_overview_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_rows = [
        {
            "subset": "overall",
            "rows": int(len(df)),
            "unique_subject_id": int(df[ID_COL].nunique()),
            "unique_hadm_id": int(df["hadm_id"].nunique()),
            "unique_stay_id": int(df["stay_id"].nunique()),
            "unique_episodes": int(df[EPISODE_KEYS].drop_duplicates().shape[0]),
        }
    ]
    for split, g in df.groupby(SPLIT_COL, dropna=False):
        cohort_rows.append({
            "subset": f"split={split}",
            "rows": int(len(g)),
            "unique_subject_id": int(g[ID_COL].nunique()),
            "unique_hadm_id": int(g["hadm_id"].nunique()),
            "unique_stay_id": int(g["stay_id"].nunique()),
            "unique_episodes": int(g[EPISODE_KEYS].drop_duplicates().shape[0]),
        })
    cohort_overview = pd.DataFrame(cohort_rows)

    state_overview = (
        df.groupby([SPLIT_COL, STATE_COL], dropna=False)
        .agg(
            rows=(STATE_COL, "size"),
            unique_subject_id=(ID_COL, "nunique"),
            unique_episodes=("stay_id", "nunique"),
        )
        .reset_index()
    )

    event_rows = []
    masks = {
        "removal_today_on_in_rows": df[STATE_COL] == "in",
        "late_removal_today_on_in_rows": (df[STATE_COL] == "in") & (df[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD),
        "cauti_today_on_cauti_risk_rows": df["cauti_risk_row"] == 1,
        "reinsertion_today_on_out_fit_rows": df["reinsertion_fit_row"] == 1,
    }
    targets = {
        "removal_today_on_in_rows": ACTION_COL,
        "late_removal_today_on_in_rows": LATE_REMOVAL_COL,
        "cauti_today_on_cauti_risk_rows": Y_CAUTI,
        "reinsertion_today_on_out_fit_rows": Y_REINS,
    }
    for name, mask in masks.items():
        g = df.loc[mask]
        y_col = targets[name]
        event_rows.append({
            "analysis_set": name,
            "rows": int(len(g)),
            "events": int(g[y_col].sum()) if len(g) else 0,
            "event_rate": float(g[y_col].mean()) if len(g) else np.nan,
        })
    event_overview = pd.DataFrame(event_rows)

    risk_set_summary = pd.DataFrame([
        {"metric": "all_rows", "value": int(len(df))},
        {"metric": "in_rows", "value": int((df[STATE_COL] == "in").sum())},
        {"metric": "out_rows", "value": int((df[STATE_COL] == "out").sum())},
        {
            "metric": "out_rows_days_in_state_le_2",
            "value": int(((df[STATE_COL] == "out") & (df[DAYS_COL] <= POST_REMOVE_RISK_DAYS)).sum()),
        },
        {"metric": "cauti_risk_rows", "value": int(df["cauti_risk_row"].sum())},
        {"metric": "removal_fit_rows", "value": int(df["removal_fit_row"].sum())},
        {"metric": "reinsertion_fit_rows", "value": int(df["reinsertion_fit_row"].sum())},
        {"metric": "excluded_post_cauti_rows", "value": int((df["prior_cauti_count"] > 0).sum())},
        {
            "metric": "excluded_terminal_out_rows",
            "value": int(((df[STATE_COL] == "out") & (df["reinsertion_fit_row"] == 0)).sum()),
        },
    ])

    return cohort_overview, state_overview, event_overview, risk_set_summary


# Summarise event rates by state and day so timing patterns are easy to review.
def build_day_rate_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cauti_day = (
        df.loc[df["cauti_risk_row"] == 1]
        .groupby([STATE_COL, DAYS_COL], dropna=False)[Y_CAUTI]
        .agg(rows="count", events="sum", event_rate="mean")
        .reset_index()
    )
    reinsertion_day = (
        df.loc[df["reinsertion_fit_row"] == 1]
        .groupby(DAYS_COL, dropna=False)[Y_REINS]
        .agg(rows="count", events="sum", event_rate="mean")
        .reset_index()
    )
    return cauti_day, reinsertion_day

# Resolve one required feature name into an actual dataframe column.
def resolve_required_feature_col(
    feature_name: str,
    pretty_to_raw: dict[str, str],
    available_cols: set[str],
) -> str:
    if feature_name in available_cols:
        return feature_name

    raw = pretty_to_raw.get(feature_name)
    if raw is not None and raw in available_cols:
        return raw

    raise ValueError(f"Required feature not found in panel columns: {feature_name}")


# Collapse the panel to one row per catheter episode for simple episode-level testing.
def build_episode_level_table(
    df: pd.DataFrame,
    removal_feature: str,
    cauti_binary_feature: str,
    cauti_continuous_feature: str,
    reinsertion_feature: str,
) -> pd.DataFrame:
    d = df.sort_values(EPISODE_KEYS + ["day_end"]).copy()

    in_rows = d.loc[d[STATE_COL] == "in"].copy()
    cauti_rows = d.loc[d["cauti_risk_row"] == 1].copy()
    reinsertion_rows = d.loc[d["reinsertion_fit_row"] == 1].copy()

    if in_rows.empty:
        return pd.DataFrame(columns=EPISODE_KEYS + [
            ID_COL,
            "hadm_id",
            "age",
            "catheter_days",
            "late_removal_episode",
            "cauti_episode",
            "reinsertion_episode",
            removal_feature,
            cauti_binary_feature,
            cauti_continuous_feature,
            reinsertion_feature,
        ])

    episode_age = d.groupby(EPISODE_KEYS, dropna=False)["age"].first().rename("age")
    episode_subject = d.groupby(EPISODE_KEYS, dropna=False)[ID_COL].first().rename(ID_COL)
    episode_hadm = d.groupby(EPISODE_KEYS, dropna=False)["hadm_id"].first().rename("hadm_id")

    catheter_days = (
        in_rows.groupby(EPISODE_KEYS, dropna=False)[DAYS_COL]
        .max()
        .rename("catheter_days")
    )

    late_removal_episode = (catheter_days >= LATE_REMOVAL_DAY_THRESHOLD).astype("int8").rename("late_removal_episode")

    cauti_episode = (
        d.groupby(EPISODE_KEYS, dropna=False)[Y_CAUTI]
        .max()
        .fillna(0)
        .clip(0, 1)
        .astype("int8")
        .rename("cauti_episode")
    )

    reinsertion_episode = (
        d.groupby(EPISODE_KEYS, dropna=False)[Y_REINS]
        .max()
        .fillna(0)
        .clip(0, 1)
        .astype("int8")
        .rename("reinsertion_episode")
    )

    removal_feature_episode = (
        in_rows.groupby(EPISODE_KEYS, dropna=False)[removal_feature]
        .median()
        .rename(removal_feature)
    )

    cauti_binary_feature_episode = (
        d.groupby(EPISODE_KEYS, dropna=False)[cauti_binary_feature]
        .first()
        .rename(cauti_binary_feature)
    )

    cauti_continuous_feature_episode = (
        cauti_rows.groupby(EPISODE_KEYS, dropna=False)[cauti_continuous_feature]
        .median()
        .rename(cauti_continuous_feature)
    )

    reinsertion_feature_episode = (
        reinsertion_rows.groupby(EPISODE_KEYS, dropna=False)[reinsertion_feature]
        .median()
        .rename(reinsertion_feature)
    )

    out = pd.concat(
        [
            episode_subject,
            episode_hadm,
            episode_age,
            catheter_days,
            late_removal_episode,
            cauti_episode,
            reinsertion_episode,
            removal_feature_episode,
            cauti_binary_feature_episode,
            cauti_continuous_feature_episode,
            reinsertion_feature_episode,
        ],
        axis=1,
    ).reset_index()

    numeric_cols = [
        "age",
        "catheter_days",
        "late_removal_episode",
        "cauti_episode",
        "reinsertion_episode",
        removal_feature,
        cauti_binary_feature,
        cauti_continuous_feature,
        reinsertion_feature,
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["late_removal_episode", "cauti_episode", "reinsertion_episode", cauti_binary_feature]:
        if col in out.columns:
            out[col] = out[col].fillna(0).clip(0, 1).astype("int8")

    out = out.dropna(subset=["catheter_days"]).copy()
    return out


# Mann-Whitney U test for a continuous value across a binary episode-level outcome.
def mannwhitney_group_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    test_name: str,
    analysis_set: str,
) -> dict[str, object]:
    tmp = df[[value_col, group_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp[group_col] = pd.to_numeric(tmp[group_col], errors="coerce")
    tmp = tmp.dropna().copy()
    tmp = tmp[tmp[group_col].isin([0, 1])]

    x1 = tmp.loc[tmp[group_col] == 1, value_col].to_numpy(dtype=float)
    x0 = tmp.loc[tmp[group_col] == 0, value_col].to_numpy(dtype=float)

    u_stat = np.nan
    p_value = np.nan
    delta = np.nan

    if x1.size >= MIN_N_PER_GROUP and x0.size >= MIN_N_PER_GROUP:
        try:
            u_stat, p_value = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            u_stat = float(u_stat)
            p_value = float(p_value)
        except Exception:
            u_stat = np.nan
            p_value = np.nan
        delta = cliffs_delta(x1, x0)

    return {
        "test_name": test_name,
        "test_type": "Mann-Whitney U",
        "analysis_set": analysis_set,
        "value_col": value_col,
        "group_col": group_col,
        "n_group_1": int(x1.size),
        "n_group_0": int(x0.size),
        "mean_group_1": float(np.mean(x1)) if x1.size else np.nan,
        "mean_group_0": float(np.mean(x0)) if x0.size else np.nan,
        "median_group_1": float(np.median(x1)) if x1.size else np.nan,
        "median_group_0": float(np.median(x0)) if x0.size else np.nan,
        "statistic": u_stat,
        "p_value": p_value,
        "cliffs_delta": delta,
    }


# Fisher exact test for a binary exposure across a binary episode-level outcome.
def binary_group_test(
    df: pd.DataFrame,
    exposure_col: str,
    outcome_col: str,
    test_name: str,
    analysis_set: str,
) -> dict[str, object]:
    tmp = df[[exposure_col, outcome_col]].copy()
    tmp[exposure_col] = pd.to_numeric(tmp[exposure_col], errors="coerce")
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
    tmp = tmp.dropna().copy()
    tmp = tmp[tmp[exposure_col].isin([0, 1]) & tmp[outcome_col].isin([0, 1])]

    if tmp.empty:
        return {
            "test_name": test_name,
            "test_type": "Fisher exact",
            "analysis_set": analysis_set,
            "exposure_col": exposure_col,
            "outcome_col": outcome_col,
            "n": 0,
            "exposed_n": 0,
            "unexposed_n": 0,
            "event_rate_exposed": np.nan,
            "event_rate_unexposed": np.nan,
            "odds_ratio": np.nan,
            "risk_difference": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "chi2_p_value": np.nan,
        }

    table = pd.crosstab(tmp[exposure_col], tmp[outcome_col]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    a = int(table.loc[1, 1])
    b = int(table.loc[1, 0])
    c = int(table.loc[0, 1])
    d = int(table.loc[0, 0])

    exposed_n = a + b
    unexposed_n = c + d
    event_rate_exposed = a / exposed_n if exposed_n > 0 else np.nan
    event_rate_unexposed = c / unexposed_n if unexposed_n > 0 else np.nan
    risk_difference = event_rate_exposed - event_rate_unexposed if exposed_n > 0 and unexposed_n > 0 else np.nan

    odds_ratio = np.nan
    fisher_p = np.nan
    try:
        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])
        odds_ratio = float(odds_ratio)
        fisher_p = float(fisher_p)
    except Exception:
        odds_ratio = np.nan
        fisher_p = np.nan

    chi2_p = np.nan
    try:
        chi2_p = float(stats.chi2_contingency(table.values, correction=False)[1])
    except Exception:
        chi2_p = np.nan

    return {
        "test_name": test_name,
        "test_type": "Fisher exact",
        "analysis_set": analysis_set,
        "exposure_col": exposure_col,
        "outcome_col": outcome_col,
        "n": int(len(tmp)),
        "exposed_n": int(exposed_n),
        "unexposed_n": int(unexposed_n),
        "event_rate_exposed": event_rate_exposed,
        "event_rate_unexposed": event_rate_unexposed,
        "odds_ratio": odds_ratio,
        "risk_difference": risk_difference,
        "statistic": odds_ratio,
        "p_value": fisher_p,
        "chi2_p_value": chi2_p,
    }


# Scan itemid mean covariates for univariable associations with each outcome and rank the top signals.
def top_covariate_screen(
    df: pd.DataFrame,
    covariates: list[str],
    covariate_name_map: dict[str, str],
    outcome_col: str,
    analysis_set: str,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    d = df.copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    d = d[d[outcome_col].isin([0, 1])].copy()

    for col in covariates:
        x1 = pd.to_numeric(d.loc[d[outcome_col] == 1, col], errors="coerce").dropna().to_numpy(dtype=float)
        x0 = pd.to_numeric(d.loc[d[outcome_col] == 0, col], errors="coerce").dropna().to_numpy(dtype=float)
        if x1.size < MIN_N_PER_GROUP or x0.size < MIN_N_PER_GROUP:
            continue
        try:
            mw_p = float(stats.mannwhitneyu(x1, x0, alternative="two-sided").pvalue)
        except Exception:
            mw_p = np.nan
        rows.append({
            "analysis_set": analysis_set,
            "outcome": outcome_col,
            "covariate": col,
            "covariate_name": covariate_name_map.get(col, col),
            "n_outcome_1": int(x1.size),
            "n_outcome_0": int(x0.size),
            "median_outcome_1": float(np.median(x1)),
            "median_outcome_0": float(np.median(x0)),
            "mean_outcome_1": float(np.mean(x1)),
            "mean_outcome_0": float(np.mean(x0)),
            "cliffs_delta": cliffs_delta(x1, x0),
            "mannwhitney_p": mw_p,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mannwhitney_p_adj_bh"] = p_adjust_bh(out["mannwhitney_p"])
    out = out.sort_values(["mannwhitney_p_adj_bh", "mannwhitney_p", "covariate_name"]).head(top_n).reset_index(drop=True)
    return out


# Resolve a list of feature names into actual dataframe columns.
def resolve_feature_cols(
    feature_names: list[str],
    pretty_to_raw: dict[str, str],
    available_cols: set[str],
) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()

    for name in feature_names:
        raw = pretty_to_raw.get(name, name)
        if raw in available_cols and raw not in seen:
            cols.append(raw)
            seen.add(raw)

    return cols


# Return describe-style summary rows for selected covariates/features.
def describe_selected_covariates(
    df: pd.DataFrame,
    value_cols: list[str],
    description_map: dict[str, str],
    analysis_set_name: str,
    model_name: str,
    feature_source: str,
    dp: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for col in value_cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        n = int(s.shape[0])

        rows.append({
            "model": model_name,
            "feature_source": feature_source,
            "analysis_set": analysis_set_name,
            "covariate": description_map.get(col, col),
            "col": col,
            "N": n,
            "mean": float(s.mean()) if n else np.nan,
            "sd": float(s.std(ddof=1)) if n > 1 else np.nan,
            "min": float(s.min()) if n else np.nan,
            "q1": float(s.quantile(0.25)) if n else np.nan,
            "median": float(s.median()) if n else np.nan,
            "q3": float(s.quantile(0.75)) if n else np.nan,
            "max": float(s.max()) if n else np.nan,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].round(dp)
    return out


# Save a simple line plot of event rates by days_in_state for the main event processes.
def plot_event_rates(cauti_day: pd.DataFrame, reinsertion_day: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    in_rows = cauti_day[cauti_day[STATE_COL] == "in"]
    out_rows = cauti_day[cauti_day[STATE_COL] == "out"]
    if not in_rows.empty:
        ax.plot(in_rows[DAYS_COL], in_rows["event_rate"], marker="o", label="CAUTI risk set: IN")
    if not out_rows.empty:
        ax.plot(out_rows[DAYS_COL], out_rows["event_rate"], marker="o", label="CAUTI risk set: OUT")
    if not reinsertion_day.empty:
        ax.plot(reinsertion_day[DAYS_COL], reinsertion_day["event_rate"], marker="o", label="Reinsertion on OUT fit rows")
    ax.set_xlabel("days_in_state")
    ax.set_ylabel("event rate")
    ax.set_title("Event rates by days_in_state")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot_event_rates_by_days_in_state.png", dpi=150)
    plt.close(fig)


# Save a simple age-group bar plot for late removal.
def plot_age_group_late_removal(df_in: pd.DataFrame, outdir: Path) -> None:
    rates = df_in.groupby(AGE_GROUP_COL)[LATE_REMOVAL_COL].mean().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"<{AGE_THRESHOLD}", f"≥{AGE_THRESHOLD}"], rates.fillna(0).to_numpy(dtype=float))
    ax.set_ylabel("late removal rate")
    ax.set_title(f"Late removal (day ≥ {LATE_REMOVAL_DAY_THRESHOLD}) by age group")
    fig.tight_layout()
    fig.savefig(outdir / "plot_late_removal_by_age_group.png", dpi=150)
    plt.close(fig)


# Run the merged panel diagnostics and supervisor-facing descriptive/inferential analysis.
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = df.columns.str.strip()

    required_cols = [
        ID_COL, TIME_COL, STATE_COL, DAYS_COL, INTERVAL_COL, ACTION_COL, SPLIT_COL,
        Y_CAUTI, Y_REINS, LAST_DAY_COL, END_REASON_COL, "day_end", *EPISODE_KEYS,
        "hadm_id", "stay_id", "age",
    ]
    check_required_columns(df, required_cols)

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()
    validate_split(df)

    cov_cols, cov_meta = detect_covariate_cols(df.columns.tolist(), KEEP_STATS)
    if len(cov_cols) == 0:
        raise ValueError("No covariate columns detected matching itemid_<ID>__<stat> for the current panel.")

    itemid_to_label = load_item_labels(D_ITEMS_PATH)
    cov_meta["label"] = cov_meta["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    cov_meta["description"] = cov_meta["label"].astype(str) + " [" + cov_meta["stat"].astype(str) + "]"
    covariate_name_map = dict(zip(cov_meta["col"], cov_meta["description"]))
    pretty_to_raw = {v: k for k, v in covariate_name_map.items()}

    numeric_cols = [
        TIME_COL, DAYS_COL, INTERVAL_COL, ACTION_COL, Y_CAUTI, Y_REINS,
        LAST_DAY_COL, "age", "hadm_id", "stay_id", CAUTI_BINARY_FEATURE,
    ] + cov_cols
    coerce_numeric(df, numeric_cols)

    covariate_count = df[cov_cols].notna().sum(axis=1).astype("int32")
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    high_covariate_count = (covariate_count >= covariate_count.median()).astype("int8")

    derived_cols = pd.DataFrame({
        "covariate_count": covariate_count,
        AGE_GROUP_COL: (age_numeric >= AGE_THRESHOLD).astype("int8"),
        "high_covariate_count": high_covariate_count,
        LATE_REMOVAL_COL: ((df[ACTION_COL] == 1) & (df[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD)).astype("int8"),
    }, index=df.index)

    df = pd.concat([df, derived_cols], axis=1).copy()

    df["day_end"] = pd.to_datetime(df["day_end"], errors="coerce")
    df["inserted"] = pd.to_datetime(df["inserted"], errors="coerce")

    df = build_risk_sets(df)

    step1_top_model_features = load_step1_top_features(STEP1_TOP_MODEL_FEATURES_FILE)
    step1_top_shap_features = load_step1_top_features(STEP1_TOP_SHAP_FEATURES_FILE)

    available_cols = set(df.columns)
    removal_feature = resolve_required_feature_col(REMOVAL_FEATURE, pretty_to_raw, available_cols)
    cauti_binary_feature = resolve_required_feature_col(CAUTI_BINARY_FEATURE, pretty_to_raw, available_cols)
    cauti_continuous_feature = resolve_required_feature_col(CAUTI_CONTINUOUS_FEATURE, pretty_to_raw, available_cols)
    reinsertion_feature = resolve_required_feature_col(REINSERTION_FEATURE, pretty_to_raw, available_cols)

    cov_meta.sort_values(["label", "itemid", "stat"]).to_csv(RESULTS_DIR / "00_covariate_dictionary.csv", index=False)

    cohort_overview, state_overview, event_overview, risk_set_summary = build_overview_tables(df)
    cohort_overview.round(DP).to_csv(RESULTS_DIR / "01_cohort_overview.csv", index=False)
    state_overview.round(DP).to_csv(RESULTS_DIR / "02_state_overview.csv", index=False)
    event_overview.round(DP).to_csv(RESULTS_DIR / "03_event_overview.csv", index=False)
    risk_set_summary.round(DP).to_csv(RESULTS_DIR / "04_risk_set_summary.csv", index=False)

    cauti_day, reinsertion_day = build_day_rate_tables(df)
    cauti_day.round(DP).to_csv(RESULTS_DIR / "05_cauti_event_rates_by_state_and_day.csv", index=False)
    reinsertion_day.round(DP).to_csv(RESULTS_DIR / "06_reinsertion_event_rates_by_day.csv", index=False)

    analysis_sets = {
        "overall": df,
        "in_rows": df.loc[df[STATE_COL] == "in"],
        "out_rows": df.loc[df[STATE_COL] == "out"],
        "cauti_risk_rows": df.loc[df["cauti_risk_row"] == 1],
        "reinsertion_fit_rows": df.loc[df["reinsertion_fit_row"] == 1],
    }
    in_rows = analysis_sets["in_rows"]
    cauti_rows = analysis_sets["cauti_risk_rows"]
    out_fit_rows = analysis_sets["reinsertion_fit_rows"]

    episode_df = build_episode_level_table(
        df=df,
        removal_feature=removal_feature,
        cauti_binary_feature=cauti_binary_feature,
        cauti_continuous_feature=cauti_continuous_feature,
        reinsertion_feature=reinsertion_feature,
    )
    episode_df_export = episode_df.copy()
    episode_feature_cols = [
        removal_feature,
        cauti_continuous_feature,
        reinsertion_feature,
    ]
    for col in episode_feature_cols:
        if col in episode_df_export.columns:
            episode_df_export[col] = pd.to_numeric(episode_df_export[col], errors="coerce").round(DP)
    episode_df_export.to_csv(RESULTS_DIR / "08_episode_level_analysis_table.csv", index=False)

    episode_tests = pd.DataFrame([
        mannwhitney_group_test(
            episode_df,
            value_col=removal_feature,
            group_col="late_removal_episode",
            test_name=f"{REMOVAL_FEATURE} by late removal episode (catheter days >= {LATE_REMOVAL_DAY_THRESHOLD})",
            analysis_set="episode_level",
        ),
        binary_group_test(
            episode_df,
            exposure_col=cauti_binary_feature,
            outcome_col="cauti_episode",
            test_name=f"{CAUTI_BINARY_FEATURE} by CAUTI episode",
            analysis_set="episode_level",
        ),
        mannwhitney_group_test(
            episode_df,
            value_col=cauti_continuous_feature,
            group_col="cauti_episode",
            test_name=f"{CAUTI_CONTINUOUS_FEATURE} by CAUTI episode",
            analysis_set="episode_level",
        ),
        mannwhitney_group_test(
            episode_df,
            value_col=reinsertion_feature,
            group_col="reinsertion_episode",
            test_name=f"{REINSERTION_FEATURE} by reinsertion episode",
            analysis_set="episode_level",
        ),
    ])
    episode_tests["p_value_adj_bh"] = p_adjust_bh(episode_tests["p_value"])
    episode_tests_export = episode_tests.copy()
    numeric_cols = [
        col for col in episode_tests_export.select_dtypes(include=[np.number]).columns
        if col not in {"p_value", "p_value_adj_bh"}
    ]
    episode_tests_export[numeric_cols] = episode_tests_export[numeric_cols].round(DP)
    episode_tests_export["p_value"] = episode_tests["p_value"].map(lambda x: f"{x:.3e}" if pd.notna(x) else "")
    episode_tests_export["p_value_adj_bh"] = episode_tests["p_value_adj_bh"].map(lambda x: f"{x:.3e}" if pd.notna(x) else "")
    episode_tests_export.to_csv(RESULTS_DIR / "09_episode_hypothesis_tests.csv", index=False)

    top_removal = top_covariate_screen(
        df=in_rows,
        covariates=cov_cols,
        covariate_name_map=covariate_name_map,
        outcome_col=ACTION_COL,
        analysis_set="in_rows_removal_today",
        top_n=TOP_N_COVARIATES,
    )
    top_reins = top_covariate_screen(
        df=out_fit_rows,
        covariates=cov_cols,
        covariate_name_map=covariate_name_map,
        outcome_col=Y_REINS,
        analysis_set="out_fit_rows_reinsertion_today",
        top_n=TOP_N_COVARIATES,
    )
    top_cauti = top_covariate_screen(
        df=cauti_rows,
        covariates=cov_cols,
        covariate_name_map=covariate_name_map,
        outcome_col=Y_CAUTI,
        analysis_set="cauti_risk_rows_cauti_today",
        top_n=TOP_N_COVARIATES,
    )
    top_signals = pd.concat([top_removal, top_reins, top_cauti], ignore_index=True)
    top_signals_export = top_signals.round(DP).copy()
    for col in ["mannwhitney_p", "mannwhitney_p_adj_bh"]:
        if col in top_signals_export.columns:
            top_signals_export[col] = top_signals[col].map(lambda x: f"{x:.3e}" if pd.notna(x) else "")
    top_signals_export.to_csv(RESULTS_DIR / "10_top_univariable_covariate_signals.csv", index=False)

    xgb_desc_parts = []
    model_describe_configs = [
        ("removal", "in_rows"),
        ("cauti", "cauti_risk_rows"),
        ("reinsertion", "reinsertion_fit_rows"),
    ]
    available_cols = set(df.columns)

    for feature_source, step1_features in [
        ("model_importance", step1_top_model_features),
        ("shap_importance", step1_top_shap_features),
    ]:
        for model_name, analysis_set_name in model_describe_configs:
            feature_names = get_step1_feature_names(step1_features, model_name)
            value_cols = resolve_feature_cols(feature_names, pretty_to_raw, available_cols)
            xgb_desc_parts.append(
                describe_selected_covariates(
                    df=analysis_sets[analysis_set_name],
                    value_cols=value_cols,
                    description_map=covariate_name_map,
                    analysis_set_name=analysis_set_name,
                    model_name=model_name,
                    feature_source=feature_source,
                    dp=DP,
                )
            )

    pd.concat(xgb_desc_parts, ignore_index=True).to_csv(
        RESULTS_DIR / "11_xgb_influential_covariate_descriptives.csv",
        index=False,
    )

    plot_event_rates(cauti_day, reinsertion_day, RESULTS_DIR)
    plot_age_group_late_removal(in_rows, RESULTS_DIR)

    print(f"Outputs saved to: {RESULTS_DIR}")
    print(f"Number of covariates analysed: {len(cov_cols)}")
    print(f"Late removal threshold: day >= {LATE_REMOVAL_DAY_THRESHOLD}")
    print(f"Age threshold for grouping plot: >= {AGE_THRESHOLD}")
    print("Episode-level hypothesis tests saved to: 09_episode_hypothesis_tests.csv")


if __name__ == "__main__":
    main()
