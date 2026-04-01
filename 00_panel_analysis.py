from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config
DATA_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\filtered_panel.csv")
RESULTS_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\00_panel_analysis")
D_ITEMS_PATH = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

STEP1_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\step1")
STEP1_TOP_MODEL_FEATURES_FILE = STEP1_DIR / "step1_top_model_features.csv"
STEP1_TOP_SHAP_FEATURES_FILE = STEP1_DIR / "step1_top_shap_features.csv"

DP = 3
KEEP_STATS = {"mean"}
MIN_N_PER_GROUP = 20
LATE_REMOVAL_DAY_THRESHOLD = 7
AGE_THRESHOLD = 70
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


# Create readable names like "Heart Rate [mean]" for tables.
def make_pretty_name(label: str, stat: str) -> str:
    return f"{label} [{stat}]"


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
    out = (
        df.sort_values(EPISODE_KEYS + ["day_end"])
        .reset_index(drop=False)
        .rename(columns={"index": "_orig_index"})
        .copy()
    )
    out["prior_cauti_count"] = (
        out.groupby(EPISODE_KEYS)[Y_CAUTI]
        .cumsum()
        .shift(fill_value=0)
    )
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
        "late_removal_today_on_in_rows": ACTION_COL,
        "cauti_today_on_cauti_risk_rows": Y_CAUTI,
        "reinsertion_today_on_out_fit_rows": Y_REINS,
    }
    for name, mask in masks.items():
        g = df.loc[mask].copy()
        y_col = targets[name]
        if name == "late_removal_today_on_in_rows":
            g["late_removal_today"] = ((g[ACTION_COL] == 1) & (g[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD)).astype(int)
            y_col = "late_removal_today"
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


# Run a tidy 2x2 association test and return rates, odds ratio, and p-values.
def binary_exposure_test(
    df: pd.DataFrame,
    exposure: pd.Series,
    outcome: pd.Series,
    test_name: str,
    analysis_set: str,
) -> dict[str, object]:
    tmp = pd.DataFrame({"exposure": exposure, "outcome": outcome}).dropna().copy()
    tmp["exposure"] = pd.to_numeric(tmp["exposure"], errors="coerce")
    tmp["outcome"] = pd.to_numeric(tmp["outcome"], errors="coerce")
    tmp = tmp[tmp["exposure"].isin([0, 1]) & tmp["outcome"].isin([0, 1])].copy()

    if tmp.empty:
        return {
            "test_name": test_name,
            "analysis_set": analysis_set,
            "n": 0,
            "exposed_n": 0,
            "unexposed_n": 0,
            "event_rate_exposed": np.nan,
            "event_rate_unexposed": np.nan,
            "odds_ratio": np.nan,
            "risk_difference": np.nan,
            "fisher_p": np.nan,
            "chi2_p": np.nan,
        }

    table = pd.crosstab(tmp["exposure"], tmp["outcome"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    a = int(table.loc[1, 1])
    b = int(table.loc[1, 0])
    c = int(table.loc[0, 1])
    d = int(table.loc[0, 0])

    exposed_n = int(a + b)
    unexposed_n = int(c + d)
    event_rate_exposed = a / exposed_n if exposed_n > 0 else np.nan
    event_rate_unexposed = c / unexposed_n if unexposed_n > 0 else np.nan
    risk_difference = event_rate_exposed - event_rate_unexposed if exposed_n > 0 and unexposed_n > 0 else np.nan

    fisher_p = np.nan
    odds_ratio = np.nan
    if min(a + b, c + d) > 0:
        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])

    chi2_p = np.nan
    if (table.values >= 0).all() and table.values.sum() > 0:
        try:
            chi2_p = float(stats.chi2_contingency(table.values, correction=False)[1])
        except ValueError:
            chi2_p = np.nan

    return {
        "test_name": test_name,
        "analysis_set": analysis_set,
        "n": int(len(tmp)),
        "exposed_n": exposed_n,
        "unexposed_n": unexposed_n,
        "event_rate_exposed": event_rate_exposed,
        "event_rate_unexposed": event_rate_unexposed,
        "odds_ratio": float(odds_ratio) if pd.notna(odds_ratio) else np.nan,
        "risk_difference": risk_difference,
        "fisher_p": float(fisher_p) if pd.notna(fisher_p) else np.nan,
        "chi2_p": chi2_p,
    }


# Compare a continuous variable between outcome groups using Welch and Mann-Whitney tests.
def continuous_group_test(
    df: pd.DataFrame,
    value_col: str,
    outcome_col: str,
    test_name: str,
    analysis_set: str,
) -> dict[str, object]:
    tmp = df[[value_col, outcome_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
    tmp = tmp.dropna().copy()
    tmp = tmp[tmp[outcome_col].isin([0, 1])]

    x1 = tmp.loc[tmp[outcome_col] == 1, value_col].to_numpy(dtype=float)
    x0 = tmp.loc[tmp[outcome_col] == 0, value_col].to_numpy(dtype=float)

    welch_p = np.nan
    mw_p = np.nan
    delta = np.nan

    if x1.size >= MIN_N_PER_GROUP and x0.size >= MIN_N_PER_GROUP:
        try:
            welch_p = float(stats.ttest_ind(x1, x0, equal_var=False).pvalue)
        except Exception:
            welch_p = np.nan
        try:
            mw_p = float(stats.mannwhitneyu(x1, x0, alternative="two-sided").pvalue)
        except Exception:
            mw_p = np.nan
        delta = cliffs_delta(x1, x0)

    return {
        "test_name": test_name,
        "analysis_set": analysis_set,
        "value_col": value_col,
        "n_event_1": int(x1.size),
        "n_event_0": int(x0.size),
        "mean_event_1": float(np.mean(x1)) if x1.size else np.nan,
        "mean_event_0": float(np.mean(x0)) if x0.size else np.nan,
        "median_event_1": float(np.median(x1)) if x1.size else np.nan,
        "median_event_0": float(np.median(x0)) if x0.size else np.nan,
        "welch_p": welch_p,
        "mannwhitney_p": mw_p,
        "cliffs_delta": delta,
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
    pretty_name_map: dict[str, str],
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
            "covariate": pretty_name_map.get(col, col),
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


# Save a simple bar plot of event rates by days_in_state for the main event processes.
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
    tmp = df_in.copy()
    tmp["age_ge_70"] = (tmp["age"] >= AGE_THRESHOLD).astype(int)
    tmp["late_removal_today"] = ((tmp[ACTION_COL] == 1) & (tmp[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD)).astype(int)
    rates = tmp.groupby("age_ge_70")["late_removal_today"].mean().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["<70", "≥70"], rates.fillna(0).to_numpy(dtype=float))
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
    cov_meta["pretty_name"] = cov_meta.apply(lambda r: make_pretty_name(str(r["label"]), str(r["stat"])), axis=1)
    covariate_name_map = dict(zip(cov_meta["col"], cov_meta["pretty_name"]))
    pretty_to_raw = {v: k for k, v in covariate_name_map.items()}

    numeric_cols = [TIME_COL, DAYS_COL, INTERVAL_COL, ACTION_COL, Y_CAUTI, Y_REINS, LAST_DAY_COL, "age", "hadm_id", "stay_id"] + cov_cols
    coerce_numeric(df, numeric_cols)
    df[cov_cols] = df[cov_cols].fillna(np.nan)

    covariate_count = df[cov_cols].notna().sum(axis=1).astype("int32")
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    high_covariate_count = (covariate_count >= covariate_count.median()).astype("int8")

    derived_cols = pd.DataFrame({
        "covariate_count": covariate_count,
        "any_covariate": (covariate_count > 0).astype("int8"),
        "age_ge_70": (age_numeric >= AGE_THRESHOLD).astype("int8"),
        "high_covariate_count": high_covariate_count,
    }, index=df.index)

    df = pd.concat([df, derived_cols], axis=1).copy()

    df["day_end"] = pd.to_datetime(df["day_end"], errors="coerce")
    df["inserted"] = pd.to_datetime(df["inserted"], errors="coerce")

    df = build_risk_sets(df)

    step1_top_model_features = load_step1_top_features(STEP1_TOP_MODEL_FEATURES_FILE)
    step1_top_shap_features = load_step1_top_features(STEP1_TOP_SHAP_FEATURES_FILE)

    cov_meta.sort_values(["label", "itemid", "stat"]).to_csv(RESULTS_DIR / "00_covariate_dictionary.csv", index=False)

    cohort_overview, state_overview, event_overview, risk_set_summary = build_overview_tables(df)
    cohort_overview.round(DP).to_csv(RESULTS_DIR / "01_cohort_overview.csv", index=False)
    state_overview.round(DP).to_csv(RESULTS_DIR / "02_state_overview.csv", index=False)
    event_overview.round(DP).to_csv(RESULTS_DIR / "03_event_overview.csv", index=False)
    risk_set_summary.round(DP).to_csv(RESULTS_DIR / "04_risk_set_summary.csv", index=False)

    cauti_day, reinsertion_day = build_day_rate_tables(df)
    cauti_day.round(DP).to_csv(RESULTS_DIR / "05_cauti_event_rates_by_state_and_day.csv", index=False)
    reinsertion_day.round(DP).to_csv(RESULTS_DIR / "06_reinsertion_event_rates_by_day.csv", index=False)

    # Simple descriptive tables aligned to current panel structure.
    in_rows = df[df[STATE_COL] == "in"].copy()
    out_rows = df[df[STATE_COL] == "out"].copy()
    cauti_rows = df[df["cauti_risk_row"] == 1].copy()
    out_fit_rows = df[df["reinsertion_fit_row"] == 1].copy()

    desc_rows = []
    for name, g in [
        ("overall", df),
        ("in_rows", in_rows),
        ("out_rows", out_rows),
        ("cauti_risk_rows", cauti_rows),
        ("reinsertion_fit_rows", out_fit_rows),
    ]:
        desc_rows.append({
            "analysis_set": name,
            "rows": int(len(g)),
            "mean_age": float(pd.to_numeric(g["age"], errors="coerce").mean()),
            "median_age": float(pd.to_numeric(g["age"], errors="coerce").median()),
            "mean_days_in_state": float(pd.to_numeric(g[DAYS_COL], errors="coerce").mean()),
            "median_days_in_state": float(pd.to_numeric(g[DAYS_COL], errors="coerce").median()),
            "mean_interval_hours": float(pd.to_numeric(g[INTERVAL_COL], errors="coerce").mean()),
            "mean_covariate_count": float(pd.to_numeric(g["covariate_count"], errors="coerce").mean()),
        })
    pd.DataFrame(desc_rows).round(DP).to_csv(RESULTS_DIR / "07_core_descriptives.csv", index=False)

    # Example binary hypothesis tests for supervisor discussion.
    binary_tests = pd.DataFrame([
        binary_exposure_test(
            in_rows,
            in_rows["age_ge_70"],
            ((in_rows[ACTION_COL] == 1) & (in_rows[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD)).astype(int),
            test_name=f"Age ≥ {AGE_THRESHOLD} vs late removal today (day ≥ {LATE_REMOVAL_DAY_THRESHOLD})",
            analysis_set="in_rows",
        ),
        binary_exposure_test(
            in_rows,
            in_rows["age_ge_70"],
            in_rows[ACTION_COL],
            test_name=f"Age ≥ {AGE_THRESHOLD} vs removal today",
            analysis_set="in_rows",
        ),
        binary_exposure_test(
            cauti_rows,
            cauti_rows["age_ge_70"],
            cauti_rows[Y_CAUTI],
            test_name=f"Age ≥ {AGE_THRESHOLD} vs CAUTI today",
            analysis_set="cauti_risk_rows",
        ),
        binary_exposure_test(
            out_fit_rows,
            out_fit_rows["age_ge_70"],
            out_fit_rows[Y_REINS],
            test_name=f"Age ≥ {AGE_THRESHOLD} vs reinsertion today",
            analysis_set="reinsertion_fit_rows",
        ),
        binary_exposure_test(
            in_rows,
            in_rows["high_covariate_count"],
            ((in_rows[ACTION_COL] == 1) & (in_rows[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD)).astype(int),
            test_name=f"High covariate count vs late removal today (day ≥ {LATE_REMOVAL_DAY_THRESHOLD})",
            analysis_set="in_rows",
        ),
    ])
    binary_tests["fisher_p_adj_bh"] = p_adjust_bh(binary_tests["fisher_p"])
    binary_tests.round(DP).to_csv(RESULTS_DIR / "08_binary_hypothesis_tests.csv", index=False)

    # Continuous-variable examples for the same outcomes.
    continuous_tests = pd.DataFrame([
        continuous_group_test(
            in_rows.assign(late_removal_today=((in_rows[ACTION_COL] == 1) & (in_rows[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD)).astype(int)),
            value_col="age",
            outcome_col="late_removal_today",
            test_name=f"Age by late removal today (day ≥ {LATE_REMOVAL_DAY_THRESHOLD})",
            analysis_set="in_rows",
        ),
        continuous_group_test(
            in_rows,
            value_col="age",
            outcome_col=ACTION_COL,
            test_name="Age by removal today",
            analysis_set="in_rows",
        ),
        continuous_group_test(
            cauti_rows,
            value_col="age",
            outcome_col=Y_CAUTI,
            test_name="Age by CAUTI today",
            analysis_set="cauti_risk_rows",
        ),
        continuous_group_test(
            out_fit_rows,
            value_col="age",
            outcome_col=Y_REINS,
            test_name="Age by reinsertion today",
            analysis_set="reinsertion_fit_rows",
        ),
        continuous_group_test(
            in_rows.assign(late_removal_today=((in_rows[ACTION_COL] == 1) & (in_rows[DAYS_COL] >= LATE_REMOVAL_DAY_THRESHOLD)).astype(int)),
            value_col="covariate_count",
            outcome_col="late_removal_today",
            test_name=f"Covariate count by late removal today (day ≥ {LATE_REMOVAL_DAY_THRESHOLD})",
            analysis_set="in_rows",
        ),
    ])
    continuous_tests["mannwhitney_p_adj_bh"] = p_adjust_bh(continuous_tests["mannwhitney_p"])
    continuous_tests.round(DP).to_csv(RESULTS_DIR / "09_continuous_hypothesis_tests.csv", index=False)

    # Univariable covariate screening, top signals only, to keep the output tidy.
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
    pd.concat([top_removal, top_reins, top_cauti], ignore_index=True).round(DP).to_csv(
        RESULTS_DIR / "10_top_univariable_covariate_signals.csv", index=False
    )

    # Describe the influential features loaded from Step 1 model and SHAP CSVs.
    xgb_desc_parts = []

    for feature_source, step1_features in [
        ("model_importance", step1_top_model_features),
        ("shap_importance", step1_top_shap_features),
    ]:
        removal_feature_names = get_step1_feature_names(step1_features, "removal")
        cauti_feature_names = get_step1_feature_names(step1_features, "cauti")
        reinsertion_feature_names = get_step1_feature_names(step1_features, "reinsertion")

        removal_top_cols = resolve_feature_cols(removal_feature_names, pretty_to_raw, set(df.columns))
        cauti_top_cols = resolve_feature_cols(cauti_feature_names, pretty_to_raw, set(df.columns))
        reinsertion_top_cols = resolve_feature_cols(reinsertion_feature_names, pretty_to_raw, set(df.columns))

        xgb_desc_parts.append(
            describe_selected_covariates(
                df=in_rows,
                value_cols=removal_top_cols,
                pretty_name_map=covariate_name_map,
                analysis_set_name="in_rows",
                model_name="removal",
                feature_source=feature_source,
                dp=DP,
            )
        )
        xgb_desc_parts.append(
            describe_selected_covariates(
                df=cauti_rows,
                value_cols=cauti_top_cols,
                pretty_name_map=covariate_name_map,
                analysis_set_name="cauti_risk_rows",
                model_name="cauti",
                feature_source=feature_source,
                dp=DP,
            )
        )
        xgb_desc_parts.append(
            describe_selected_covariates(
                df=out_fit_rows,
                value_cols=reinsertion_top_cols,
                pretty_name_map=covariate_name_map,
                analysis_set_name="reinsertion_fit_rows",
                model_name="reinsertion",
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

    summary = {
        "data_file": str(DATA_FILE),
        "results_dir": str(RESULTS_DIR),
        "step1_top_model_features_file": str(STEP1_TOP_MODEL_FEATURES_FILE),
        "step1_top_shap_features_file": str(STEP1_TOP_SHAP_FEATURES_FILE),
        "n_rows": int(len(df)),
        "n_patients": int(df[ID_COL].nunique()),
        "n_episodes": int(df[EPISODE_KEYS].drop_duplicates().shape[0]),
        "n_covariates": int(len(cov_cols)),
        "late_removal_threshold_day": int(LATE_REMOVAL_DAY_THRESHOLD),
        "age_threshold": int(AGE_THRESHOLD),
        "outputs": [
            "00_covariate_dictionary.csv",
            "01_cohort_overview.csv",
            "02_state_overview.csv",
            "03_event_overview.csv",
            "04_risk_set_summary.csv",
            "05_cauti_event_rates_by_state_and_day.csv",
            "06_reinsertion_event_rates_by_day.csv",
            "07_core_descriptives.csv",
            "08_binary_hypothesis_tests.csv",
            "09_continuous_hypothesis_tests.csv",
            "10_top_univariable_covariate_signals.csv",
            "11_xgb_influential_covariate_descriptives.csv",
            "plot_event_rates_by_days_in_state.png",
            "plot_late_removal_by_age_group.png",
        ],
    }
    (RESULTS_DIR / "analysis_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Outputs saved to: {RESULTS_DIR}")
    print(f"[INFO] N covariates analysed: {len(cov_cols)}")
    print(f"[INFO] Late removal threshold: day >= {LATE_REMOVAL_DAY_THRESHOLD}")
    print(f"[INFO] Age threshold for example tests: >= {AGE_THRESHOLD}")


if __name__ == "__main__":
    main()