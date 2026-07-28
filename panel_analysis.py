# Build descriptive summaries for the catheter modelling panel.
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config
DATA_FILE = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data\modelling_panel.csv")
RESULTS_DIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\panel_analysis")
COVARIATE_DICT_FILE = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data\covariate_dictionary.csv")

STEP1_DIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step1")
STEP1_TOP_MODEL_FEATURES_FILE = STEP1_DIR / "top_model_features.csv"
STEP1_TOP_SHAP_FEATURES_FILE = STEP1_DIR / "top_shap_features.csv"

DP = 3
KEEP_STATS = {"mean"}
MIN_N_PER_GROUP = 20
LATE_REMOVAL_DAY_THRESHOLD = 7
AGE_THRESHOLD = 60
TOP_N_COVARIATES = 20

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
INTERVAL_COL = "interval_hours"
ACTION_COL = "removed_in_period"
DECISION_ROW_COL = "is_decision_row"
DECISION_PERIOD_COL = "catheter_period_at_decision"
Y_CAUTI = "cauti_in_period"
Y_REINS = "reinsertion_in_period"
LAST_PERIOD_COL = "is_last_period_of_episode"
END_REASON_COL = "episode_end_reason"
EPISODE_KEYS = ["stay_id", "inserted"]
POST_REMOVE_RISK_PERIODS = 2
LATE_REMOVAL_COL = "late_removal_in_period"
AGE_GROUP_COL = "age_ge_threshold"

REMOVAL_FEATURE = "GCS - Verbal Response [mean]"
CAUTI_BINARY_FEATURE = "sex_M"
CAUTI_CONTINUOUS_FEATURE = "Anion gap [mean]"
REINSERTION_FEATURE = "Bladder Scan Estimate [mean]"


# Detect columns like itemid_<ID>__mean/min/max and return the matching columns plus metadata.
def detect_covariate_cols(columns, keep_stats):
    # Detect covariate columns.
    pat = re.compile(r"^itemid_(\d+)__(mean|min|max)$", flags=re.IGNORECASE)
    cov_cols = []
    meta = []
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


# Coerce a set of columns to numeric, handling TRUE/FALSE strings.
def coerce_numeric(df, cols):
    # Coerce numeric.
    for col in cols:
        if df[col].dtype == object:
            df[col] = df[col].replace({
                "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0,
                "true": 1, "false": 0,
            })
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Load a Step 1 top-features table saved by 01_step1_transition_models.py.
def load_step1_top_features(path):
    # Load step1 top features.
    df = pd.read_csv(path)

    out = df.copy()
    out["model"] = out["model"].astype(str).str.strip().str.lower()
    out["feature"] = out["feature"].astype(str).str.strip()

    sort_cols = [c for c in ["model", "rank", "feature"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


# Return the ordered feature names for one model from a Step 1 feature table.
def get_step1_feature_names(step1_features, model_name):
    # Get step1 feature names.
    tmp = step1_features[step1_features["model"] == model_name].copy()

    if "rank" in tmp.columns:
        tmp = tmp.sort_values(["rank", "feature"])
    else:
        tmp = tmp.sort_values(["feature"])

    feature_names = []
    seen = set()

    for feature in tmp["feature"].tolist():
        if feature not in seen:
            feature_names.append(feature)
            seen.add(feature)

    return feature_names


# Calculate Cliff's delta for two groups as a simple effect-size summary.
def cliffs_delta(x1, x0):
    # Calculate Cliff's delta.
    xy = np.concatenate([x1, x0])
    ranks = stats.rankdata(xy)
    rx = ranks[: x1.size].sum()
    u = rx - x1.size * (x1.size + 1) / 2
    delta = (2 * u) / (x1.size * x0.size) - 1
    return float(delta)


# Apply Benjamini-Hochberg correction to a list/series of p-values.
def p_adjust_bh(pvalues):
    # Calculate adjust BH.
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
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
def build_risk_sets(df):
    # Mark which rows belong to each event process.
    # Build risk sets.
    out = df.sort_values(EPISODE_KEYS + ["period_end"]).copy()
    y_cauti = pd.to_numeric(out[Y_CAUTI], errors="coerce").fillna(0)
    out["prior_cauti_count"] = out.groupby(EPISODE_KEYS)[Y_CAUTI].cumsum() - y_cauti
    out[DECISION_PERIOD_COL] = (
        out[STATE_COL].eq("in").astype(int).groupby(
            [out[key] for key in EPISODE_KEYS], dropna=False
        ).cumsum()
    )
    out["cauti_risk_row"] = (
        (out[STATE_COL] == "in") |
        ((out[STATE_COL] == "out") & (out[PERIODS_COL] <= POST_REMOVE_RISK_PERIODS))
    ).astype(int)
    out["reinsertion_fit_row"] = (
        (out[STATE_COL] == "out") &
        ~(
            (out[LAST_PERIOD_COL] == 1) &
            (out[END_REASON_COL] == "icu_end") &
            (out[Y_REINS] == 0)
        )
    ).astype(int)
    out["removal_fit_row"] = (
        (out[DECISION_ROW_COL] == 1) &
        (out["prior_cauti_count"] == 0)
    ).astype(int)
    return out


# Convert the current panel structure into tidy overview tables suitable for sharing.
def build_overview_tables(df):
    # Count rows and episodes at cohort level.
    # Build overview tables.
    cohort_overview = pd.DataFrame([{
        "subset": "overall",
        "rows": int(len(df)),
        "unique_subject_id": int(df[ID_COL].nunique()),
        "unique_hadm_id": int(df["hadm_id"].nunique()),
        "unique_stay_id": int(df["stay_id"].nunique()),
        "unique_episodes": int(df[EPISODE_KEYS].drop_duplicates().shape[0]),
    }])

    # Summarise rows by catheter state.
    state_overview = (
        df.groupby(STATE_COL, dropna=False)
        .agg(
            rows=(STATE_COL, "size"),
            unique_subject_id=(ID_COL, "nunique"),
            unique_episodes=("stay_id", "nunique"),
        )
        .reset_index()
    )

    # Summarise event counts on the relevant row subsets.
    event_rows = []
    masks = {
        "removal_in_period_on_decision_rows": df[DECISION_ROW_COL] == 1,
        "late_removal_in_period_on_decision_rows": (
            (df[DECISION_ROW_COL] == 1) &
            (df[DECISION_PERIOD_COL] >= LATE_REMOVAL_DAY_THRESHOLD)
        ),
        "cauti_in_period_on_cauti_risk_rows": df["cauti_risk_row"] == 1,
        "reinsertion_in_period_on_out_fit_rows": df["reinsertion_fit_row"] == 1,
    }
    targets = {
        "removal_in_period_on_decision_rows": ACTION_COL,
        "late_removal_in_period_on_decision_rows": LATE_REMOVAL_COL,
        "cauti_in_period_on_cauti_risk_rows": Y_CAUTI,
        "reinsertion_in_period_on_out_fit_rows": Y_REINS,
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

    # Record the main row sets used downstream.
    risk_set_summary = pd.DataFrame([
        {"metric": "all_rows", "value": int(len(df))},
        {"metric": "in_rows", "value": int((df[STATE_COL] == "in").sum())},
        {"metric": "out_rows", "value": int((df[STATE_COL] == "out").sum())},
        {
            "metric": "out_rows_periods_in_state_le_2",
            "value": int(((df[STATE_COL] == "out") & (df[PERIODS_COL] <= POST_REMOVE_RISK_PERIODS)).sum()),
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


# Collapse the panel to one row per catheter episode for simple episode-level testing.
def build_episode_level_table(
    df,
    removal_feature,
    cauti_binary_feature,
    cauti_continuous_feature,
    reinsertion_feature,
):
    # Collapse panel periods down to one row per episode.
    # Build episode level table.
    d = df.sort_values(EPISODE_KEYS + ["period_end"]).copy()

    in_rows = d.loc[d[STATE_COL] == "in"].copy()
    decision_rows = d.loc[d[DECISION_ROW_COL] == 1].copy()
    cauti_rows = d.loc[d["cauti_risk_row"] == 1].copy()
    reinsertion_rows = d.loc[d["reinsertion_fit_row"] == 1].copy()

    # Pull one summary value per episode.
    episode_age = d.groupby(EPISODE_KEYS, dropna=False)["age"].first().rename("age")
    episode_subject = d.groupby(EPISODE_KEYS, dropna=False)[ID_COL].first().rename(ID_COL)
    episode_hadm = d.groupby(EPISODE_KEYS, dropna=False)["hadm_id"].first().rename("hadm_id")

    catheter_days = (
        in_rows.groupby(EPISODE_KEYS, dropna=False)[PERIODS_COL]
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
        decision_rows.groupby(EPISODE_KEYS, dropna=False)[removal_feature]
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

    # Combine the episode summaries into one table.
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
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["late_removal_episode", "cauti_episode", "reinsertion_episode", cauti_binary_feature]:
        out[col] = out[col].fillna(0).clip(0, 1).astype("int8")

    out = out.dropna(subset=["catheter_days"]).copy()
    return out


# Mann-Whitney U test for a continuous value across a binary episode-level outcome.
def mannwhitney_group_test(
    df,
    value_col,
    group_col,
    test_name,
):
    # Calculate Mann-Whitney group test.
    tmp = df[[value_col, group_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp[group_col] = pd.to_numeric(tmp[group_col], errors="coerce")
    tmp = tmp.dropna().copy()
    tmp = tmp[tmp[group_col].isin([0, 1])]

    x1 = tmp.loc[tmp[group_col] == 1, value_col].to_numpy(dtype=float)
    x0 = tmp.loc[tmp[group_col] == 0, value_col].to_numpy(dtype=float)

    u_stat, p_value = stats.mannwhitneyu(x1, x0, alternative="two-sided")
    u_stat = float(u_stat)
    p_value = float(p_value)
    # Calculate Cliff's delta.
    delta = cliffs_delta(x1, x0)

    return {
        "test_name": test_name,
        "test_type": "Mann-Whitney U",
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
    df,
    exposure_col,
    outcome_col,
    test_name,
):
    # Convert group test.
    tmp = df[[exposure_col, outcome_col]].copy()
    tmp[exposure_col] = pd.to_numeric(tmp[exposure_col], errors="coerce")
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
    tmp = tmp.dropna().copy()
    tmp = tmp[tmp[exposure_col].isin([0, 1]) & tmp[outcome_col].isin([0, 1])]

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

    odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])
    odds_ratio = float(odds_ratio)
    fisher_p = float(fisher_p)

    chi2_p = float(stats.chi2_contingency(table.values, correction=False)[1])

    return {
        "test_name": test_name,
        "test_type": "Fisher exact",
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
    df,
    covariates,
    outcome_col,
    analysis_set,
    top_n,
):
    # Rank raw covariates by univariable association strength.
    # Build covariate screen.
    rows = []
    d = df.copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    d = d[d[outcome_col].isin([0, 1])].copy()

    # Calculate Cliff's delta.
    for col in covariates:
        x1 = pd.to_numeric(d.loc[d[outcome_col] == 1, col], errors="coerce").dropna().to_numpy(dtype=float)
        x0 = pd.to_numeric(d.loc[d[outcome_col] == 0, col], errors="coerce").dropna().to_numpy(dtype=float)
        mw_p = float(stats.mannwhitneyu(x1, x0, alternative="two-sided").pvalue)
        # Calculate Cliff's delta.
        rows.append({
            "analysis_set": analysis_set,
            "outcome": outcome_col,
            "covariate": col,
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
    # Calculate adjust BH.
    out["mannwhitney_p_adj_bh"] = p_adjust_bh(out["mannwhitney_p"])
    out = out.sort_values(["mannwhitney_p_adj_bh", "mannwhitney_p", "covariate"]).head(top_n).reset_index(drop=True)
    return out


# Resolve a list of feature names into actual dataframe columns.
def resolve_feature_cols(
    feature_names,
    feature_lookup,
    available_cols,
):
    # Map saved Step 1 names back to panel columns.
    # Resolve feature columns.
    cols = []
    seen = set()

    for name in feature_names:
        raw = feature_lookup.get(name, name)
        if raw in available_cols and raw not in seen:
            cols.append(raw)
            seen.add(raw)

    return cols


# Return describe-style summary rows for selected covariates/features.
def describe_selected_covariates(
    df,
    value_cols,
    analysis_set_name,
    model_name,
    feature_source,
    dp=3,
):
    # Build descriptive stats for the selected model features.
    # Describe selected covariates.
    rows = []

    for col in value_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        n = int(s.shape[0])

        rows.append({
            "model": model_name,
            "feature_source": feature_source,
            "analysis_set": analysis_set_name,
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
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(dp)
    return out


def save_csv(df, path, round_dp=None, sci_cols=None):
    # Apply output formatting only at save time.
    # Save CSV.
    out = df.copy()
    if round_dp is not None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].round(round_dp)
    for col in sci_cols or []:
        out[col] = df[col].map(lambda x: f"{x:.3e}" if pd.notna(x) else "")
    float_format = f"%.{round_dp}f" if round_dp is not None else None
    out.to_csv(path, index=False, float_format=float_format)


# Save a simple line plot of event rates by periods_in_state for the main event processes.
def plot_event_rates(cauti_period, reinsertion_period, outdir):
    # Plot event rates.
    fig, ax = plt.subplots(figsize=(8, 5))
    in_rows = cauti_period[cauti_period[STATE_COL] == "in"]
    out_rows = cauti_period[cauti_period[STATE_COL] == "out"]
    ax.plot(in_rows[PERIODS_COL], in_rows["event_rate"], marker="o", label="CAUTI risk set: IN")
    ax.plot(out_rows[PERIODS_COL], out_rows["event_rate"], marker="o", label="CAUTI risk set: OUT")
    ax.plot(reinsertion_period[PERIODS_COL], reinsertion_period["event_rate"], marker="o", label="Reinsertion on OUT fit rows")
    ax.set_xlabel("periods_in_state")
    ax.set_ylabel("event rate")
    ax.set_title("Event rates by periods_in_state")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot_event_rates_by_periods_in_state.png", dpi=150)
    plt.close(fig)


# Run the merged panel diagnostics and supervisor-facing descriptive/inferential analysis.
def main():
    # Run the script workflow.
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Load and standardise the panel.
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()

    # Detect the itemid covariates used in the analysis and load their saved labels.
    cov_cols, _ = detect_covariate_cols(df.columns.tolist(), KEEP_STATS)
    cov_meta = pd.read_csv(COVARIATE_DICT_FILE)
    cov_meta["description"] = cov_meta["label"].astype(str) + " [mean]"
    feature_lookup = dict(zip(cov_meta["description"], cov_meta["col"]))
    feature_lookup.update({f"{desc} [missing]": f"{col}__missing" for col, desc in zip(cov_meta["col"], cov_meta["description"])})

    # Coerce the core numeric inputs.
    numeric_cols = [
        TIME_COL, PERIODS_COL, INTERVAL_COL, ACTION_COL, DECISION_ROW_COL, Y_CAUTI, Y_REINS,
        LAST_PERIOD_COL, "age", "hadm_id", "stay_id", CAUTI_BINARY_FEATURE,
    ] + cov_cols
    # Coerce numeric.
    coerce_numeric(df, numeric_cols)

    # Create a few derived panel features.
    covariate_count = df[cov_cols].notna().sum(axis=1).astype("int32")
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    high_covariate_count = (covariate_count.astype(float) >= float(covariate_count.median())).astype("int8")


    derived_cols = pd.DataFrame({
        "covariate_count": covariate_count,
        AGE_GROUP_COL: (age_numeric >= AGE_THRESHOLD).astype("int8"),
        "high_covariate_count": high_covariate_count,
    }, index=df.index)

    df = pd.concat([df, derived_cols], axis=1).copy()

    # Normalise datetime fields for grouping.
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["inserted"] = pd.to_datetime(df["inserted"], errors="coerce")

    # Build risk sets.
    df = build_risk_sets(df)
    df[LATE_REMOVAL_COL] = (
        (df[ACTION_COL] == 1) &
        (df[DECISION_PERIOD_COL] >= LATE_REMOVAL_DAY_THRESHOLD)
    ).astype("int8")

    # Load the Step 1 feature lists used later.
    step1_top_model_features = load_step1_top_features(STEP1_TOP_MODEL_FEATURES_FILE)
    # Load step1 top features.
    step1_top_shap_features = load_step1_top_features(STEP1_TOP_SHAP_FEATURES_FILE)

    removal_feature = feature_lookup.get(REMOVAL_FEATURE, REMOVAL_FEATURE)
    cauti_binary_feature = feature_lookup.get(CAUTI_BINARY_FEATURE, CAUTI_BINARY_FEATURE)
    cauti_continuous_feature = feature_lookup.get(CAUTI_CONTINUOUS_FEATURE, CAUTI_CONTINUOUS_FEATURE)
    reinsertion_feature = feature_lookup.get(REINSERTION_FEATURE, REINSERTION_FEATURE)

    # Save the main overview tables.
    cohort_overview, state_overview, event_overview, risk_set_summary = build_overview_tables(df)
    # Save CSV.
    for name, table in [
        ("01_cohort_overview.csv", cohort_overview),
        ("02_state_overview.csv", state_overview),
        ("03_event_overview.csv", event_overview),
        ("04_risk_set_summary.csv", risk_set_summary),
    ]:
        # Save CSV.
        save_csv(table, RESULTS_DIR / name, round_dp=DP)

    # Summarise event rates by period in state.
    cauti_period = (
        df.loc[df["cauti_risk_row"] == 1]
        .groupby([STATE_COL, PERIODS_COL], dropna=False)[Y_CAUTI]
        .agg(rows="count", events="sum", event_rate="mean")
        .reset_index()
    )
    reinsertion_period = (
        df.loc[df["reinsertion_fit_row"] == 1]
        .groupby(PERIODS_COL, dropna=False)[Y_REINS]
        .agg(rows="count", events="sum", event_rate="mean")
        .reset_index()
    )
    # Save CSV.
    save_csv(cauti_period, RESULTS_DIR / "05_cauti_event_rates_by_state_and_period.csv", round_dp=DP)
    # Save CSV.
    save_csv(reinsertion_period, RESULTS_DIR / "06_reinsertion_event_rates_by_period.csv", round_dp=DP)

    # Reuse these row subsets across later outputs.
    analysis_sets = {
        "overall": df,
        "in_rows": df.loc[df[STATE_COL] == "in"],
        "decision_rows": df.loc[df[DECISION_ROW_COL] == 1],
        "out_rows": df.loc[df[STATE_COL] == "out"],
        "cauti_risk_rows": df.loc[df["cauti_risk_row"] == 1],
        "reinsertion_fit_rows": df.loc[df["reinsertion_fit_row"] == 1],
    }
    decision_rows = analysis_sets["decision_rows"]
    cauti_rows = analysis_sets["cauti_risk_rows"]
    out_fit_rows = analysis_sets["reinsertion_fit_rows"]

    # Describe the Step 1 features highlighted by the models.
    model_describe_configs = [
        ("removal", "decision_rows"),
        ("cauti", "cauti_risk_rows"),
        ("reinsertion", "reinsertion_fit_rows"),
    ]
    available_cols = set(df.columns)
    # Save CSV.
    save_csv(
        pd.concat(
            [
                describe_selected_covariates(
                    analysis_sets[analysis_set_name],
                    resolve_feature_cols(get_step1_feature_names(step1_features, model_name), feature_lookup, available_cols),
                    analysis_set_name,
                    model_name,
                    feature_source,
                    DP,
                )
                for feature_source, step1_features in [
                    ("model", step1_top_model_features),
                    ("shap", step1_top_shap_features),
                ]
                for model_name, analysis_set_name in model_describe_configs
            ],
            ignore_index=True,
        ),
        RESULTS_DIR / "07_xgb_influential_covariate_descriptives.csv",
    )

    # Build and save the episode-level table.
    episode_df = build_episode_level_table(
        df=df,
        removal_feature=removal_feature,
        cauti_binary_feature=cauti_binary_feature,
        cauti_continuous_feature=cauti_continuous_feature,
        reinsertion_feature=reinsertion_feature,
    )
    # Save CSV.
    save_csv(episode_df, RESULTS_DIR / "08_episode_level_analysis_table.csv", round_dp=DP)

    # Run the episode-level hypothesis tests.
    episode_tests = pd.DataFrame([
        fn(episode_df, **kwargs)
        for fn, kwargs in [
            (
                mannwhitney_group_test,
                {
                    "value_col": removal_feature,
                    "group_col": "late_removal_episode",
                    "test_name": f"{REMOVAL_FEATURE} by late removal episode (catheter days >= {LATE_REMOVAL_DAY_THRESHOLD})",
                },
            ),
            (
                binary_group_test,
                {
                    "exposure_col": cauti_binary_feature,
                    "outcome_col": "cauti_episode",
                    "test_name": f"{CAUTI_BINARY_FEATURE} by CAUTI episode",
                },
            ),
            (
                mannwhitney_group_test,
                {
                    "value_col": cauti_continuous_feature,
                    "group_col": "cauti_episode",
                    "test_name": f"{CAUTI_CONTINUOUS_FEATURE} by CAUTI episode",
                },
            ),
            (
                mannwhitney_group_test,
                {
                    "value_col": reinsertion_feature,
                    "group_col": "reinsertion_episode",
                    "test_name": f"{REINSERTION_FEATURE} by reinsertion episode",
                },
            ),
        ]
    ])
    # Calculate adjust BH.
    episode_tests["p_value_adj_bh"] = p_adjust_bh(episode_tests["p_value"])
    episode_tests_export = episode_tests.copy()
    episode_test_numeric_cols = episode_tests_export.select_dtypes(include=[np.number]).columns
    episode_tests_export[episode_test_numeric_cols] = episode_tests_export[episode_test_numeric_cols].round(DP)
    for col in ["p_value", "p_value_adj_bh"]:
        episode_tests_export[col] = episode_tests[col].map(lambda x: f"{x:.3e}" if pd.notna(x) else "")
    episode_tests_export = episode_tests_export.set_index("test_name").T.reset_index().rename(columns={"index": "metric"})
    episode_tests_export.to_csv(RESULTS_DIR / "09_episode_hypothesis_tests.csv", index=False)

    # Rank the top univariable raw covariate signals.
    top_signals = pd.concat(
        [
            top_covariate_screen(df_slice, cov_cols, outcome_col, analysis_set, TOP_N_COVARIATES)
            for df_slice, outcome_col, analysis_set in [
                (decision_rows, ACTION_COL, "decision_rows_removal_in_period"),
                (out_fit_rows, Y_REINS, "out_fit_rows_reinsertion_in_period"),
                (cauti_rows, Y_CAUTI, "cauti_risk_rows_cauti_in_period"),
            ]
        ],
        ignore_index=True,
    )
    # Save CSV.
    save_csv(
        top_signals,
        RESULTS_DIR / "10_top_univariable_covariate_signals.csv",
        round_dp=DP,
        sci_cols=["mannwhitney_p", "mannwhitney_p_adj_bh"],
    )

    # Save the summary figures.
    plot_event_rates(cauti_period, reinsertion_period, RESULTS_DIR)

    print(f"Outputs saved to: {RESULTS_DIR}")
    print(f"Number of covariates analysed: {len(cov_cols)}")
    print(f"Late removal threshold: period >= {LATE_REMOVAL_DAY_THRESHOLD}")
    
# Run the script workflow.
if __name__ == "__main__":
    # Run the script workflow.
    main()
