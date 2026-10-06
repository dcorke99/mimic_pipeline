import re
import os
from collections import defaultdict
from pathlib import Path

# Data-processing libraries
import numpy as np
import pandas as pd

# Define pipeline directories
REPO_ROOT = Path(__file__).resolve().parent
# Raw data lives outside the repository by default. Override with MIMIC_DIR
MIMIC_DIR = Path(os.environ.get("MIMIC_DIR") or REPO_ROOT.parent / "Data" / "MIMIC-IV" / "mimic-iv-3.1").expanduser()
if not MIMIC_DIR.is_absolute():
    MIMIC_DIR = REPO_ROOT / MIMIC_DIR
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"

# Limit memory use per read
CHUNK_ROWS = 1_000_000

# Define cohort timing
FOLEY_ITEMID = 229351
MIN_EPISODE_DURATION = pd.Timedelta(hours=24)
PERIOD_DURATION = pd.Timedelta(hours=24)
LOOKBACK_DURATION = pd.Timedelta(hours=24)
POST_REMOVE_RISK_PERIODS = 2

# Define temperature normalisation
FAHRENHEIT_UNITS = {"F", "DEG F", "DEGREES F", "Â°F", "Â° F"}
CELSIUS_UNIT = "Â°C"
TEMP_F_ITEMID = 223761
TEMP_C_ITEMID = 223762

# Configure value cleaning
ITEM_COL = "itemid"
VALUE_COL = "valuenum"
MIN_N_FOR_RULES = 100
ZERO_MAX_FRAC = 0.10
FAR_OUT_SPREAD_MULT = 3.0

# Select chart summaries
AGG_STATS = [
    "count",
    "mean",
    "std",
    "last",
    "slope_per_hour",
]

# Configure feature retention
MIN_ROW_COVERAGE = 0.05
MIN_STAY_COVERAGE = 0.10
ROUND_DP = 3

# Name modelling columns
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
ACTION_COL = "removed_in_period"
Y_CAUTI = "cauti_in_period"
Y_REINS = "reinsertion_in_period"
Y_DEATH = "death_in_period"
Y_ICU_EXIT_ALIVE = "icu_exit_alive_in_period"
OBSERVED_ACTION_COL = "observed_action"
TERMINAL_REASONS = {"death", "reinsertion", "icu_exit_alive"}
ETHNICITY_UNAVAILABLE_MARKERS = (
    "UNKNOWN",
    "UNABLE TO OBTAIN",
    "PATIENT DECLINED",
    "DECLINED",
    "NOT SPECIFIED",
    "NOT RECORDED",
    "NOT REPORTED",
    "NOT AVAILABLE",
    "UNAVAILABLE",
    "UNOBTAINABLE",
    "NO INFORMATION",
)

def load_item_labels():
    # Load one label per item
    d_items_df = pd.read_csv(
        MIMIC_DIR / "icu" / "d_items.csv",
        usecols=["itemid", "label"],
        low_memory=False,
    ).drop_duplicates("itemid")

    # Remove missing item IDs
    d_items_df["itemid"] = pd.to_numeric(d_items_df["itemid"])
    d_items_df = d_items_df.dropna(subset=["itemid"]).copy()
    d_items_df["itemid"] = d_items_df["itemid"].astype(int)
    d_items_df["label"] = d_items_df["label"].astype(str)

    # Return an ID lookup
    return d_items_df.set_index("itemid")["label"].to_dict()

def print_section(title):
    # Print a stage heading
    print()
    print(f"=== {title} ===")

def normalise_ethnicity_text(value):
    # Standardise whitespace and case for matching
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()

def ethnicity_is_missing(value):
    # Identify unavailable ethnicity information
    ethnicity_text = normalise_ethnicity_text(value)
    return not ethnicity_text or any(
        marker in ethnicity_text for marker in ETHNICITY_UNAVAILABLE_MARKERS
    )

def map_ethnicity_group(value):
    # Collapse substantive raw ethnicity labels
    if ethnicity_is_missing(value):
        return pd.NA

    # Match broad groups
    ethnicity_text = normalise_ethnicity_text(value)
    if "WHITE" in ethnicity_text:
        return "White"
    if "BLACK" in ethnicity_text:
        return "Black"
    if "ASIAN" in ethnicity_text:
        return "Asian"
    if "HISPANIC" in ethnicity_text or "LATIN" in ethnicity_text:
        return "Hispanic"
    return "Other"

def build_ethnicity_mapping_audit(episodes):
    # Summarise source ethnicity mappings in the episode cohort
    audit = episodes[["ethnicity", "ethnicity_group", "ethnicity_missing"]].copy()
    audit["source_ethnicity"] = (
        audit["ethnicity"].astype("string").str.strip().replace("", pd.NA)
        .fillna("<MISSING>")
    )
    audit["mapped_ethnicity_category"] = (
        audit["ethnicity_group"].astype("string").fillna("Unavailable")
    )
    return (
        audit.groupby(
            ["source_ethnicity", "mapped_ethnicity_category", "ethnicity_missing"],
            dropna=False,
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
        .sort_values(
            ["ethnicity_missing", "mapped_ethnicity_category", "source_ethnicity"]
        )
        .reset_index(drop=True)
    )

def merge_overlapping_foley_events(df):
    # Merge overlapping Foley records
    episodes = []

    # Process each ICU stay separately
    for stay_id, stay_events in df.groupby("stay_id"):
        stay_events = stay_events.sort_values("inserted")
        current_start = None
        current_end = None

        for event in stay_events.itertuples():
            # Start the first interval
            if current_start is None:
                current_start = event.inserted
                current_end = event.removed
                continue

            # Extend an overlapping interval
            if event.inserted <= current_end:
                current_end = max(current_end, event.removed)
            else:
                # Close a completed interval
                episodes.append((stay_id, current_start, current_end))
                current_start = event.inserted
                current_end = event.removed

        # Close the final interval
        if current_start is not None:
            episodes.append((stay_id, current_start, current_end))

    # Return one row per episode
    return pd.DataFrame(episodes, columns=["stay_id", "inserted", "removed"])

def add_episode_endpoints(catheterised):
    # End each catheter episode at its first absorbing or episode-closing event
    out = catheterised.copy()
    endpoint_candidates = out[["death_time", "reinsertion_time", "ICU_out"]]
    out["episode_end_time"] = endpoint_candidates.min(axis=1)

    # Reject impossible trajectories rather than silently creating empty windows
    invalid_end = (
        out["episode_end_time"].isna()
        | out["inserted"].isna()
        | out["episode_end_time"].le(out["inserted"])
    )
    if invalid_end.any():
        examples = out.loc[
            invalid_end,
            [
                "stay_id",
                "inserted",
                "death_time",
                "reinsertion_time",
                "ICU_out",
                "episode_end_time",
            ],
        ].head(10)
        raise ValueError(
            "Catheter episodes must end after insertion at death, reinsertion, "
            f"or ICU exit. Examples:\n{examples}"
        )

    # Resolve exact timestamp ties deterministically: death, reinsertion, ICU exit alive
    reason = pd.Series(pd.NA, index=out.index, dtype="string")
    reason.loc[out["ICU_out"].eq(out["episode_end_time"])] = "icu_exit_alive"
    reason.loc[out["reinsertion_time"].eq(out["episode_end_time"])] = "reinsertion"
    reason.loc[out["death_time"].eq(out["episode_end_time"])] = "death"
    out["episode_end_reason"] = reason

    if out["episode_end_reason"].isna().any():
        raise ValueError("Every catheter episode must have one resolved ending reason")

    return out

def exclude_post_terminal_episode_starts(catheterised):
    # Remove source episodes that begin after the patient has died or left ICU
    out = catheterised.copy()
    first_patient_terminal = out[["death_time", "ICU_out"]].min(axis=1)
    invalid_start = (
        out["inserted"].isna()
        | first_patient_terminal.isna()
        | out["inserted"].ge(first_patient_terminal)
    )

    if invalid_start.any():
        invalid_death = (
            out["death_time"].notna()
            & out["inserted"].ge(out["death_time"])
        )
        invalid_icu_exit = (
            out["ICU_out"].notna()
            & out["inserted"].ge(out["ICU_out"])
        )
        print(
            "[Cohort exclusion] Foley episodes starting at/after a terminal event:",
            int(invalid_start.sum()),
            f"(at/after death={int((invalid_start & invalid_death).sum())}, "
            f"at/after ICU exit={int((invalid_start & invalid_icu_exit).sum())})",
        )

    return out.loc[~invalid_start].reset_index(drop=True)

def build_catheter_episodes():
    # Load ICU stay boundaries
    icu = pd.read_csv(
        MIMIC_DIR / "icu" / "icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )

    # Parse and validate stay times
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])
    icu = icu.dropna(subset=["stay_id", "intime", "outtime"]).copy()

    # Load patient demographics
    patients = pd.read_csv(
        MIMIC_DIR / "hosp" / "patients.csv",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        low_memory=False,
    )
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"])
    patients["anchor_year"] = pd.to_numeric(patients["anchor_year"])

    # Estimate age at admission
    icu["icu_year"] = icu["intime"].dt.year
    icu = icu.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )
    icu["age"] = icu["anchor_age"] + (icu["icu_year"] - icu["anchor_year"])
    icu = icu.drop(columns=["icu_year", "anchor_age", "anchor_year"])

    # Load admission details
    admissions = pd.read_csv(
        MIMIC_DIR / "hosp" / "admissions.csv",
        usecols=["subject_id", "hadm_id", "race", "deathtime"],
        low_memory=False,
    ).rename(columns={"race": "ethnicity"})
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"])

    # Attach admission attributes
    icu = icu.merge(
        admissions[["subject_id", "hadm_id", "ethnicity", "deathtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    icu["ethnicity_group"] = icu["ethnicity"].apply(map_ethnicity_group)
    icu["ethnicity_missing"] = icu["ethnicity"].apply(ethnicity_is_missing).astype(int)

    # Load Foley procedures
    procedure_events = pd.read_csv(
        MIMIC_DIR / "icu" / "procedureevents.csv",
        usecols=["stay_id", "itemid", "starttime", "endtime"],
    )

    # Keep Foley records only
    procedure_events = procedure_events[procedure_events["itemid"] == FOLEY_ITEMID].copy()
    procedure_events["starttime"] = pd.to_datetime(procedure_events["starttime"])
    procedure_events["endtime"] = pd.to_datetime(procedure_events["endtime"])

    # Attach ICU boundaries
    procedure_events = procedure_events.merge(
        icu[["stay_id", "intime", "outtime"]],
        on="stay_id",
        how="left",
    )

    # Standardise time names
    procedure_events = procedure_events.rename(columns={
        "starttime": "inserted",
        "endtime": "removed",
        "intime": "ICU_in",
        "outtime": "ICU_out",
    })

    # Reject missing removal times
    missing_removed = procedure_events["removed"].isna()
    if missing_removed.any():
        raise ValueError(
            f"{int(missing_removed.sum())} Foley procedure rows have no endtime. "
            "These must not be treated as removals at ICU exit."
        )

    # Drop unusable procedure rows
    procedure_events = procedure_events.dropna(
        subset=["inserted", "removed", "ICU_in", "ICU_out"]
    ).copy()

    # Collapse duplicate episode spans
    collapsed = merge_overlapping_foley_events(procedure_events)

    # Attach stay-level attributes
    catheterised = collapsed.merge(
        icu[[
            "stay_id", "subject_id", "hadm_id", "intime", "outtime",
            "gender", "age", "ethnicity", "ethnicity_group", "ethnicity_missing",
            "deathtime",
        ]],
        on="stay_id",
        how="inner",
    ).rename(columns={"intime": "ICU_in", "outtime": "ICU_out"})

    # Keep deaths inside the ICU stay
    catheterised["death_time"] = catheterised["deathtime"].where(
        (catheterised["deathtime"] >= catheterised["ICU_in"]) &
        (catheterised["deathtime"] <= catheterised["ICU_out"])
    )
    catheterised = catheterised.drop(columns=["deathtime"])

    # Exclude impossible post-death/post-discharge episode starts before linkage
    catheterised = exclude_post_terminal_episode_starts(catheterised)

    # Find the next insertion
    catheterised = catheterised.sort_values(["stay_id", "inserted"]).reset_index(drop=True)
    catheterised["reinsertion_time"] = catheterised.groupby("stay_id")["inserted"].shift(-1)

    # Resolve the earliest episode-ending event
    catheterised = add_episode_endpoints(catheterised)

    # Keep sufficiently long episodes
    observed_catheter_end = catheterised[["removed", "episode_end_time"]].min(axis=1)
    episode_duration = observed_catheter_end - catheterised["inserted"]
    catheterised = catheterised[episode_duration >= MIN_EPISODE_DURATION].copy()

    # Load microbiology results
    micro = pd.read_csv(
        MIMIC_DIR / "hosp" / "microbiologyevents.csv",
        usecols=["subject_id", "hadm_id", "charttime", "spec_type_desc", "org_name"],
    )
    micro["charttime"] = pd.to_datetime(micro["charttime"])

    # Keep positive urine cultures
    micro = micro[
        micro["spec_type_desc"].str.contains("urine", case=False, na=False) &
        micro["org_name"].notna()
    ].copy()

    # Match cultures to episodes
    micro_matched = micro.merge(
        catheterised[[
            "subject_id",
            "hadm_id",
            "stay_id",
            "inserted",
            "removed",
            "episode_end_time",
        ]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    # Limit the CAUTI window
    micro_matched["cauti_window_end"] = (
        micro_matched["removed"] + pd.Timedelta(hours=48)
    ).where(
        micro_matched["removed"] + pd.Timedelta(hours=48)
        <= micro_matched["episode_end_time"],
        micro_matched["episode_end_time"],
    )
    micro_matched = micro_matched[
        (micro_matched["charttime"] >= micro_matched["inserted"]) &
        (micro_matched["charttime"] <= micro_matched["cauti_window_end"])
    ].copy()

    # Keep the first culture
    micro_matched = (
        micro_matched.sort_values("charttime")
        .drop_duplicates(["stay_id", "inserted"])
        [["stay_id", "inserted", "charttime"]]
        .rename(columns={"charttime": "cauti_time"})
    )

    # Attach episode outcomes
    catheterised = catheterised.merge(micro_matched, on=["stay_id", "inserted"], how="left")

    # Return episodes in time order
    return catheterised.sort_values(["stay_id", "inserted"]).reset_index(drop=True)

def make_state_windows(state_start, state_end):
    # Reject empty state spans
    if pd.isna(state_start) or pd.isna(state_end) or state_end <= state_start:
        return []

    # Initialise the first period
    rows = []
    window_start = state_start
    state_idx = 0

    # Split the state into periods
    while window_start < state_end:
        window_end = min(window_start + PERIOD_DURATION, state_end)
        rows.append((state_idx, window_start, window_end))
        window_start = window_end
        state_idx += 1

    # Return period boundaries
    return rows

def validate_cauti_risk_set(panel, episode_keys):
    # Reconstruct first-event eligibility in episode-time order
    ordered = panel.sort_values(
        [*episode_keys, "period_start", "period_end", "catheter_state"]
    )
    prior_cauti = (
        ordered.groupby(episode_keys, sort=False)[Y_CAUTI].cumsum()
        - ordered[Y_CAUTI]
    ).gt(0)
    state_window_eligible = (
        ordered["catheter_state"].eq("in")
        | (
            ordered["catheter_state"].eq("out")
            & ordered["periods_in_state"].le(POST_REMOVE_RISK_PERIODS)
        )
    )
    at_risk = ordered["at_risk_cauti"].eq(1)

    # The event row belongs to the first-event risk set
    if (ordered[Y_CAUTI].eq(1) & ~at_risk).any():
        raise ValueError("Every CAUTI event row must be marked at risk")

    # Follow-up remains in the panel, but not in the observed first-event risk set
    if (prior_cauti & at_risk).any():
        raise ValueError(
            "Panel contains CAUTI at-risk rows after an earlier episode CAUTI"
        )

    expected_at_risk = state_window_eligible & ~prior_cauti
    if not at_risk.eq(expected_at_risk).all():
        raise ValueError(
            "CAUTI risk set does not match catheter-state, post-removal-window, "
            "and first-event eligibility"
        )

def build_base_panel(catheterised):
    # Collect panel rows
    rows = []

    # Expand each catheter episode
    for episode in catheterised.itertuples():
        # A terminal event can occur before the recorded procedure end time
        in_state_end = min(episode.removed, episode.episode_end_time)

        # Create catheter-in periods
        for state_idx, period_start, period_end in make_state_windows(episode.inserted, in_state_end):
            rows.append({
                "subject_id": episode.subject_id,
                "hadm_id": episode.hadm_id,
                "stay_id": episode.stay_id,
                "inserted": episode.inserted,
                "removed": episode.removed,
                "reinsertion_time": episode.reinsertion_time,
                "episode_end_time": episode.episode_end_time,
                "_episode_end_reason": episode.episode_end_reason,
                "catheter_state": "in",
                "state_index": state_idx,
                "period_start": period_start,
                "period_end": period_end,
                "cauti_time": episode.cauti_time,
                "death_time": episode.death_time,
                "ICU_out": episode.ICU_out,
                "intime": episode.ICU_in,
                "gender": episode.gender,
                "age": episode.age,
                "ethnicity": episode.ethnicity,
                "ethnicity_group": episode.ethnicity_group,
                "ethnicity_missing": episode.ethnicity_missing,
            })

        # Create catheter-out periods only after a removal preceding the terminal event
        for state_idx, period_start, period_end in make_state_windows(
            episode.removed,
            episode.episode_end_time,
        ):
            rows.append({
                "subject_id": episode.subject_id,
                "hadm_id": episode.hadm_id,
                "stay_id": episode.stay_id,
                "inserted": episode.inserted,
                "removed": episode.removed,
                "reinsertion_time": episode.reinsertion_time,
                "episode_end_time": episode.episode_end_time,
                "_episode_end_reason": episode.episode_end_reason,
                "catheter_state": "out",
                "state_index": state_idx,
                "period_start": period_start,
                "period_end": period_end,
                "cauti_time": episode.cauti_time,
                "death_time": episode.death_time,
                "ICU_out": episode.ICU_out,
                "intime": episode.ICU_in,
                "gender": episode.gender,
                "age": episode.age,
                "ethnicity": episode.ethnicity,
                "ethnicity_group": episode.ethnicity_group,
                "ethnicity_missing": episode.ethnicity_missing,
            })

    # Build and order the panel
    panel = pd.DataFrame(rows)
    panel = panel.sort_values(
        ["stay_id", "inserted", "period_start", "period_end", "catheter_state"]
    ).reset_index(drop=True)

    # Number periods within episodes
    panel["episode_index"] = panel.groupby(["stay_id", "inserted"]).cumcount()
    panel["periods_in_state"] = panel["state_index"] + 1
    panel = panel.drop(columns=["state_index"])

    # Find each final period
    episode_keys = ["stay_id", "inserted"]
    last_row_index = panel.groupby(episode_keys)["period_end"].idxmax()
    is_last_period = panel.index.isin(last_row_index)

    # Flag clinical removal periods; terminal-time ties are not removal decisions
    panel["removed_in_period"] = (
        (panel["catheter_state"] == "in") &
        (panel["removed"] < panel["episode_end_time"]) &
        (panel["removed"] > panel["period_start"]) &
        (panel["removed"] <= panel["period_end"])
    ).astype(int)

    # Encode mutually exclusive terminal events on the final trajectory row
    panel["reinsertion_in_period"] = (
        is_last_period &
        panel["_episode_end_reason"].eq("reinsertion")
    ).astype(int)
    panel["death_in_period"] = (
        is_last_period &
        panel["_episode_end_reason"].eq("death")
    ).astype(int)
    panel["icu_exit_alive_in_period"] = (
        is_last_period &
        panel["_episode_end_reason"].eq("icu_exit_alive")
    ).astype(int)

    # Flag CAUTI periods
    panel["cauti_in_period"] = (
        panel["cauti_time"].notna() &
        (panel["cauti_time"] > panel["period_start"]) &
        (panel["cauti_time"] <= panel["period_end"])
    ).astype(int)

    # Mark existing catheter-state and post-removal CAUTI eligibility
    cauti_state_window_eligible = (
        (panel["catheter_state"] == "in") |
        ((panel["catheter_state"] == "out") & (panel["periods_in_state"] <= POST_REMOVE_RISK_PERIODS))
    )

    # Retain the event row, then remove only later rows from the first-event risk set
    prior_cauti = (
        panel.groupby(episode_keys, sort=False)[Y_CAUTI].cumsum()
        - panel[Y_CAUTI]
    ).gt(0)
    panel["at_risk_cauti"] = (
        cauti_state_window_eligible & ~prior_cauti
    ).astype(int)

    # Mark reinsertion risk periods
    panel["at_risk_reinsertion"] = (panel["catheter_state"] == "out").astype(int)

    # Record observed episode endings
    panel["episode_end_reason"] = panel["_episode_end_reason"].where(is_last_period)

    # Validate terminal-state construction before dropping source timestamps
    terminal_count = panel[
        ["reinsertion_in_period", "death_in_period", "icu_exit_alive_in_period"]
    ].sum(axis=1)
    if not terminal_count.eq(is_last_period.astype(int)).all():
        raise ValueError("Each episode must have exactly one terminal event on its final row")
    if panel["period_end"].gt(panel["episode_end_time"]).any():
        raise ValueError("Panel contains follow-up after an episode terminal event")
    final_end_matches = panel.loc[is_last_period, "period_end"].eq(
        panel.loc[is_last_period, "episode_end_time"]
    )
    if not final_end_matches.all():
        raise ValueError("Each final panel period must end at episode_end_time")
    invalid_reasons = set(panel["episode_end_reason"].dropna()) - TERMINAL_REASONS
    if invalid_reasons:
        raise ValueError(f"Unexpected episode ending reasons: {sorted(invalid_reasons)}")
    validate_cauti_risk_set(panel, episode_keys)

    # Encode demographic categories
    panel["sex_M"] = (panel["gender"] == "M").astype(int)
    panel["sex_missing"] = panel["gender"].isna().astype(int)
    eth_dummies = pd.get_dummies(
        panel["ethnicity_group"], prefix="ethnicity", dtype=int
    )
    panel = pd.concat([panel, eth_dummies], axis=1)

    # Define covariate lookbacks
    panel["cov_start"] = panel["period_start"] - LOOKBACK_DURATION
    panel["cov_end"] = panel["period_start"]
    panel["cov_start"] = panel["cov_start"].clip(lower=panel["intime"])
    panel["row_id"] = np.arange(1, len(panel) + 1)

    # Drop temporary source fields
    panel = panel.drop(columns=[
        "_episode_end_reason",
        "cauti_time",
        "death_time",
        "gender",
        "ethnicity",
        "ethnicity_group",
        "intime",
        "ICU_out",
    ])

    # Place columns consistently
    ethnicity_cols = sorted([c for c in panel.columns if c.startswith("ethnicity_")])
    ordered_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "inserted",
        "removed",
        "reinsertion_time",
        "episode_end_time",
        "catheter_state",
        "episode_index",
        "period_start",
        "period_end",
        "periods_in_state",
        "removed_in_period",
        "reinsertion_in_period",
        "cauti_in_period",
        "death_in_period",
        "icu_exit_alive_in_period",
        "episode_end_reason",
        "at_risk_cauti",
        "at_risk_reinsertion",
        "age",
        "sex_M",
        "sex_missing",
        *ethnicity_cols,
        "cov_start",
        "cov_end",
        "row_id",
    ]

    # Return the ordered panel
    return panel[ordered_cols]

def create_episode_cohort_and_base_panel():
    # Define stage outputs
    catheter_episodes_file = DATA_DIR / "catheter_episodes.csv"
    base_panel_file = DATA_DIR / "base_panel.csv"
    ethnicity_mapping_audit_file = DATA_DIR / "ethnicity_mapping_audit.csv"

    # Build the cohort and panel
    episodes = build_catheter_episodes()
    base_panel = build_base_panel(episodes)
    ethnicity_mapping_audit = build_ethnicity_mapping_audit(episodes)

    # Select exported episode fields
    episode_export = episodes[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "inserted",
            "removed",
            "reinsertion_time",
            "episode_end_time",
            "episode_end_reason",
            "ICU_in",
            "ICU_out",
            "death_time",
        ]
    ].copy()

    # Save both datasets
    episode_export.to_csv(catheter_episodes_file, index=False)
    base_panel.to_csv(base_panel_file, index=False)
    ethnicity_mapping_audit.to_csv(ethnicity_mapping_audit_file, index=False)

    # Report cohort sizes
    print("[SAVE]", catheter_episodes_file)
    print("[SAVE]", base_panel_file)
    print("[SAVE]", ethnicity_mapping_audit_file)
    print(ethnicity_mapping_audit.to_string(index=False))
    print("Episodes:", len(episodes))
    print("Stays:", episodes["stay_id"].nunique())
    print("Base panel rows:", len(base_panel))

def build_chart_extraction_windows(episodes):
    # Select episode boundaries
    windows = episodes[
        ["stay_id", "inserted", "episode_end_time", "ICU_in"]
    ].copy()

    # Define extraction bounds
    windows["window_start"] = (windows["inserted"] - LOOKBACK_DURATION).clip(
        lower=windows["ICU_in"]
    )
    windows["window_end"] = windows["episode_end_time"]

    # Remove invalid windows
    windows = windows.dropna(subset=["window_start", "window_end"]).copy()
    windows = windows[windows["window_end"] > windows["window_start"]].copy()

    # Merge windows within stays
    merged_windows = []
    for stay_id, stay_windows in windows.groupby("stay_id"):
        stay_windows = stay_windows.sort_values("window_start")
        current_start = None
        current_end = None

        for row in stay_windows.itertuples():
            # Start the first window
            if current_start is None:
                current_start = row.window_start
                current_end = row.window_end
                continue

            # Extend an overlapping window
            if row.window_start <= current_end:
                current_end = max(current_end, row.window_end)
            else:
                # Close a completed window
                merged_windows.append((stay_id, current_start, current_end))
                current_start = row.window_start
                current_end = row.window_end

        # Close the final window
        if current_start is not None:
            merged_windows.append((stay_id, current_start, current_end))

    # Return merged windows
    return pd.DataFrame(merged_windows, columns=["stay_id", "window_start", "window_end"])

def extract_chart_covariates():
    # Define stage files
    catheter_episodes_file = DATA_DIR / "catheter_episodes.csv"
    kept_preprocessed_chart_file = DATA_DIR / "preprocessed_raw_chart_covariates_kept.csv"
    d_items_keep_file = CONFIG_DIR / "d_items_keep.csv"

    # Load episode timing
    episode_cols = ["stay_id", "inserted", "episode_end_time", "ICU_in"]
    episodes = pd.read_csv(
        catheter_episodes_file,
        usecols=episode_cols,
        low_memory=False,
    )

    # Parse episode timestamps
    for col in ["inserted", "episode_end_time", "ICU_in"]:
        episodes[col] = pd.to_datetime(episodes[col])

    # Build target stays and windows
    windows = build_chart_extraction_windows(episodes)
    stay_ids = set(windows["stay_id"].dropna().astype(int).unique())

    # Load the item allowlist
    keep_df = pd.read_csv(d_items_keep_file, usecols=["itemid"], low_memory=False)
    keep_df["itemid"] = pd.to_numeric(keep_df["itemid"])
    keep_itemids = set(keep_df["itemid"].dropna().astype(int))

    # Include Fahrenheit source rows
    source_itemids = keep_itemids - {TEMP_F_ITEMID}
    if TEMP_C_ITEMID in keep_itemids:
        source_itemids.add(TEMP_F_ITEMID)

    # Define input and output schemas
    source_cols = [
        "subject_id", "hadm_id", "stay_id", "itemid", "charttime",
        "storetime", "valuenum", "value", "valueuom",
    ]
    output_cols = ["stay_id", "itemid", "charttime", "valuenum", "valueuom"]
    chartevents_file = MIMIC_DIR / "icu" / "chartevents.csv"

    # Use an atomic output file
    tmp_outfile = kept_preprocessed_chart_file.with_suffix(
        kept_preprocessed_chart_file.suffix + ".writing"
    )

    # Initialise the output file
    pd.DataFrame(columns=output_cols).to_csv(tmp_outfile, index=False)

    # Track extraction totals
    kept_rows_total = 0
    converted_rows_total = 0

    # Report extraction settings
    print("[CONFIG]", MIMIC_DIR)
    print("[Catheter episodes]", len(episodes))
    print("[Chart windows]", len(windows))
    print(f"[Chart item allowlist] {len(keep_itemids):,}")
    print("[EHR] Extracting chart covariates...")

    # Stream the chart table
    for chunk_idx, chunk in enumerate(
        pd.read_csv(chartevents_file, usecols=source_cols, chunksize=CHUNK_ROWS, low_memory=False),
        start=1,
    ):
        # Keep cohort stays
        selected = chunk.loc[chunk["stay_id"].isin(stay_ids)].copy()

        # Keep allowed items and temperatures
        selected_itemids = pd.to_numeric(selected["itemid"])
        selected_units = selected["valueuom"].fillna("").astype(str).str.strip().str.upper()
        selected = selected.loc[
            selected_itemids.isin(source_itemids)
            | selected_units.isin(FAHRENHEIT_UNITS)
        ].copy()

        # Normalise event timestamps
        selected["charttime"] = pd.to_datetime(selected["charttime"])
        selected["storetime"] = pd.to_datetime(selected["storetime"])
        selected = selected.dropna(subset=["stay_id", "charttime"])

        # Match events to extraction windows
        matched = selected.merge(windows, on="stay_id", how="inner")
        matched = matched.loc[
            matched["charttime"].ge(matched["window_start"])
            & matched["charttime"].lt(matched["window_end"])
        ]

        # Remove exact source duplicates
        matched = matched.drop_duplicates(subset=source_cols)
        output_chunk = matched[output_cols].copy()

        # Convert Fahrenheit values
        unit_clean = output_chunk["valueuom"].fillna("").astype(str).str.strip().str.upper()
        output_itemids = pd.to_numeric(output_chunk["itemid"])
        fahrenheit_mask = unit_clean.isin(FAHRENHEIT_UNITS) | output_itemids.eq(TEMP_F_ITEMID)
        fahrenheit_values = pd.to_numeric(
            output_chunk.loc[fahrenheit_mask, "valuenum"],
        )
        output_chunk.loc[fahrenheit_mask, "valuenum"] = (
            fahrenheit_values - 32.0
        ) * (5.0 / 9.0)
        output_chunk.loc[fahrenheit_mask, "itemid"] = TEMP_C_ITEMID
        output_chunk.loc[fahrenheit_mask, "valueuom"] = CELSIUS_UNIT
        converted_rows_total += int(fahrenheit_mask.sum())

        # Reapply the final allowlist
        output_itemids = pd.to_numeric(output_chunk["itemid"])
        output_chunk = output_chunk.loc[output_itemids.isin(keep_itemids)]
        kept = len(output_chunk)
        kept_rows_total += kept

        # Append retained rows
        output_chunk.to_csv(tmp_outfile, mode="a", header=False, index=False)

        # Report chunk progress
        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} keep={kept:,} "
            f"cum_keep={kept_rows_total:,}"
        )

    # Publish the completed file
    tmp_outfile.replace(kept_preprocessed_chart_file)

    # Report extraction totals
    print(f"[SAVE] {kept_preprocessed_chart_file}")
    print(f"[INFO] rows kept: {kept_rows_total:,}")
    print(f"[INFO] Fahrenheit rows converted: {converted_rows_total:,}")

def _load_numeric_values_by_itemid():
    # Collect values across chunks
    value_parts = defaultdict(list)

    # Stream item-value pairs
    chart_file = DATA_DIR / "preprocessed_raw_chart_covariates_kept.csv"
    for chunk in pd.read_csv(chart_file, usecols=[ITEM_COL, VALUE_COL], chunksize=CHUNK_ROWS, low_memory=False):
        # Keep valid numeric values
        chunk[ITEM_COL] = pd.to_numeric(chunk[ITEM_COL])
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL])
        chunk = chunk.dropna(subset=[ITEM_COL, VALUE_COL]).copy()
        chunk[ITEM_COL] = chunk[ITEM_COL].astype(int)

        # Store values by item
        for itemid, item_rows in chunk.groupby(ITEM_COL, sort=False):
            value_parts[int(itemid)].append(item_rows[VALUE_COL].reset_index(drop=True))

    # Join each item's chunks
    return {
        itemid: pd.concat(parts, ignore_index=True)
        for itemid, parts in value_parts.items()
    }

def fit_cleaning_rules():
    # Load values by item
    values_by_itemid = _load_numeric_values_by_itemid()
    rule_rows = []

    # Fit one rule per item
    for itemid, values in values_by_itemid.items():
        # Summarise zero values
        nonzero_values = values[values != 0]
        n_non_missing = int(values.shape[0])
        zero_fraction = float(values.eq(0).mean())
        p5_nonzero = float(nonzero_values.quantile(0.05))

        # Detect likely missing zeros
        zero_to_missing = (
            n_non_missing >= MIN_N_FOR_RULES
            and p5_nonzero > 0
            and 0 < zero_fraction <= ZERO_MAX_FRAC
        )
        filtered_values = nonzero_values if zero_to_missing else values

        # Record robust quantiles
        rule_rows.append({
            ITEM_COL: itemid,
            "n_non_missing": n_non_missing,
            "zero_fraction": zero_fraction,
            "p5_nonzero": p5_nonzero,
            "zero_to_missing": zero_to_missing,
            "n_for_thresholds": int(filtered_values.shape[0]),
            "p1": float(filtered_values.quantile(0.01)),
            "q1": float(filtered_values.quantile(0.25)),
            "q3": float(filtered_values.quantile(0.75)),
            "p99": float(filtered_values.quantile(0.99)),
        })

    # Flag underpowered rules
    rules = pd.DataFrame(rule_rows)
    rules["status"] = "ok"
    rules.loc[rules["n_for_thresholds"] < MIN_N_FOR_RULES, "status"] = "too_few_values_for_thresholds"

    # Estimate each item's spread
    iqr = rules["q3"] - rules["q1"]
    tail_span = rules["p99"] - rules["p1"]
    spread = pd.concat([iqr, tail_span], axis=1).max(axis=1)
    spread = spread.fillna(0.0).clip(lower=1e-8)

    # Define cleaning bounds
    rules["lower_clip"] = rules["p1"]
    rules["upper_clip"] = rules["p99"]
    rules["lower_delete"] = rules["p1"] - FAR_OUT_SPREAD_MULT * spread
    rules["upper_delete"] = rules["p99"] + FAR_OUT_SPREAD_MULT * spread

    # Return all fitted rules
    return rules

def apply_cleaning_rules(rules):
    # Prepare an atomic output
    infile = DATA_DIR / "preprocessed_raw_chart_covariates_kept.csv"
    outfile = DATA_DIR / "cleaned_chart_covariates.csv"
    tmp_outfile = outfile.with_suffix(outfile.suffix + ".writing")

    # Keep required rule columns
    rules_small = rules[
        [
            ITEM_COL,
            "status",
            "zero_to_missing",
            "lower_clip",
            "upper_clip",
            "lower_delete",
            "upper_delete",
        ]
    ].copy()

    # Write the header once
    first_write = True

    # Clean the file in chunks
    for chunk in pd.read_csv(infile, chunksize=CHUNK_ROWS, low_memory=False):
        # Preserve the source schema
        original_columns = list(chunk.columns)

        # Normalise numeric fields
        chunk[ITEM_COL] = pd.to_numeric(chunk[ITEM_COL])
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL])

        # Attach item-specific rules
        chunk = chunk.merge(rules_small, on=ITEM_COL, how="left")

        # Replace likely missing zeros
        zero_mask = (
            chunk["status"].eq("ok") &
            chunk["zero_to_missing"].fillna(False) &
            chunk[VALUE_COL].eq(0)
        )
        chunk.loc[zero_mask, VALUE_COL] = np.nan

        # Remove extreme outliers
        far_low_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] < chunk["lower_delete"])
        )
        far_high_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] > chunk["upper_delete"])
        )
        far_mask = far_low_mask | far_high_mask
        chunk.loc[far_mask, VALUE_COL] = np.nan

        # Identify moderate outliers
        clip_low_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] < chunk["lower_clip"])
        )
        clip_high_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] > chunk["upper_clip"])
        )

        # Clip moderate outliers
        chunk.loc[clip_low_mask, VALUE_COL] = chunk.loc[clip_low_mask, "lower_clip"]
        chunk.loc[clip_high_mask, VALUE_COL] = chunk.loc[clip_high_mask, "upper_clip"]

        # Write cleaned source columns
        chunk = chunk[original_columns]
        chunk.to_csv(tmp_outfile, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    # Publish the cleaned file
    tmp_outfile.replace(outfile)

def clean_chart_covariates():
    # Define stage files
    kept_preprocessed_chart_file = DATA_DIR / "preprocessed_raw_chart_covariates_kept.csv"
    cleaned_chart_file = DATA_DIR / "cleaned_chart_covariates.csv"
    cleaning_rules_file = DATA_DIR / "chart_covariate_cleaning_rules.csv"

    # Fit and save rules
    print("[FIT RULES]", kept_preprocessed_chart_file)
    rules = fit_cleaning_rules()
    rules.round(3).to_csv(
        cleaning_rules_file,
        index=False,
        float_format="%.3f",
    )
    print("[SAVE RULES]", cleaning_rules_file)

    # Apply the fitted rules
    print("[APPLY RULES]", kept_preprocessed_chart_file)
    apply_cleaning_rules(rules)

    print("[SAVE CLEANED]", cleaned_chart_file)

def build_retention_log(aggregated, panel, itemid_to_label):
    # Calculate coverage denominators
    total_rows = len(panel)
    total_stays = panel["stay_id"].nunique()

    # Attach stays to aggregates
    aggregated = aggregated.merge(
        panel[["row_id", "stay_id"]],
        on="row_id",
        how="left",
    )
    summaries = []

    # Summarise each statistic
    for stat in AGG_STATS:
        # Keep observed values
        observed = aggregated.loc[
            aggregated[stat].notna(), ["row_id", "itemid", "stay_id"]
        ]

        # Count covered rows and stays
        stat_summary = (
            observed.groupby("itemid", sort=False)
            .agg(n_rows=("row_id", "size"), n_stays=("stay_id", "nunique"))
            .reset_index()
        )
        stat_summary["itemid"] = stat_summary["itemid"].astype(int)
        stat_summary["stat"] = stat
        summaries.append(stat_summary)

    # Define report columns
    columns = [
        "itemid", "label", "stat", "column_name", "n_rows",
        "row_coverage", "n_stays", "stay_coverage", "decision",
    ]

    # Combine statistic summaries
    summary = pd.concat(summaries, ignore_index=True)

    # Add readable feature names
    summary["label"] = summary["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    summary["column_name"] = (
        "itemid_" + summary["itemid"].astype(str) + "__" + summary["stat"]
    )

    # Calculate coverage rates
    summary["row_coverage"] = summary["n_rows"] / total_rows
    summary["stay_coverage"] = summary["n_stays"] / total_stays

    # Apply retention thresholds
    retain = (
        summary["row_coverage"].ge(MIN_ROW_COVERAGE)
        & summary["stay_coverage"].ge(MIN_STAY_COVERAGE)
    )
    summary["decision"] = np.where(retain, "retain", "drop")

    # Round displayed coverage
    summary[["row_coverage", "stay_coverage"]] = summary[
        ["row_coverage", "stay_coverage"]
    ].round(4)

    # Return an ordered report
    return summary[columns].sort_values(
        ["decision", "row_coverage", "stay_coverage", "n_rows", "itemid", "stat"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)

def aggregate_itemid_covariates(panel, itemid_to_label):
    # Select panel lookback windows
    windows = panel[["row_id", "stay_id", "cov_start", "cov_end"]].copy()

    # Initialise chunk accumulators
    cleaned_chart_file = DATA_DIR / "cleaned_chart_covariates.csv"
    chart_cols = ["stay_id", "itemid", "charttime", "valuenum"]
    partial_stats = []
    last_obs_parts = []

    kept_rows_total = 0

    # Report aggregation settings
    print("[Base panel rows]", len(panel))
    print("[EHR] Aggregating chartevents covariates...")

    # Process chart rows in chunks
    for chunk_idx, chunk in enumerate(
        pd.read_csv(cleaned_chart_file, usecols=chart_cols, chunksize=CHUNK_ROWS, low_memory=False),
        start=1,
    ):
        rows_read = len(chunk)

        # Keep valid numeric observations
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"]).copy()

        # Match observations to lookbacks
        merged = chunk.merge(windows, on="stay_id", how="inner")
        merged = merged[
            (merged["charttime"] >= merged["cov_start"])
            & (merged["charttime"] < merged["cov_end"])
        ].copy()

        kept = len(merged)
        kept_rows_total += kept

        if kept > 0:
            # Build regression components
            merged["charttime_seconds"] = merged["charttime"].astype("int64") / 1_000_000_000.0
            merged["valuenum_sq"] = merged["valuenum"] * merged["valuenum"]
            merged["charttime_sq"] = merged["charttime_seconds"] * merged["charttime_seconds"]
            merged["charttime_value"] = merged["charttime_seconds"] * merged["valuenum"]

            # Aggregate chunk-level components
            item_summary = (
                merged.groupby(["row_id", "itemid"])
                .agg(
                    count=("valuenum", "count"),
                    total=("valuenum", "sum"),
                    sum_sq=("valuenum_sq", "sum"),
                    sum_t=("charttime_seconds", "sum"),
                    sum_tt=("charttime_sq", "sum"),
                    sum_ty=("charttime_value", "sum"),
                )
                .reset_index()
            )
            partial_stats.append(item_summary)

            # Keep each chunk's last value
            last_obs_parts.append(
                merged.sort_values(["row_id", "itemid", "charttime"])
                .drop_duplicates(["row_id", "itemid"], keep="last")
                [["row_id", "itemid", "charttime", "valuenum"]]
                .rename(columns={"charttime": "last_time", "valuenum": "last"})
            )

        # Report chunk progress
        print(
            f"[EHR][{chunk_idx}] read={rows_read:,} numeric={len(chunk):,} "
            f"matched={kept:,} cum_matched={kept_rows_total:,}"
        )

    # Combine chunk-level components
    agg = pd.concat(partial_stats, ignore_index=True)
    agg = (
        agg.groupby(["row_id", "itemid"])
        .agg(
            count=("count", "sum"),
            total=("total", "sum"),
            sum_sq=("sum_sq", "sum"),
            sum_t=("sum_t", "sum"),
            sum_tt=("sum_tt", "sum"),
            sum_ty=("sum_ty", "sum"),
        )
        .reset_index()
    )

    # Calculate means
    agg["mean"] = agg["total"] / agg["count"]

    # Calculate sample deviations
    variance_num = agg["sum_sq"] - (agg["total"] * agg["total"] / agg["count"])
    agg["std"] = np.where(
        agg["count"] > 1,
        np.sqrt((variance_num / (agg["count"] - 1)).clip(lower=0)),
        np.nan,
    )

    # Select overall last values
    last_obs = (
        pd.concat(last_obs_parts, ignore_index=True)
        .sort_values(["row_id", "itemid", "last_time"])
        .drop_duplicates(["row_id", "itemid"], keep="last")
    )
    agg = agg.merge(last_obs[["row_id", "itemid", "last"]], on=["row_id", "itemid"], how="left")

    # Calculate hourly slopes
    slope_denom = agg["count"] * agg["sum_tt"] - agg["sum_t"] * agg["sum_t"]
    slope_num = agg["count"] * agg["sum_ty"] - agg["sum_t"] * agg["total"]
    agg["slope_per_hour"] = np.where(
        (agg["count"] > 1) & (slope_denom != 0),
        (slope_num / slope_denom) * 3600.0,
        np.nan,
    )

    # Identify retained features
    retention_log = build_retention_log(agg, panel, itemid_to_label)

    # Clear sparse statistic values
    for stat in AGG_STATS:
        retained_itemids = set(
            retention_log.loc[
                retention_log["decision"].eq("retain") & retention_log["stat"].eq(stat),
                "itemid",
            ]
        )
        agg.loc[~agg["itemid"].isin(retained_itemids), stat] = np.nan

    # Pivot features into columns
    wide = agg.pivot_table(
        index="row_id",
        columns="itemid",
        values=AGG_STATS,
        aggfunc="first",
    )
    wide.columns = [f"itemid_{itemid}__{stat}" for stat, itemid in wide.columns]
    wide = wide.reset_index()

    # Attach features to the panel
    panel = panel.merge(wide, on="row_id", how="left")
    print("[EHR] Covariates aggregated.")
    return panel, retention_log

def detect_covariate_itemids(columns):
    # Match chart feature names
    pattern = re.compile(r"^itemid_(\d+)__", flags=re.IGNORECASE)
    itemids = set()

    # Collect unique item IDs
    for col in columns:
        match = pattern.match(str(col))
        if match:
            itemids.add(int(match.group(1)))

    # Return sorted IDs
    return pd.DataFrame({"itemid": sorted(itemids)})

def add_itemid_missing_indicators(df):
    # Add one missingness indicator per final itemid-derived feature
    chart_feature_cols = sorted(
        col for col in df.columns
        if col.startswith("itemid_") and not col.endswith("__missing")
    )
    missing_indicators = {
        f"{col}__missing": df[col].isna().astype(int)
        for col in chart_feature_cols
        if f"{col}__missing" not in df.columns
    }
    if not missing_indicators:
        return df
    return pd.concat(
        [df, pd.DataFrame(missing_indicators, index=df.index)],
        axis=1,
    ).copy()

def build_modelling_panel():
    # Define stage files
    base_panel_file = DATA_DIR / "base_panel.csv"
    modelling_panel_file = DATA_DIR / "modelling_panel.csv"
    retained_covariates_file = DATA_DIR / "retained_covariates.csv"
    covariate_dictionary_file = DATA_DIR / "covariate_dictionary.csv"

    # Load the base panel
    df = pd.read_csv(base_panel_file, low_memory=False)

    # Parse panel timestamps
    for col in [
        "inserted",
        "removed",
        "reinsertion_time",
        "episode_end_time",
        "period_start",
        "period_end",
        "cov_start",
        "cov_end",
    ]:
        df[col] = pd.to_datetime(df[col])

    # Add retained chart features
    df, retention_log = aggregate_itemid_covariates(
        df,
        load_item_labels(),
    )

    df = add_itemid_missing_indicators(df)

    # Remove aggregation helpers
    df = df.drop(columns=["cov_start", "cov_end", "row_id"])

    # Place chart features last
    chart_feature_cols = sorted([col for col in df.columns if col.startswith("itemid_")])
    base_cols = [col for col in df.columns if not col.startswith("itemid_")]
    df = df[base_cols + chart_feature_cols]

    # Normalise chart features together to avoid fragmenting the panel.
    chart_features = df[chart_feature_cols].apply(pd.to_numeric).round(ROUND_DP)
    df = pd.concat([df[base_cols], chart_features], axis=1).copy()

    # Normalise event indicators
    for col in [ACTION_COL, Y_CAUTI, Y_REINS, Y_DEATH, Y_ICU_EXIT_ALIVE]:
        df[col] = pd.to_numeric(df[col]).fillna(0).astype(int)

    # Label observed actions
    df[OBSERVED_ACTION_COL] = "keep"
    df.loc[
        (df[STATE_COL] == "in") & (df[ACTION_COL] == 1),
        OBSERVED_ACTION_COL,
    ] = "remove"
    df.loc[df[STATE_COL] == "out", OBSERVED_ACTION_COL] = "out"

    # Select model features
    feature_cols = [
        col for col in df.columns
        if (
            col.startswith("itemid_")
            or col.startswith("sex_")
            or col.startswith("ethnicity_")
        )
    ]
    feature_cols.append("age")

    # Coerce model inputs
    for col in [*feature_cols, TIME_COL, PERIODS_COL]:
        df[col] = pd.to_numeric(df[col])

    # Save panel and retention report
    df.to_csv(modelling_panel_file, index=False)
    retention_log.round(3).to_csv(
        retained_covariates_file,
        index=False,
        float_format="%.3f",
    )

    # Build the item dictionary
    covariate_dict = detect_covariate_itemids(df.columns)
    itemid_to_label = load_item_labels()
    covariate_dict["label"] = covariate_dict["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    covariate_dict.sort_values(["label", "itemid"]).round(3).to_csv(
        covariate_dictionary_file,
        index=False,
        float_format="%.3f",
    )

    # Report modelling outputs
    print(f"[SAVE] modelling panel: {modelling_panel_file}")
    print(f"[SAVE] covariate dictionary: {covariate_dictionary_file}")
    print("Removals:", df["removed_in_period"].sum())
    print("Reinsertions:", df["reinsertion_in_period"].sum())
    print("CAUTI:", df["cauti_in_period"].sum())
    print("Deaths:", df["death_in_period"].sum())
    print("ICU exits alive:", df["icu_exit_alive_in_period"].sum())
    print(f"Rows: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print("Retained chart columns:", int(retention_log["decision"].eq("retain").sum()))
    print("Dropped chart columns:", int(retention_log["decision"].eq("drop").sum()))

def main():
    # Create the output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Report active directories
    print("[CONFIG] repo root:", REPO_ROOT)
    print("[CONFIG] mimic dir:", MIMIC_DIR)
    print("[CONFIG] data dir:", DATA_DIR)
    print("[CONFIG] config dir:", CONFIG_DIR)

    # Build the episode panel
    print_section("Define catheter episode cohort and base panel")
    create_episode_cohort_and_base_panel()

    # Extract chart covariates
    print_section("Extract and preprocess chart-event covariates")
    extract_chart_covariates()

    # Clean chart covariates
    print_section("Clean chart covariates")
    clean_chart_covariates()
    (DATA_DIR / "preprocessed_raw_chart_covariates_kept.csv").unlink()

    # Build the modelling panel
    print_section("Aggregate cleaned covariates onto the base panel")
    build_modelling_panel()
    (DATA_DIR / "cleaned_chart_covariates.csv").unlink()

    # Report completion
    print()
    print("Data panel creation completed.")

if __name__ == "__main__":
    # Run the full pipeline
    main()
