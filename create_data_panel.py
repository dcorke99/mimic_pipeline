#!/usr/bin/env python3
"""
Create the full modeling data panel for the CAUTI catheter-removal pipeline.

This script consolidates the original panel-building scripts while preserving
the same logical development flow and intermediate outputs.

The code is organised as a single readable pipeline:
- define the catheter episode cohort and base row-level panel
- extract, preprocess, filter, validate, and clean chart-event covariates
- aggregate cleaned covariates onto the base panel
- retain usable covariates and create the train/test split
- write the final modeling panel, feature spec, and covariate dictionary
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MIMIC_DIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\Data\MIMIC-IV\mimic-iv-3.1")

CHUNK_ROWS = 1_000_000
SAMPLE_ROWS = 1000
AUDIT_DECIMAL_PLACES = 3

FOLEY_ITEMID = 229351
MIN_EPISODE_DURATION = pd.Timedelta(hours=24)
PERIOD_DURATION = pd.Timedelta(hours=24)
LOOKBACK_DURATION = pd.Timedelta(hours=24)
LOOKBACK_HOURS = 24
POST_REMOVE_RISK_PERIODS = 2

FAHRENHEIT_UNITS = {"F", "DEG F", "DEGREES F", "°F", "° F"}
CELSIUS_UNIT = "°C"
TEMP_F_ITEMID = 223761
TEMP_C_ITEMID = 223762

ITEM_COL = "itemid"
VALUE_COL = "valuenum"
MIN_N_FOR_RULES = 100
ZERO_MAX_FRAC = 0.10
FAR_OUT_SPREAD_MULT = 3.0
ALWAYS_ZERO_TO_MISSING: set[int] = set()
NEVER_ZERO_TO_MISSING: set[int] = set()

AGG_STATS = [
    "count",
    "mean",
    "std",
    "last",
    "slope_per_hour",
]

MIN_ROW_COVERAGE = 0.05
MIN_STAY_COVERAGE = 0.10
MEAN_ONLY = False
ROUND_DP = 3
SUBJECT_ID_COL = "subject_id"
TEST_SIZE = 0.20
SEED = 42

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
SPLIT_COL = "split"
ACTION_COL = "removed_in_period"
Y_CAUTI = "cauti_in_period"
Y_REINS = "reinsertion_in_period"
Y_DEATH = "death_in_period"
Y_ICU_EXIT = "icu_end_in_period"
TRANSITION_LABEL_COL = "next_state"
OBSERVED_ACTION_COL = "observed_action"
ACTION_REMOVE_COL = "action_remove"
LAST_PERIOD_COL = "is_last_period_of_episode"
END_REASON_COL = "episode_end_reason"


@dataclass(frozen=True)
class PanelBuildConfig:
    repo_root: Path
    mimic_dir: Path
    data_dir: Path
    config_dir: Path

    @property
    def d_items_path(self) -> Path:
        return self.mimic_dir / "icu" / "d_items.csv"

    @property
    def required_episodes_file(self) -> Path:
        return self.data_dir / "required_catheter_episodes.csv"

    @property
    def base_panel_file(self) -> Path:
        return self.data_dir / "base_panel.csv"

    @property
    def raw_chart_file(self) -> Path:
        return self.data_dir / "raw_chart_covariates.csv"

    @property
    def raw_chart_sample_file(self) -> Path:
        return self.data_dir / "raw_chart_covariates__first_1000_rows.csv"

    @property
    def preprocessed_chart_file(self) -> Path:
        return self.data_dir / "preprocessed_raw_chart_covariates.csv"

    @property
    def preprocessed_chart_sample_file(self) -> Path:
        return self.data_dir / "preprocessed_raw_chart_covariates__first_1000_rows.csv"

    @property
    def kept_preprocessed_chart_file(self) -> Path:
        return self.data_dir / "preprocessed_raw_chart_covariates_kept.csv"

    @property
    def kept_preprocessed_chart_sample_file(self) -> Path:
        return self.data_dir / "preprocessed_raw_chart_covariates_kept__first_1000_rows.csv"

    @property
    def cleaned_chart_file(self) -> Path:
        return self.data_dir / "cleaned_chart_covariates.csv"

    @property
    def cleaning_rules_file(self) -> Path:
        return self.data_dir / "chart_covariate_cleaning_rules.csv"

    @property
    def cleaning_audit_file(self) -> Path:
        return self.data_dir / "chart_covariate_cleaning_audit.csv"

    @property
    def master_panel_file(self) -> Path:
        return self.data_dir / "master_panel.csv"

    @property
    def covariate_retention_log_file(self) -> Path:
        return self.data_dir / "covariate_retention_log.csv"

    @property
    def filtered_panel_file(self) -> Path:
        return self.data_dir / "filtered_panel.csv"

    @property
    def train_test_split_file(self) -> Path:
        return self.data_dir / "train_test_split.csv"

    @property
    def modeling_panel_file(self) -> Path:
        return self.data_dir / "modeling_panel.csv"

    @property
    def feature_spec_file(self) -> Path:
        return self.data_dir / "feature_spec.json"

    @property
    def covariate_dictionary_file(self) -> Path:
        return self.data_dir / "covariate_dictionary.csv"

    @property
    def d_items_keep_file(self) -> Path:
        return self.config_dir / "d_items_keep.csv"

    @property
    def bounds_file(self) -> Path:
        return self.data_dir / "panel_covariate_bounds.csv"


# =============================================================================
# Generic helpers
# =============================================================================

def remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot remove existing file: {path}\n"
            "Close any program using it, including Excel/preview panes, and pause "
            "or let OneDrive finish syncing this folder before rerunning the pipeline."
        ) from exc


def replace_output(tmp_path: Path, final_path: Path) -> None:
    try:
        tmp_path.replace(final_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot replace output file: {final_path}\n"
            "Close any program using it, including Excel/preview panes, and pause "
            "or let OneDrive finish syncing this folder before rerunning the pipeline.\n"
            f"The completed replacement file is still available at: {tmp_path}"
        ) from exc


def load_item_labels(d_items_path: Path) -> dict[int, str]:
    if not d_items_path.exists():
        raise FileNotFoundError(f"d_items file not found: {d_items_path}")

    d_items_df = pd.read_csv(d_items_path, usecols=["itemid", "label"], low_memory=False).drop_duplicates("itemid")
    d_items_df["itemid"] = pd.to_numeric(d_items_df["itemid"], errors="coerce")
    d_items_df = d_items_df.dropna(subset=["itemid"]).copy()
    d_items_df["itemid"] = d_items_df["itemid"].astype(int)
    d_items_df["label"] = d_items_df["label"].astype(str)
    return d_items_df.set_index("itemid")["label"].to_dict()


def load_bounds(bounds_file: Path) -> dict[int, tuple[float, float]]:
    by_itemid: dict[int, tuple[float, float]] = {}

    if not bounds_file.exists():
        return by_itemid

    bounds_df = pd.read_csv(bounds_file)
    required_cols = {"itemid", "lower_bound", "upper_bound"}
    if not required_cols.issubset(bounds_df.columns):
        raise ValueError(f"Bounds file is missing required columns: {sorted(required_cols)}")

    bounds_rows = bounds_df[["itemid", "lower_bound", "upper_bound"]].copy()
    bounds_rows["itemid"] = pd.to_numeric(bounds_rows["itemid"], errors="coerce")
    bounds_rows["lower_bound"] = pd.to_numeric(bounds_rows["lower_bound"], errors="coerce")
    bounds_rows["upper_bound"] = pd.to_numeric(bounds_rows["upper_bound"], errors="coerce")
    bounds_rows = bounds_rows.dropna(subset=["itemid"]).copy()
    bounds_rows["itemid"] = bounds_rows["itemid"].astype(int)
    bounds_rows = bounds_rows.drop_duplicates(subset=["itemid"], keep="first")

    for _, row in bounds_rows.iterrows():
        by_itemid[int(row["itemid"])] = (row["lower_bound"], row["upper_bound"])

    return by_itemid


def print_section(title: str) -> float:
    print()
    print(f"=== {title} ===")
    print(time.strftime("Start: %Y-%m-%d %H:%M:%S"))
    return time.time()


def print_section_done(start_time: float) -> None:
    print(time.strftime("Done:  %Y-%m-%d %H:%M:%S"))
    print(f"Elapsed: {time.time() - start_time:.1f}s")


# =============================================================================
# Catheter episode cohort and base panel
# =============================================================================

def map_ethnicity_group(value):
    if pd.isna(value):
        return "Unknown"
    ethnicity_text = str(value).upper()
    if "WHITE" in ethnicity_text:
        return "White"
    if "BLACK" in ethnicity_text:
        return "Black"
    if "ASIAN" in ethnicity_text:
        return "Asian"
    if "HISPANIC" in ethnicity_text or "LATIN" in ethnicity_text:
        return "Hispanic"
    if "DECLINED" in ethnicity_text or "UNKNOWN" in ethnicity_text or "UNABLE" in ethnicity_text:
        return "Unknown"
    return "Other"


def merge_overlapping_foley_events(df: pd.DataFrame) -> pd.DataFrame:
    episodes = []

    for stay_id, stay_events in df.groupby("stay_id"):
        stay_events = stay_events.sort_values("inserted")
        current_start = None
        current_end = None

        for event in stay_events.itertuples():
            if current_start is None:
                current_start = event.inserted
                current_end = event.removed
                continue

            if event.inserted <= current_end:
                if pd.isna(current_end):
                    current_end = event.removed
                elif pd.notna(event.removed) and event.removed > current_end:
                    current_end = event.removed
            else:
                episodes.append((stay_id, current_start, current_end))
                current_start = event.inserted
                current_end = event.removed

        if current_start is not None:
            episodes.append((stay_id, current_start, current_end))

    return pd.DataFrame(episodes, columns=["stay_id", "inserted", "removed"])


def build_required_catheter_episodes(mimic_dir: Path) -> pd.DataFrame:
    icu = pd.read_csv(
        mimic_dir / "icu" / "icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["stay_id", "intime", "outtime"]).copy()

    patients = pd.read_csv(
        mimic_dir / "hosp" / "patients.csv",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        low_memory=False,
    )
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")
    patients["anchor_year"] = pd.to_numeric(patients["anchor_year"], errors="coerce")

    icu["icu_year"] = icu["intime"].dt.year
    icu = icu.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )
    icu["age"] = icu["anchor_age"] + (icu["icu_year"] - icu["anchor_year"])
    icu = icu.drop(columns=["icu_year", "anchor_age", "anchor_year"])

    admissions = pd.read_csv(
        mimic_dir / "hosp" / "admissions.csv",
        usecols=["subject_id", "hadm_id", "race", "deathtime"],
        low_memory=False,
    ).rename(columns={"race": "ethnicity"})
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")

    icu = icu.merge(
        admissions[["subject_id", "hadm_id", "ethnicity", "deathtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    icu["ethnicity_group"] = icu["ethnicity"].apply(map_ethnicity_group)

    procedure_events = pd.read_csv(
        mimic_dir / "icu" / "procedureevents.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"],
    )
    procedure_events = procedure_events[procedure_events["itemid"] == FOLEY_ITEMID].copy()
    procedure_events["starttime"] = pd.to_datetime(procedure_events["starttime"], errors="coerce")
    procedure_events["endtime"] = pd.to_datetime(procedure_events["endtime"], errors="coerce")

    procedure_events = procedure_events.merge(
        icu[["stay_id", "subject_id", "hadm_id", "intime", "outtime"]],
        on="stay_id",
        how="left",
    )
    procedure_events = procedure_events.rename(columns={"intime": "ICU_in", "outtime": "ICU_out"})
    procedure_events["inserted"] = procedure_events["starttime"]
    procedure_events["removed"] = procedure_events["endtime"]
    procedure_events.loc[procedure_events["removed"].isna(), "removed"] = procedure_events["ICU_out"]
    procedure_events = procedure_events.dropna(subset=["inserted", "removed", "ICU_in", "ICU_out"]).copy()
    procedure_events = procedure_events.sort_values(["stay_id", "inserted"])

    collapsed = merge_overlapping_foley_events(procedure_events)
    catheterised = collapsed.merge(
        icu[[
            "stay_id", "subject_id", "hadm_id", "intime", "outtime",
            "gender", "age", "ethnicity_group", "deathtime",
        ]],
        on="stay_id",
        how="inner",
    ).rename(columns={"intime": "ICU_in", "outtime": "ICU_out"})

    catheterised["death_time"] = catheterised["deathtime"].where(
        (catheterised["deathtime"] >= catheterised["ICU_in"]) &
        (catheterised["deathtime"] <= catheterised["ICU_out"])
    )
    catheterised = catheterised.drop(columns=["deathtime"])

    catheterised = catheterised.sort_values(["stay_id", "inserted"]).reset_index(drop=True)
    catheterised["reinsertion_time"] = catheterised.groupby("stay_id")["inserted"].shift(-1)

    episode_duration = catheterised["removed"] - catheterised["inserted"]
    catheterised = catheterised[episode_duration >= MIN_EPISODE_DURATION].copy()

    micro = pd.read_csv(
        mimic_dir / "hosp" / "microbiologyevents.csv",
        usecols=["subject_id", "hadm_id", "charttime", "spec_type_desc", "org_name"],
    )
    micro["charttime"] = pd.to_datetime(micro["charttime"], errors="coerce")
    micro = micro[
        micro["spec_type_desc"].str.contains("urine", case=False, na=False) &
        micro["org_name"].notna()
    ].copy()

    micro_matched = micro.merge(
        catheterised[["subject_id", "hadm_id", "stay_id", "inserted", "removed"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    micro_matched = micro_matched[
        (micro_matched["charttime"] >= micro_matched["inserted"]) &
        (micro_matched["charttime"] <= micro_matched["removed"] + pd.Timedelta(hours=48))
    ].copy()
    micro_matched = (
        micro_matched.sort_values("charttime")
        .drop_duplicates(["stay_id", "inserted"])
        [["stay_id", "inserted", "charttime"]]
        .rename(columns={"charttime": "cauti_time"})
    )

    catheterised = catheterised.merge(micro_matched, on=["stay_id", "inserted"], how="left")
    return catheterised.sort_values(["stay_id", "inserted"]).reset_index(drop=True)


def make_state_windows(state_start, state_end):
    if pd.isna(state_start) or pd.isna(state_end) or state_end <= state_start:
        return []

    rows = []
    window_start = state_start
    state_idx = 0

    while window_start < state_end:
        window_end = min(window_start + PERIOD_DURATION, state_end)
        interval_hours = round((window_end - window_start).total_seconds() / 3600.0, 2)
        rows.append((state_idx, window_start, window_end, interval_hours))
        window_start = window_end
        state_idx += 1

    return rows


def build_base_panel(catheterised: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for episode in catheterised.itertuples():
        for state_idx, period_start, period_end, interval_hours in make_state_windows(episode.inserted, episode.removed):
            rows.append({
                "subject_id": episode.subject_id,
                "hadm_id": episode.hadm_id,
                "stay_id": episode.stay_id,
                "inserted": episode.inserted,
                "removed": episode.removed,
                "reinsertion_time": episode.reinsertion_time,
                "catheter_state": "in",
                "state_index": state_idx,
                "period_start": period_start,
                "period_end": period_end,
                "interval_hours": interval_hours,
                "cauti_time": episode.cauti_time,
                "death_time": episode.death_time,
                "ICU_out": episode.ICU_out,
                "intime": episode.ICU_in,
                "gender": episode.gender,
                "age": episode.age,
                "ethnicity_group": episode.ethnicity_group,
            })

        out_state_end = episode.reinsertion_time if pd.notna(episode.reinsertion_time) else episode.ICU_out
        for state_idx, period_start, period_end, interval_hours in make_state_windows(episode.removed, out_state_end):
            rows.append({
                "subject_id": episode.subject_id,
                "hadm_id": episode.hadm_id,
                "stay_id": episode.stay_id,
                "inserted": episode.inserted,
                "removed": episode.removed,
                "reinsertion_time": episode.reinsertion_time,
                "catheter_state": "out",
                "state_index": state_idx,
                "period_start": period_start,
                "period_end": period_end,
                "interval_hours": interval_hours,
                "cauti_time": episode.cauti_time,
                "death_time": episode.death_time,
                "ICU_out": episode.ICU_out,
                "intime": episode.ICU_in,
                "gender": episode.gender,
                "age": episode.age,
                "ethnicity_group": episode.ethnicity_group,
            })

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(
        ["stay_id", "inserted", "period_start", "period_end", "catheter_state"]
    ).reset_index(drop=True)

    panel["episode_index"] = panel.groupby(["stay_id", "inserted"]).cumcount()
    panel["periods_in_state"] = panel["state_index"] + 1
    panel = panel.drop(columns=["state_index"])

    panel["removed_in_period"] = (
        (panel["catheter_state"] == "in") &
        (panel["removed"] > panel["period_start"]) &
        (panel["removed"] <= panel["period_end"])
    ).astype(int)

    panel["reinsertion_in_period"] = (
        (panel["catheter_state"] == "out") &
        panel["reinsertion_time"].notna() &
        (panel["reinsertion_time"] > panel["period_start"]) &
        (panel["reinsertion_time"] <= panel["period_end"])
    ).astype(int)

    panel["icu_end_in_period"] = (
        panel["ICU_out"].notna() &
        (panel["ICU_out"] > panel["period_start"]) &
        (panel["ICU_out"] <= panel["period_end"])
    ).astype(int)

    panel["cauti_in_period"] = (
        panel["cauti_time"].notna() &
        (panel["cauti_time"] > panel["period_start"]) &
        (panel["cauti_time"] <= panel["period_end"])
    ).astype(int)

    panel["death_in_period"] = (
        panel["death_time"].notna() &
        (panel["death_time"] > panel["period_start"]) &
        (panel["death_time"] <= panel["period_end"])
    ).astype(int)

    panel["next_state"] = "NO_EVENT_CONTINUE"
    panel.loc[(panel["icu_end_in_period"] == 1) & (panel["death_in_period"] == 0), "next_state"] = "ICU_EXIT_ALIVE"
    panel.loc[panel["death_in_period"] == 1, "next_state"] = "DEATH"
    panel.loc[panel["removed_in_period"] == 1, "next_state"] = "REMOVAL"
    panel.loc[panel["reinsertion_in_period"] == 1, "next_state"] = "REINSERTION"
    panel.loc[panel["cauti_in_period"] == 1, "next_state"] = "CAUTI"

    panel["at_risk_cauti"] = (
        (panel["catheter_state"] == "in") |
        ((panel["catheter_state"] == "out") & (panel["periods_in_state"] <= POST_REMOVE_RISK_PERIODS))
    ).astype(int)
    panel["at_risk_reinsertion"] = (panel["catheter_state"] == "out").astype(int)

    episode_keys = ["stay_id", "inserted"]
    panel["is_last_period_of_episode"] = 0
    last_row_index = panel.groupby(episode_keys)["period_end"].idxmax()
    panel.loc[last_row_index, "is_last_period_of_episode"] = 1

    panel["episode_end_reason"] = pd.NA
    panel.loc[
        (panel["is_last_period_of_episode"] == 1) & (panel["reinsertion_in_period"] == 1),
        "episode_end_reason",
    ] = "reinsertion"
    panel.loc[
        (panel["is_last_period_of_episode"] == 1) &
        (panel["episode_end_reason"].isna()) &
        (panel["icu_end_in_period"] == 1),
        "episode_end_reason",
    ] = "icu_end"

    panel["sex_M"] = (panel["gender"] == "M").astype(int)
    panel["sex_missing"] = panel["gender"].isna().astype(int)
    eth_dummies = pd.get_dummies(panel["ethnicity_group"], prefix="ethnicity")
    panel = pd.concat([panel, eth_dummies], axis=1)

    panel["cov_start"] = panel["period_start"] - LOOKBACK_DURATION
    panel["cov_end"] = panel["period_start"]
    panel["cov_start"] = panel[["cov_start", "intime"]].max(axis=1)
    panel["row_id"] = np.arange(1, len(panel) + 1)

    panel = panel.drop(columns=["cauti_time", "death_time", "gender", "ethnicity_group", "intime", "ICU_out"])
    non_ethnicity_cols = [c for c in panel.columns if not c.startswith("ethnicity_")]
    ethnicity_cols = sorted([c for c in panel.columns if c.startswith("ethnicity_")])
    ordered_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "inserted",
        "removed",
        "reinsertion_time",
        "catheter_state",
        "episode_index",
        "period_start",
        "period_end",
        "interval_hours",
        "periods_in_state",
        "removed_in_period",
        "reinsertion_in_period",
        "cauti_in_period",
        "death_in_period",
        "icu_end_in_period",
        "next_state",
        "is_last_period_of_episode",
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
    return panel[[c for c in ordered_cols if c in non_ethnicity_cols or c in ethnicity_cols]]


def create_episode_cohort_and_base_panel(config: PanelBuildConfig) -> None:
    config.data_dir.mkdir(exist_ok=True, parents=True)
    episodes = build_required_catheter_episodes(config.mimic_dir)
    base_panel = build_base_panel(episodes)

    episode_export = episodes[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "inserted",
            "removed",
            "reinsertion_time",
            "ICU_in",
            "ICU_out",
            "death_time",
        ]
    ].copy()

    episode_export.to_csv(config.required_episodes_file, index=False)
    base_panel.to_csv(config.base_panel_file, index=False)

    print("[SAVE]", config.required_episodes_file)
    print("[SAVE]", config.base_panel_file)
    print("Episodes:", len(episodes))
    print("Stays:", episodes["stay_id"].nunique())
    print("Base panel rows:", len(base_panel))


# =============================================================================
# Raw chart-event extraction
# =============================================================================

def build_chart_extraction_windows(episodes: pd.DataFrame, lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    windows = episodes.copy()
    window_end = windows["reinsertion_time"].where(windows["reinsertion_time"].notna(), windows["ICU_out"])
    window_start = windows["inserted"] - pd.Timedelta(hours=lookback_hours)
    window_start = windows[["ICU_in"]].assign(window_start=window_start).max(axis=1)

    windows = windows.assign(window_start=window_start, window_end=window_end)
    windows = windows.dropna(subset=["window_start", "window_end"]).copy()
    windows = windows[windows["window_end"] > windows["window_start"]].copy()

    merged_windows = []
    for stay_id, stay_windows in windows.groupby("stay_id"):
        stay_windows = stay_windows.sort_values("window_start")
        current_start = None
        current_end = None

        for row in stay_windows.itertuples():
            if current_start is None:
                current_start = row.window_start
                current_end = row.window_end
                continue

            if row.window_start <= current_end:
                current_end = max(current_end, row.window_end)
            else:
                merged_windows.append((stay_id, current_start, current_end))
                current_start = row.window_start
                current_end = row.window_end

        if current_start is not None:
            merged_windows.append((stay_id, current_start, current_end))

    return pd.DataFrame(merged_windows, columns=["stay_id", "window_start", "window_end"])


def extract_raw_chart_covariates(config: PanelBuildConfig) -> None:
    episodes = pd.read_csv(config.required_episodes_file, low_memory=False)
    for col in ["inserted", "removed", "reinsertion_time", "ICU_in", "ICU_out"]:
        episodes[col] = pd.to_datetime(episodes[col], errors="coerce")

    windows = build_chart_extraction_windows(episodes)

    print("[CONFIG]", config.mimic_dir)
    print("[Catheter episodes]", len(episodes))
    print("[Chart windows]", len(windows))
    print("[EHR] Extracting raw chartevents covariates...")

    chart_cols = [
        "subject_id", "hadm_id", "stay_id", "itemid", "charttime",
        "storetime", "valuenum", "value", "valueuom",
    ]

    config.data_dir.mkdir(exist_ok=True)
    remove_if_exists(config.raw_chart_file)
    remove_if_exists(config.raw_chart_sample_file)

    stay_ids = set(windows["stay_id"].dropna().astype(int).unique())
    chartevents_file = config.mimic_dir / "icu" / "chartevents.csv"
    kept_rows_total = 0
    sample_rows_written = 0
    first_write = True
    t0 = time.time()

    for chunk_idx, chunk in enumerate(
        pd.read_csv(chartevents_file, usecols=chart_cols, chunksize=CHUNK_ROWS, low_memory=False),
        start=1,
    ):
        t_chunk0 = time.time()
        chunk_filtered = chunk[chunk["stay_id"].isin(stay_ids)].copy()

        if len(chunk_filtered) == 0:
            kept = 0
        else:
            chunk_filtered["charttime"] = pd.to_datetime(chunk_filtered["charttime"], errors="coerce")
            chunk_filtered["storetime"] = pd.to_datetime(chunk_filtered["storetime"], errors="coerce")
            chunk_filtered = chunk_filtered.dropna(subset=["stay_id", "charttime"]).copy()

            if len(chunk_filtered) == 0:
                kept = 0
            else:
                matched = chunk_filtered.merge(windows, on="stay_id", how="inner")
                matched = matched[
                    (matched["charttime"] >= matched["window_start"]) &
                    (matched["charttime"] < matched["window_end"])
                ].copy()
                chunk_filtered = matched[chart_cols].drop_duplicates()
                kept = len(chunk_filtered)

        kept_rows_total += kept

        if kept > 0:
            chunk_filtered.to_csv(
                config.raw_chart_file,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False

            if sample_rows_written < SAMPLE_ROWS:
                sample_chunk = chunk_filtered.head(SAMPLE_ROWS - sample_rows_written).copy()
                sample_chunk.to_csv(
                    config.raw_chart_sample_file,
                    mode="w" if sample_rows_written == 0 else "a",
                    header=sample_rows_written == 0,
                    index=False,
                )
                sample_rows_written += len(sample_chunk)

        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} keep={kept:,} "
            f"cum_keep={kept_rows_total:,} dt={time.time() - t_chunk0:.1f}s"
        )

    print(f"[EHR] Raw chart covariates extracted. Total time: {time.time() - t0:.1f}s")
    print("[SAVE]", config.raw_chart_file)
    print("[SAVE SAMPLE]", config.raw_chart_sample_file)
    print("Rows:", kept_rows_total)


# =============================================================================
# Chart covariate preprocessing and allowlist filtering
# =============================================================================

def fahrenheit_to_celsius(values: pd.Series) -> pd.Series:
    return (values - 32.0) * (5.0 / 9.0)


def preprocess_raw_chart_covariates(config: PanelBuildConfig) -> None:
    config.data_dir.mkdir(exist_ok=True, parents=True)
    remove_if_exists(config.preprocessed_chart_file)
    remove_if_exists(config.preprocessed_chart_sample_file)

    first_write = True
    converted_rows_total = 0
    sample_rows_written = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(config.raw_chart_file, chunksize=CHUNK_ROWS, low_memory=False),
        start=1,
    ):
        unit_clean = chunk["valueuom"].fillna("").astype(str).str.strip().str.upper()
        itemids = pd.to_numeric(chunk["itemid"], errors="coerce")
        fahrenheit_mask = unit_clean.isin(FAHRENHEIT_UNITS) | itemids.eq(TEMP_F_ITEMID)

        if fahrenheit_mask.any():
            chunk.loc[fahrenheit_mask, "valuenum"] = pd.to_numeric(
                chunk.loc[fahrenheit_mask, "valuenum"],
                errors="coerce",
            )
            chunk.loc[fahrenheit_mask, "valuenum"] = fahrenheit_to_celsius(chunk.loc[fahrenheit_mask, "valuenum"])

            numeric_value = pd.to_numeric(chunk.loc[fahrenheit_mask, "value"], errors="coerce")
            numeric_mask = numeric_value.notna()
            if numeric_mask.any():
                converted_value = fahrenheit_to_celsius(numeric_value.loc[numeric_mask]).round(3)
                chunk.loc[numeric_value.loc[numeric_mask].index, "value"] = converted_value.astype(str)

            chunk.loc[fahrenheit_mask, "itemid"] = TEMP_C_ITEMID
            chunk.loc[fahrenheit_mask, "valueuom"] = CELSIUS_UNIT
            converted_rows_total += int(fahrenheit_mask.sum())

        chunk.to_csv(
            config.preprocessed_chart_file,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        if sample_rows_written < SAMPLE_ROWS:
            sample_chunk = chunk.head(SAMPLE_ROWS - sample_rows_written).copy()
            sample_chunk.to_csv(
                config.preprocessed_chart_sample_file,
                mode="w" if sample_rows_written == 0 else "a",
                header=sample_rows_written == 0,
                index=False,
            )
            sample_rows_written += len(sample_chunk)

        first_write = False
        print(
            f"[CHUNK {chunk_idx}] rows={len(chunk):,} "
            f"converted={int(fahrenheit_mask.sum()):,} "
            f"cum_converted={converted_rows_total:,}"
        )

    print("[SAVE]", config.preprocessed_chart_file)
    print("[SAVE SAMPLE]", config.preprocessed_chart_sample_file)
    print("Converted rows:", converted_rows_total)


def load_keep_itemids(config: PanelBuildConfig) -> set[int]:
    keep_df = pd.read_csv(config.d_items_keep_file, usecols=["itemid"], low_memory=False)
    keep_df["itemid"] = pd.to_numeric(keep_df["itemid"], errors="coerce")
    keep_df = keep_df.dropna(subset=["itemid"]).copy()
    return set(keep_df["itemid"].astype(int))


def filter_preprocessed_chart_covariates(config: PanelBuildConfig) -> None:
    config.data_dir.mkdir(exist_ok=True, parents=True)

    keep_itemids = load_keep_itemids(config)
    tmp_outfile = config.kept_preprocessed_chart_file.with_suffix(config.kept_preprocessed_chart_file.suffix + ".writing")
    tmp_sample_outfile = config.kept_preprocessed_chart_sample_file.with_suffix(
        config.kept_preprocessed_chart_sample_file.suffix + ".writing"
    )
    remove_if_exists(tmp_outfile)
    remove_if_exists(tmp_sample_outfile)

    first_write = True
    sample_rows_written = 0
    read_rows_total = 0
    kept_rows_total = 0

    print(f"[LOAD] keep itemids={len(keep_itemids):,} from {config.d_items_keep_file}")

    for chunk_idx, chunk in enumerate(
        pd.read_csv(config.preprocessed_chart_file, chunksize=CHUNK_ROWS, low_memory=False),
        start=1,
    ):
        read_rows_total += len(chunk)
        itemids = pd.to_numeric(chunk["itemid"], errors="coerce")
        kept = chunk.loc[itemids.isin(keep_itemids)].copy()
        kept_rows_total += len(kept)

        if len(kept) > 0:
            kept.to_csv(tmp_outfile, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False

            if sample_rows_written < SAMPLE_ROWS:
                sample_chunk = kept.head(SAMPLE_ROWS - sample_rows_written).copy()
                sample_chunk.to_csv(
                    tmp_sample_outfile,
                    mode="w" if sample_rows_written == 0 else "a",
                    header=sample_rows_written == 0,
                    index=False,
                )
                sample_rows_written += len(sample_chunk)

        print(f"[CHUNK {chunk_idx}] rows={len(chunk):,} keep={len(kept):,} cum_keep={kept_rows_total:,}")

    if first_write:
        header = pd.read_csv(config.preprocessed_chart_file, nrows=0)
        header.to_csv(tmp_outfile, index=False)
        print(f"[WARN] no rows matched d_items_keep.csv; wrote empty file with headers: {config.kept_preprocessed_chart_file}")

    replace_output(tmp_outfile, config.kept_preprocessed_chart_file)
    print(f"[SAVE] {config.kept_preprocessed_chart_file}")
    if tmp_sample_outfile.exists():
        replace_output(tmp_sample_outfile, config.kept_preprocessed_chart_sample_file)
        print(f"[SAVE SAMPLE] {config.kept_preprocessed_chart_sample_file}")
    print(f"[INFO] rows read: {read_rows_total:,}")
    print(f"[INFO] rows kept: {kept_rows_total:,}")
    print(f"[INFO] rows dropped: {read_rows_total - kept_rows_total:,}")


# =============================================================================
# Chart covariate validation
# =============================================================================

def build_chart_value_audit_from_parts(
    value_parts: dict[int, list[pd.Series]],
    row_counts: dict[int, int],
    unit_counts_by_itemid: dict[int, dict[str, int]],
    itemid_to_label: dict[int, str],
    bounds_by_itemid: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for itemid in sorted(row_counts):
        non_missing = (
            pd.concat(value_parts[itemid], ignore_index=True)
            if value_parts[itemid]
            else pd.Series(dtype=float)
        )

        rows_for_itemid = int(row_counts[itemid])
        non_missing_n = int(non_missing.shape[0])
        pct_missing = float((1.0 - (non_missing_n / rows_for_itemid)) * 100.0) if rows_for_itemid > 0 else np.nan
        lower_bound, upper_bound = bounds_by_itemid.get(itemid, (np.nan, np.nan))

        if non_missing_n > 0:
            min_val = float(non_missing.min())
            p01 = float(non_missing.quantile(0.01))
            median_val = float(non_missing.median())
            p99 = float(non_missing.quantile(0.99))
            max_val = float(non_missing.max())
        else:
            min_val = p01 = median_val = p99 = max_val = np.nan

        n_below = int((non_missing < lower_bound).sum()) if pd.notna(lower_bound) else np.nan
        n_above = int((non_missing > upper_bound).sum()) if pd.notna(upper_bound) else np.nan

        if pd.notna(n_below) and pd.notna(n_above) and non_missing_n > 0:
            n_out_of_range = int(n_below + n_above)
            pct_out_of_range_non_missing = float((n_out_of_range / non_missing_n) * 100.0)
        else:
            n_out_of_range = np.nan
            pct_out_of_range_non_missing = np.nan

        unit_counts = pd.Series(unit_counts_by_itemid[itemid]).sort_values(ascending=False)
        if len(unit_counts) > 0:
            n_unique_units = int(unit_counts.shape[0])
            top_unit = str(unit_counts.index[0])
            top_unit_n = int(unit_counts.iloc[0])
            units_seen = " | ".join([f"{u} ({n})" for u, n in unit_counts.items()])
        else:
            n_unique_units = 0
            top_unit = ""
            top_unit_n = 0
            units_seen = ""

        rows.append({
            "itemid": itemid,
            "label": itemid_to_label.get(itemid, "UNKNOWN ITEMID"),
            "rows_for_itemid": rows_for_itemid,
            "non_missing_n": non_missing_n,
            "pct_missing": pct_missing,
            "min": min_val,
            "p01": p01,
            "median": median_val,
            "p99": p99,
            "max": max_val,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_below_lower_bound": n_below,
            "n_above_upper_bound": n_above,
            "n_out_of_range": n_out_of_range,
            "pct_out_of_range_non_missing": pct_out_of_range_non_missing,
            "n_unique_units": n_unique_units,
            "top_unit": top_unit,
            "top_unit_n": top_unit_n,
            "units_seen": units_seen,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].round(AUDIT_DECIMAL_PLACES)
    return out


def build_unit_audit_from_parts(
    unit_counts_by_itemid: dict[int, dict[str, int]],
    itemid_to_label: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for itemid in sorted(unit_counts_by_itemid):
        for unit, n_rows in unit_counts_by_itemid[itemid].items():
            rows.append({
                "itemid": itemid,
                "label": itemid_to_label.get(itemid, "UNKNOWN ITEMID"),
                "valueuom": unit,
                "n_rows": n_rows,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out.reindex(columns=["itemid", "label", "valueuom", "n_rows"])

    return out.sort_values(
        ["itemid", "n_rows", "valueuom"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def validate_chart_covariates(
    data_file: Path,
    outdir: Path,
    d_items_path: Path,
    bounds_file: Path,
    output_prefix: str,
) -> None:
    outdir.mkdir(exist_ok=True, parents=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Chart covariate file not found: {data_file}")

    header = pd.read_csv(data_file, nrows=0)
    header.columns = header.columns.str.strip()
    available_cols = set(header.columns)
    desired_cols = ["stay_id", "itemid", "charttime", "valuenum", "value", "valueuom"]
    usecols = [c for c in desired_cols if c in available_cols]

    print(f"[RAW FILE] {data_file}")
    print(f"[AVAILABLE COLS] {sorted(available_cols)}")
    print(f"[READING COLS] {usecols}")

    if "itemid" not in available_cols:
        raise ValueError("Chart file must contain 'itemid'.")
    if "valuenum" not in available_cols and "value" not in available_cols:
        raise ValueError("Chart file must contain at least one of 'valuenum' or 'value'.")

    itemid_to_label = load_item_labels(d_items_path)
    bounds_by_itemid = load_bounds(bounds_file)
    row_counts: dict[int, int] = defaultdict(int)
    value_parts: dict[int, list[pd.Series]] = defaultdict(list)
    unit_counts_by_itemid: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for chunk in pd.read_csv(data_file, usecols=usecols, chunksize=CHUNK_ROWS, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        if "valuenum" in chunk.columns:
            chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        else:
            chunk["valuenum"] = np.nan
        if "valueuom" in chunk.columns:
            chunk["valueuom"] = chunk["valueuom"].fillna("").astype(str).str.strip()
        else:
            chunk["valueuom"] = ""

        chunk = chunk.dropna(subset=["itemid"]).copy()
        chunk["itemid"] = chunk["itemid"].astype(int)

        row_count_chunk = chunk.groupby("itemid").size()
        for itemid, count in row_count_chunk.items():
            row_counts[int(itemid)] += int(count)

        numeric_chunk = chunk.dropna(subset=["valuenum"])
        for itemid, group in numeric_chunk.groupby("itemid", sort=False):
            value_parts[int(itemid)].append(group["valuenum"].reset_index(drop=True))

        unit_chunk = chunk.loc[chunk["valueuom"] != "", ["itemid", "valueuom"]].copy()
        if len(unit_chunk) > 0:
            unit_count_chunk = unit_chunk.groupby(["itemid", "valueuom"]).size()
            for (itemid, unit), count in unit_count_chunk.items():
                unit_counts_by_itemid[int(itemid)][str(unit)] += int(count)

    audit = build_chart_value_audit_from_parts(
        value_parts=value_parts,
        row_counts=row_counts,
        unit_counts_by_itemid=unit_counts_by_itemid,
        itemid_to_label=itemid_to_label,
        bounds_by_itemid=bounds_by_itemid,
    )

    audit_file = outdir / f"{output_prefix}_chart_integrity_audit.csv"
    audit_problem_file = outdir / f"{output_prefix}_chart_integrity_audit__sorted_problem_first.csv"
    units_file = outdir / f"{output_prefix}_chart_units_by_itemid.csv"

    audit.to_csv(audit_file, index=False)

    audit_problem = audit.copy()
    audit_problem["abs_max_minus_p99"] = (
        pd.to_numeric(audit_problem["max"], errors="coerce") -
        pd.to_numeric(audit_problem["p99"], errors="coerce")
    ).abs()
    audit_problem["abs_p01_minus_min"] = (
        pd.to_numeric(audit_problem["p01"], errors="coerce") -
        pd.to_numeric(audit_problem["min"], errors="coerce")
    ).abs()

    sort_cols = [
        "n_out_of_range",
        "pct_out_of_range_non_missing",
        "n_unique_units",
        "abs_max_minus_p99",
        "abs_p01_minus_min",
        "pct_missing",
    ]
    audit_problem = audit_problem.sort_values(sort_cols, ascending=[False, False, False, False, False, False])
    audit_problem.to_csv(audit_problem_file, index=False)

    unit_audit = build_unit_audit_from_parts(unit_counts_by_itemid, itemid_to_label)
    unit_audit.to_csv(units_file, index=False)

    print(f"Saved: {audit_file}")
    print(f"Saved: {audit_problem_file}")
    print(f"Saved: {units_file}")


def validate_raw_chart_covariates(config: PanelBuildConfig) -> None:
    validate_chart_covariates(
        data_file=config.kept_preprocessed_chart_file,
        outdir=config.data_dir,
        d_items_path=config.d_items_path,
        bounds_file=config.bounds_file,
        output_prefix="raw",
    )


def validate_cleaned_chart_covariates(config: PanelBuildConfig) -> None:
    validate_chart_covariates(
        data_file=config.cleaned_chart_file,
        outdir=config.data_dir,
        d_items_path=config.d_items_path,
        bounds_file=config.bounds_file,
        output_prefix="cleaned",
    )


# =============================================================================
# Chart covariate cleaning
# =============================================================================

def _load_numeric_values_by_itemid(infile: Path) -> dict[int, pd.Series]:
    value_parts: dict[int, list[pd.Series]] = defaultdict(list)

    for chunk in pd.read_csv(infile, usecols=[ITEM_COL, VALUE_COL], chunksize=CHUNK_ROWS, low_memory=False):
        chunk[ITEM_COL] = pd.to_numeric(chunk[ITEM_COL], errors="coerce")
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[ITEM_COL, VALUE_COL]).copy()
        if chunk.empty:
            continue

        chunk[ITEM_COL] = chunk[ITEM_COL].astype(int)
        for itemid, item_rows in chunk.groupby(ITEM_COL, sort=False):
            value_parts[int(itemid)].append(item_rows[VALUE_COL].reset_index(drop=True))

    return {
        itemid: pd.concat(parts, ignore_index=True)
        for itemid, parts in value_parts.items()
        if parts
    }


def _count_numeric_values_by_itemid(infile: Path) -> pd.DataFrame:
    count_parts = []

    for chunk in pd.read_csv(infile, usecols=[ITEM_COL, VALUE_COL], chunksize=CHUNK_ROWS, low_memory=False):
        chunk[ITEM_COL] = pd.to_numeric(chunk[ITEM_COL], errors="coerce")
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[ITEM_COL]).copy()
        if chunk.empty:
            continue

        count_parts.append(
            chunk.groupby(ITEM_COL, sort=False).agg(
                n_rows_total=(VALUE_COL, "size"),
                n_non_missing=(VALUE_COL, lambda values: values.notna().sum()),
            )
        )

    if not count_parts:
        return pd.DataFrame(columns=[ITEM_COL, "n_rows_total", "n_non_missing"])

    counts = pd.concat(count_parts).groupby(level=0, sort=False).sum().reset_index()
    counts[ITEM_COL] = counts[ITEM_COL].astype(int)
    return counts


def fit_cleaning_rules(infile: Path) -> pd.DataFrame:
    values_by_itemid = _load_numeric_values_by_itemid(infile)
    if not values_by_itemid:
        raise ValueError("No numeric valuenum rows found in input file.")

    rule_rows = []
    for itemid, values in values_by_itemid.items():
        nonzero_values = values[values != 0]
        rule_rows.append(
            {
                ITEM_COL: itemid,
                "n_non_missing": int(values.shape[0]),
                "zero_fraction": float(values.eq(0).mean()),
                "p5_nonzero": float(nonzero_values.quantile(0.05)) if not nonzero_values.empty else np.nan,
            }
        )

    rules = pd.DataFrame(rule_rows).set_index(ITEM_COL)
    rules["zero_to_missing"] = False

    auto_zero_mask = (
        (rules["n_non_missing"] >= MIN_N_FOR_RULES) &
        (rules["p5_nonzero"] > 0) &
        (rules["zero_fraction"] > 0) &
        (rules["zero_fraction"] <= ZERO_MAX_FRAC)
    )
    rules.loc[auto_zero_mask, "zero_to_missing"] = True

    if ALWAYS_ZERO_TO_MISSING:
        rules.loc[rules.index.isin(ALWAYS_ZERO_TO_MISSING), "zero_to_missing"] = True
    if NEVER_ZERO_TO_MISSING:
        rules.loc[rules.index.isin(NEVER_ZERO_TO_MISSING), "zero_to_missing"] = False

    threshold_rows = []
    for itemid, values in values_by_itemid.items():
        filtered_values = values.copy()
        if bool(rules.loc[itemid, "zero_to_missing"]):
            filtered_values = filtered_values[filtered_values != 0]

        if filtered_values.empty:
            threshold_rows.append({
                ITEM_COL: itemid,
                "n_for_thresholds": 0,
                "p1": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "p99": np.nan,
            })
            continue

        threshold_rows.append({
            ITEM_COL: itemid,
            "n_for_thresholds": int(filtered_values.shape[0]),
            "p1": float(filtered_values.quantile(0.01)),
            "q1": float(filtered_values.quantile(0.25)),
            "q3": float(filtered_values.quantile(0.75)),
            "p99": float(filtered_values.quantile(0.99)),
        })

    threshold_df = pd.DataFrame(threshold_rows).set_index(ITEM_COL)
    rules = rules.join(threshold_df, how="left")
    rules["status"] = "ok"
    rules.loc[rules["n_for_thresholds"].fillna(0) < MIN_N_FOR_RULES, "status"] = "too_few_values_for_thresholds"

    iqr = rules["q3"] - rules["q1"]
    tail_span = rules["p99"] - rules["p1"]
    spread = pd.concat([iqr, tail_span], axis=1).max(axis=1)
    spread = spread.fillna(0.0).clip(lower=1e-8)

    rules["lower_clip"] = rules["p1"]
    rules["upper_clip"] = rules["p99"]
    rules["lower_delete"] = rules["p1"] - FAR_OUT_SPREAD_MULT * spread
    rules["upper_delete"] = rules["p99"] + FAR_OUT_SPREAD_MULT * spread

    return rules.reset_index()


def apply_cleaning_rules(infile: Path, outfile: Path, rules: pd.DataFrame) -> pd.DataFrame:
    remove_if_exists(outfile)

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

    audit_parts = []
    first_write = True

    for chunk in pd.read_csv(infile, chunksize=CHUNK_ROWS, low_memory=False):
        original_columns = list(chunk.columns)
        chunk[ITEM_COL] = pd.to_numeric(chunk[ITEM_COL], errors="coerce")
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL], errors="coerce")

        chunk = chunk.merge(rules_small, on=ITEM_COL, how="left")

        action = pd.Series("unchanged", index=chunk.index, dtype="object")
        action.loc[chunk[VALUE_COL].isna()] = "original_missing_or_non_numeric"
        action.loc[chunk["status"].isna() & chunk[VALUE_COL].notna()] = "no_rule"

        bad_rule_mask = chunk["status"].notna() & chunk["status"].ne("ok") & chunk[VALUE_COL].notna()
        action.loc[bad_rule_mask] = "rule_not_applied"

        zero_mask = (
            chunk["status"].eq("ok") &
            chunk["zero_to_missing"].fillna(False) &
            chunk[VALUE_COL].eq(0)
        )
        chunk.loc[zero_mask, VALUE_COL] = np.nan
        action.loc[zero_mask] = "zero_to_missing"

        far_low_mask = chunk["status"].eq("ok") & chunk[VALUE_COL].notna() & (chunk[VALUE_COL] < chunk["lower_delete"])
        far_high_mask = chunk["status"].eq("ok") & chunk[VALUE_COL].notna() & (chunk[VALUE_COL] > chunk["upper_delete"])
        far_mask = far_low_mask | far_high_mask
        chunk.loc[far_mask, VALUE_COL] = np.nan
        action.loc[far_mask] = "far_out_to_missing"

        clip_low_mask = chunk["status"].eq("ok") & chunk[VALUE_COL].notna() & (chunk[VALUE_COL] < chunk["lower_clip"])
        clip_high_mask = chunk["status"].eq("ok") & chunk[VALUE_COL].notna() & (chunk[VALUE_COL] > chunk["upper_clip"])

        chunk.loc[clip_low_mask, VALUE_COL] = chunk.loc[clip_low_mask, "lower_clip"]
        chunk.loc[clip_high_mask, VALUE_COL] = chunk.loc[clip_high_mask, "upper_clip"]
        clip_mask = (clip_low_mask | clip_high_mask) & action.eq("unchanged")
        action.loc[clip_mask] = "tail_clipped"

        audit_chunk = (
            pd.DataFrame({ITEM_COL: chunk[ITEM_COL], "action": action})
            .groupby([ITEM_COL, "action"], dropna=False)
            .size()
            .rename("n_rows")
            .reset_index()
        )
        audit_parts.append(audit_chunk)

        chunk = chunk[original_columns]
        chunk.to_csv(outfile, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    audit_long = pd.concat(audit_parts, ignore_index=True)
    audit_long = audit_long.groupby([ITEM_COL, "action"], dropna=False)["n_rows"].sum().reset_index()

    audit_wide = audit_long.pivot(index=ITEM_COL, columns="action", values="n_rows").fillna(0)
    audit_wide.columns.name = None
    audit_wide = audit_wide.reset_index()

    before_counts = _count_numeric_values_by_itemid(infile).rename(columns={"n_non_missing": "n_non_missing_before"})
    after_counts = _count_numeric_values_by_itemid(outfile).rename(columns={"n_non_missing": "n_non_missing_after"})[
        [ITEM_COL, "n_non_missing_after"]
    ]

    audit = (
        rules.merge(audit_wide, on=ITEM_COL, how="left")
        .merge(before_counts, on=ITEM_COL, how="left")
        .merge(after_counts, on=ITEM_COL, how="left")
    )

    rule_cols = {
        ITEM_COL,
        "n_non_missing",
        "zero_fraction",
        "p5_nonzero",
        "zero_to_missing",
        "n_for_thresholds",
        "p1",
        "q1",
        "q3",
        "p99",
        "status",
        "lower_clip",
        "upper_clip",
        "lower_delete",
        "upper_delete",
    }
    for col in audit.columns:
        if col not in rule_cols:
            audit[col] = audit[col].fillna(0)

    return audit


def clean_chart_covariates(config: PanelBuildConfig) -> None:
    print("[FIT RULES]", config.kept_preprocessed_chart_file)
    rules = fit_cleaning_rules(config.kept_preprocessed_chart_file)
    rules.to_csv(config.cleaning_rules_file, index=False)
    print("[SAVE RULES]", config.cleaning_rules_file)

    print("[APPLY RULES]", config.kept_preprocessed_chart_file)
    audit = apply_cleaning_rules(config.kept_preprocessed_chart_file, config.cleaned_chart_file, rules)
    audit.to_csv(config.cleaning_audit_file, index=False)

    print("[SAVE CLEANED]", config.cleaned_chart_file)
    print("[SAVE AUDIT]", config.cleaning_audit_file)


# =============================================================================
# Master panel aggregation
# =============================================================================

def aggregate_itemid_covariates(panel: pd.DataFrame, cleaned_chart_file: Path) -> pd.DataFrame:
    windows = panel[["row_id", "stay_id", "cov_start", "cov_end"]].copy()
    stay_ids = set(windows["stay_id"].dropna().astype(int).unique())

    chart_cols = ["stay_id", "itemid", "charttime", "valuenum"]
    partial_stats = []
    first_obs_parts = []
    last_obs_parts = []

    kept_rows_total = 0
    t0 = time.time()

    print("[Base panel rows]", len(panel))
    print("[EHR] Aggregating chartevents covariates...")

    for chunk_idx, chunk in enumerate(
        pd.read_csv(cleaned_chart_file, usecols=chart_cols, chunksize=CHUNK_ROWS, low_memory=False),
        start=1,
    ):
        t_chunk0 = time.time()
        chunk_filtered = chunk[chunk["stay_id"].isin(stay_ids)].copy()
        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} keep=0 dt={time.time() - t_chunk0:.1f}s")
            continue

        chunk_filtered["charttime"] = pd.to_datetime(chunk_filtered["charttime"], errors="coerce")
        chunk_filtered["valuenum"] = pd.to_numeric(chunk_filtered["valuenum"], errors="coerce")
        chunk_filtered = chunk_filtered.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"]).copy()

        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} numeric_keep=0 dt={time.time() - t_chunk0:.1f}s")
            continue

        merged = chunk_filtered.merge(windows, on="stay_id", how="inner")
        merged = merged[(merged["charttime"] >= merged["cov_start"]) & (merged["charttime"] < merged["cov_end"])].copy()

        kept = len(merged)
        kept_rows_total += kept

        if kept > 0:
            merged["charttime_seconds"] = merged["charttime"].astype("int64") / 1_000_000_000.0
            merged["valuenum_sq"] = merged["valuenum"] * merged["valuenum"]
            merged["charttime_sq"] = merged["charttime_seconds"] * merged["charttime_seconds"]
            merged["charttime_value"] = merged["charttime_seconds"] * merged["valuenum"]

            item_summary = (
                merged.groupby(["row_id", "itemid"])
                .agg(
                    count=("valuenum", "count"),
                    total=("valuenum", "sum"),
                    sum_sq=("valuenum_sq", "sum"),
                    min=("valuenum", "min"),
                    max=("valuenum", "max"),
                    sum_t=("charttime_seconds", "sum"),
                    sum_tt=("charttime_sq", "sum"),
                    sum_ty=("charttime_value", "sum"),
                )
                .reset_index()
            )
            partial_stats.append(item_summary)

            first_obs_parts.append(
                merged.sort_values(["row_id", "itemid", "charttime"])
                .drop_duplicates(["row_id", "itemid"], keep="first")
                [["row_id", "itemid", "charttime", "valuenum"]]
                .rename(columns={"charttime": "first_time", "valuenum": "first"})
            )
            last_obs_parts.append(
                merged.sort_values(["row_id", "itemid", "charttime"])
                .drop_duplicates(["row_id", "itemid"], keep="last")
                [["row_id", "itemid", "charttime", "valuenum"]]
                .rename(columns={"charttime": "last_time", "valuenum": "last"})
            )

        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} stay_filtered={len(chunk_filtered):,} "
            f"matched={kept:,} cum_matched={kept_rows_total:,} dt={time.time() - t_chunk0:.1f}s"
        )

    if not partial_stats:
        print(f"[EHR] No chartevents matched windows. Total time: {time.time() - t0:.1f}s")
        return panel

    agg = pd.concat(partial_stats, ignore_index=True)
    agg = (
        agg.groupby(["row_id", "itemid"])
        .agg(
            count=("count", "sum"),
            total=("total", "sum"),
            sum_sq=("sum_sq", "sum"),
            min=("min", "min"),
            max=("max", "max"),
            sum_t=("sum_t", "sum"),
            sum_tt=("sum_tt", "sum"),
            sum_ty=("sum_ty", "sum"),
        )
        .reset_index()
    )

    agg["mean"] = agg["total"] / agg["count"]
    agg["range"] = agg["max"] - agg["min"]

    variance_num = agg["sum_sq"] - (agg["total"] * agg["total"] / agg["count"])
    agg["std"] = np.where(
        agg["count"] > 1,
        np.sqrt((variance_num / (agg["count"] - 1)).clip(lower=0)),
        np.nan,
    )

    if first_obs_parts:
        first_obs = (
            pd.concat(first_obs_parts, ignore_index=True)
            .sort_values(["row_id", "itemid", "first_time"])
            .drop_duplicates(["row_id", "itemid"], keep="first")
        )
        agg = agg.merge(first_obs[["row_id", "itemid", "first"]], on=["row_id", "itemid"], how="left")

    if last_obs_parts:
        last_obs = (
            pd.concat(last_obs_parts, ignore_index=True)
            .sort_values(["row_id", "itemid", "last_time"])
            .drop_duplicates(["row_id", "itemid"], keep="last")
        )
        agg = agg.merge(last_obs[["row_id", "itemid", "last"]], on=["row_id", "itemid"], how="left")

    agg["delta"] = agg["last"] - agg["first"]

    slope_denom = agg["count"] * agg["sum_tt"] - agg["sum_t"] * agg["sum_t"]
    slope_num = agg["count"] * agg["sum_ty"] - agg["sum_t"] * agg["total"]
    agg["slope_per_hour"] = np.where(
        (agg["count"] > 1) & (slope_denom != 0),
        (slope_num / slope_denom) * 3600.0,
        np.nan,
    )

    value_cols = [c for c in AGG_STATS if c in agg.columns]
    wide = agg.pivot_table(index="row_id", columns="itemid", values=value_cols, aggfunc="first")
    wide.columns = [f"itemid_{itemid}__{stat}" for stat, itemid in wide.columns]
    wide = wide.reset_index()

    panel = panel.merge(wide, on="row_id", how="left")
    print(f"[EHR] Covariates aggregated. Total time: {time.time() - t0:.1f}s")
    return panel


def build_master_panel(config: PanelBuildConfig) -> None:
    panel = pd.read_csv(config.base_panel_file, low_memory=False)
    for col in ["inserted", "removed", "reinsertion_time", "period_start", "period_end", "cov_start", "cov_end"]:
        panel[col] = pd.to_datetime(panel[col], errors="coerce")

    panel = aggregate_itemid_covariates(panel, config.cleaned_chart_file)
    panel = panel.drop(columns=["cov_start", "cov_end", "row_id"])

    covariate_cols = sorted([c for c in panel.columns if c.startswith("itemid_")])
    base_cols = [c for c in panel.columns if not c.startswith("itemid_")]
    panel = panel[base_cols + covariate_cols]
    panel.to_csv(config.master_panel_file, index=False)

    print("[SAVE]", config.master_panel_file)
    print("Removals:", panel["removed_in_period"].sum())
    print("Reinsertions:", panel["reinsertion_in_period"].sum())
    print("CAUTI:", panel["cauti_in_period"].sum())
    print("ICU end rows:", panel["icu_end_in_period"].sum())
    print("Last episode periods:", panel["is_last_period_of_episode"].sum())
    print("Rows:", len(panel))


# =============================================================================
# Covariate retention
# =============================================================================

def detect_covariate_cols(columns: list[str]) -> list[dict[str, object]]:
    if MEAN_ONLY:
        col_pattern = re.compile(r"^itemid_(\d+)__mean$")
    else:
        col_pattern = re.compile(r"^itemid_(\d+)__([a-z0-9_]+)$", flags=re.IGNORECASE)

    covariates: list[dict[str, object]] = []
    for column_name in columns:
        match = col_pattern.match(column_name)
        if match:
            covariates.append({
                "column_name": column_name,
                "itemid": int(match.group(1)),
                "stat": "mean" if MEAN_ONLY else match.group(2),
            })
    return covariates


def decide_retention(row_cov: float, stay_cov: float) -> tuple[str, str]:
    keep_col = row_cov >= MIN_ROW_COVERAGE and stay_cov >= MIN_STAY_COVERAGE
    return ("retain" if keep_col else "drop", "coverage")


def build_retention_log(
    panel: pd.DataFrame,
    covariates: list[dict[str, object]],
    itemid_to_label: dict[int, str],
) -> pd.DataFrame:
    total_rows = len(panel)
    total_stays = panel["stay_id"].nunique()
    stay_ids = panel["stay_id"]

    rows: list[dict[str, object]] = []
    for covariate in covariates:
        column_name = str(covariate["column_name"])
        itemid = int(covariate["itemid"])
        stat = str(covariate["stat"])
        values = panel[column_name]

        n_rows = int(values.notna().sum())
        row_cov = n_rows / total_rows if total_rows else 0.0
        has_value_by_stay = values.notna().groupby(stay_ids, sort=False).any()
        n_stays = int(has_value_by_stay.sum())
        stay_cov = n_stays / total_stays if total_stays else 0.0

        decision, reason = decide_retention(row_cov, stay_cov)
        rows.append({
            "itemid": itemid,
            "label": itemid_to_label.get(itemid, "UNKNOWN ITEMID"),
            "stat": stat,
            "column_name": column_name,
            "n_rows": n_rows,
            "row_coverage": round(row_cov, 4),
            "n_stays": n_stays,
            "stay_coverage": round(stay_cov, 4),
            "decision": decision,
            "selection_reason": reason,
        })

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(
        ["decision", "row_coverage", "stay_coverage", "n_rows", "itemid", "stat"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)


def select_retained_covariates(config: PanelBuildConfig) -> None:
    all_columns = pd.read_csv(config.master_panel_file, nrows=0).columns.tolist()
    covariates = detect_covariate_cols(all_columns)
    usecols = ["stay_id"] + [str(covariate["column_name"]) for covariate in covariates]

    print(f"[LOAD] {config.master_panel_file}")
    print(f"[INFO] loading stay_id + {len(covariates):,} covariate columns")

    panel = pd.read_csv(config.master_panel_file, usecols=usecols, low_memory=False)
    summary_df = build_retention_log(panel, covariates, load_item_labels(config.d_items_path))
    summary_df.to_csv(config.covariate_retention_log_file, index=False)

    retained_total = int((summary_df["decision"] == "retain").sum())
    dropped_total = int((summary_df["decision"] == "drop").sum())

    print(f"[INFO] panel rows: {len(panel):,}")
    print(f"[INFO] catheterised stays: {panel['stay_id'].nunique():,}")
    print(f"[INFO] thresholds: row_coverage >= {MIN_ROW_COVERAGE}, stay_coverage >= {MIN_STAY_COVERAGE}")
    print(f"[WRITE] {config.covariate_retention_log_file}")
    print(f"[INFO] retained columns: {retained_total:,}")
    print(f"[INFO] dropped columns: {dropped_total:,}")


# =============================================================================
# Filtered panel and train/test split
# =============================================================================

def create_patient_split(panel: pd.DataFrame) -> pd.DataFrame:
    subject_ids = panel[SUBJECT_ID_COL].dropna().astype(str).unique()
    train_ids, test_ids = train_test_split(subject_ids, test_size=TEST_SIZE, random_state=SEED)

    return pd.DataFrame({
        SUBJECT_ID_COL: list(train_ids) + list(test_ids),
        "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
    })


def build_filtered_panel(config: PanelBuildConfig) -> None:
    summary_df = pd.read_csv(config.covariate_retention_log_file, low_memory=False)
    retained_cov_cols = summary_df.loc[
        summary_df["decision"].astype(str).str.lower() == "retain",
        "column_name",
    ].astype(str).tolist()

    panel = pd.read_csv(config.master_panel_file, low_memory=False)
    panel[SUBJECT_ID_COL] = panel[SUBJECT_ID_COL].astype(str)

    covariate_cols = [c for c in panel.columns if c.startswith("itemid_") and "__" in c]
    base_cols = [c for c in panel.columns if c not in covariate_cols]
    retained_cov_set = set(retained_cov_cols)
    kept_cov_cols = [c for c in covariate_cols if c in retained_cov_set]
    filtered_panel = panel[base_cols + kept_cov_cols].copy()

    for column_name in kept_cov_cols:
        filtered_panel[column_name] = pd.to_numeric(filtered_panel[column_name], errors="coerce")
    filtered_panel[kept_cov_cols] = filtered_panel[kept_cov_cols].round(ROUND_DP)

    split_df = create_patient_split(filtered_panel)
    split_df[SUBJECT_ID_COL] = split_df[SUBJECT_ID_COL].astype(str)
    filtered_panel = filtered_panel.merge(split_df, on=SUBJECT_ID_COL, how="left")

    filtered_panel.to_csv(config.filtered_panel_file, index=False)
    split_df.to_csv(config.train_test_split_file, index=False)

    patient_total = split_df[SUBJECT_ID_COL].nunique()
    train_patients = split_df.loc[split_df["split"] == "train", SUBJECT_ID_COL].nunique()
    test_patients = split_df.loc[split_df["split"] == "test", SUBJECT_ID_COL].nunique()
    train_rows = int((filtered_panel["split"] == "train").sum())
    test_rows = int((filtered_panel["split"] == "test").sum())

    print(f"[READ]  {config.master_panel_file} rows={len(panel):,} cols={len(panel.columns):,}")
    print(f"[KEEP]  retained covariate columns={len(kept_cov_cols):,}")
    print(f"[SPLIT] patients total={patient_total:,} train={train_patients:,} test={test_patients:,}")
    print(f"[SPLIT] rows train={train_rows:,} test={test_rows:,}")
    print(f"[WRITE] {config.filtered_panel_file} rows={len(filtered_panel):,} cols={len(filtered_panel.columns):,}")
    print(f"[WRITE] {config.train_test_split_file} rows={len(split_df):,} cols={len(split_df.columns):,}")


# =============================================================================
# Modeling panel and feature metadata
# =============================================================================

def validate_split(df: pd.DataFrame) -> None:
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    valid_splits = {"train", "test"}
    found_splits = set(df[SPLIT_COL].dropna().unique())
    invalid_splits = sorted(found_splits - valid_splits)
    if invalid_splits:
        raise ValueError(f"Unexpected split values in {SPLIT_COL}: {invalid_splits}")


def base_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [
        c for c in df.columns
        if c.startswith("itemid_") or c.startswith("sex_") or c.startswith("ethnicity_")
    ]
    cols.append("age")

    seen = set()
    out = []
    for col in cols:
        if col in df.columns and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def coerce_numeric(df: pd.DataFrame, cols: list[str], fill_missing_with_zero: bool) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = df[col].replace({
                "TRUE": 1,
                "FALSE": 0,
                "True": 1,
                "False": 0,
                "true": 1,
                "false": 0,
            })
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if fill_missing_with_zero:
            df[col] = df[col].fillna(0).astype(int)


def json_ready(obj):
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    return obj


def detect_covariate_itemids(columns: list[str]) -> pd.DataFrame:
    pattern = re.compile(r"^itemid_(\d+)__([a-z0-9_]+)$", flags=re.IGNORECASE)
    itemids = set()
    for col in columns:
        match = pattern.match(str(col))
        if match:
            itemids.add(int(match.group(1)))
    return pd.DataFrame({"itemid": sorted(itemids)})


def add_transition_columns(df: pd.DataFrame) -> None:
    required_cols = [STATE_COL, ACTION_COL, Y_CAUTI, Y_REINS, Y_DEATH, Y_ICU_EXIT]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required transition-label columns: {missing_cols}")

    for col in [ACTION_COL, Y_CAUTI, Y_REINS, Y_DEATH, Y_ICU_EXIT]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df[TRANSITION_LABEL_COL] = "no_event_continue"
    df.loc[df[ACTION_COL] == 1, TRANSITION_LABEL_COL] = "removal"
    df.loc[df[Y_REINS] == 1, TRANSITION_LABEL_COL] = "reinsertion"
    df.loc[df[Y_CAUTI] == 1, TRANSITION_LABEL_COL] = "cauti"
    df.loc[(df[Y_ICU_EXIT] == 1) & (df[Y_DEATH] == 0), TRANSITION_LABEL_COL] = "icu_exit_alive"
    df.loc[df[Y_DEATH] == 1, TRANSITION_LABEL_COL] = "death"

    df[OBSERVED_ACTION_COL] = "keep"
    df.loc[(df[STATE_COL] == "in") & (df[ACTION_COL] == 1), OBSERVED_ACTION_COL] = "remove"
    df.loc[df[STATE_COL] == "out", OBSERVED_ACTION_COL] = "out"
    df[ACTION_REMOVE_COL] = (df[OBSERVED_ACTION_COL] == "remove").astype(int)


def build_modeling_panel(config: PanelBuildConfig) -> None:
    config.data_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(config.filtered_panel_file, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()

    validate_split(df)
    df["state_is_out"] = (df[STATE_COL] == "out").astype(int)
    add_transition_columns(df)

    base_cols_for_features = base_feature_cols(df)
    coerce_numeric(
        df,
        base_cols_for_features + [TIME_COL, PERIODS_COL, "state_is_out", ACTION_REMOVE_COL],
        fill_missing_with_zero=False,
    )

    target_flag_cols = [ACTION_COL, Y_CAUTI, Y_REINS, Y_DEATH, Y_ICU_EXIT, LAST_PERIOD_COL]
    coerce_numeric(df, target_flag_cols, fill_missing_with_zero=True)

    feature_cols = list(base_cols_for_features)
    x_cols_remove = [TIME_COL, PERIODS_COL, *feature_cols]
    x_cols_cauti = [TIME_COL, PERIODS_COL, "state_is_out", *feature_cols]
    x_cols_reins = [PERIODS_COL, *feature_cols]
    x_cols_transition = [TIME_COL, PERIODS_COL, "state_is_out", ACTION_REMOVE_COL, *feature_cols]

    required_feature_cols = sorted(set(x_cols_remove + x_cols_cauti + x_cols_reins + x_cols_transition))
    missing_required = [c for c in required_feature_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required model feature columns after preprocessing: {missing_required}")

    df.to_csv(config.modeling_panel_file, index=False)

    period_hours = int(pd.to_numeric(df["interval_hours"], errors="coerce").dropna().mode().iloc[0])

    covariate_dict = detect_covariate_itemids(df.columns.tolist())
    itemid_to_label = load_item_labels(config.d_items_path)
    covariate_dict["label"] = covariate_dict["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    covariate_dict.sort_values(["label", "itemid"]).to_csv(config.covariate_dictionary_file, index=False)

    spec = {
        "id_col": ID_COL,
        "time_col": TIME_COL,
        "state_col": STATE_COL,
        "periods_col": PERIODS_COL,
        "split_col": SPLIT_COL,
        "action_col": ACTION_COL,
        "y_cauti": Y_CAUTI,
        "y_reins": Y_REINS,
        "y_death": Y_DEATH,
        "y_icu_exit": Y_ICU_EXIT,
        "transition_label_col": TRANSITION_LABEL_COL,
        "observed_action_col": OBSERVED_ACTION_COL,
        "action_remove_col": ACTION_REMOVE_COL,
        "last_period_col": LAST_PERIOD_COL,
        "end_reason_col": END_REASON_COL,
        "period_hours": period_hours,
        "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
        "base_feature_cols": base_cols_for_features,
        "features": feature_cols,
        "x_cols_remove": x_cols_remove,
        "x_cols_cauti": x_cols_cauti,
        "x_cols_reins": x_cols_reins,
        "x_cols_transition": x_cols_transition,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
    }
    config.feature_spec_file.write_text(json.dumps(json_ready(spec), indent=2), encoding="utf-8")

    print(f"[SAVE] modeling panel: {config.modeling_panel_file}")
    print(f"[SAVE] feature spec: {config.feature_spec_file}")
    print(f"[SAVE] covariate dictionary: {config.covariate_dictionary_file}")
    print(f"Rows: {len(df)}")
    print(f"Base features: {len(base_cols_for_features)}")
    print(f"Total features: {len(feature_cols)}")


# =============================================================================
# Entrypoint
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the full CAUTI modeling data panel.")
    parser.add_argument("--mimic-dir", type=Path, default=DEFAULT_MIMIC_DIR, help="Path to the MIMIC-IV root directory.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Pipeline data directory.")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "config", help="Pipeline config directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PanelBuildConfig(
        repo_root=REPO_ROOT,
        mimic_dir=args.mimic_dir,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
    )

    print("[CONFIG] repo root:", config.repo_root)
    print("[CONFIG] mimic dir:", config.mimic_dir)
    print("[CONFIG] data dir:", config.data_dir)
    print("[CONFIG] config dir:", config.config_dir)

    start_time = print_section("Define catheter episode cohort and base panel")
    create_episode_cohort_and_base_panel(config)
    print_section_done(start_time)

    start_time = print_section("Extract raw chart-event covariates")
    extract_raw_chart_covariates(config)
    print_section_done(start_time)

    start_time = print_section("Preprocess raw chart-event covariates")
    preprocess_raw_chart_covariates(config)
    print_section_done(start_time)

    start_time = print_section("Filter chart covariates to the item allowlist")
    filter_preprocessed_chart_covariates(config)
    print_section_done(start_time)

    start_time = print_section("Validate raw kept chart covariates")
    validate_raw_chart_covariates(config)
    print_section_done(start_time)

    start_time = print_section("Clean chart covariates")
    clean_chart_covariates(config)
    print_section_done(start_time)

    start_time = print_section("Validate cleaned chart covariates")
    validate_cleaned_chart_covariates(config)
    print_section_done(start_time)

    start_time = print_section("Aggregate cleaned covariates onto the base panel")
    build_master_panel(config)
    print_section_done(start_time)

    start_time = print_section("Select retained covariates")
    select_retained_covariates(config)
    print_section_done(start_time)

    start_time = print_section("Build filtered panel and train/test split")
    build_filtered_panel(config)
    print_section_done(start_time)

    start_time = print_section("Build modeling panel and feature metadata")
    build_modeling_panel(config)
    print_section_done(start_time)

    print()
    print("Data panel creation completed.")


if __name__ == "__main__":
    main()
