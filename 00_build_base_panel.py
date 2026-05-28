# Identify the catheter episodes that define the downstream master-panel cohort.

from pathlib import Path

import numpy as np
import pandas as pd

MIMIC_DIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\Data\MIMIC-IV\mimic-iv-3.1")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
OUTFILE = OUTDIR / "required_catheter_episodes.csv"
BASE_PANEL_OUTFILE = OUTDIR / "base_panel.csv"
# Minimum retained episode duration.
MIN_EPISODE_DURATION = pd.Timedelta(hours=24)
# Length of one panel period.
PERIOD_DURATION = pd.Timedelta(hours=24)
# Foley catheter procedure itemid.
FOLEY_ITEMID = 229351
LOOKBACK_DURATION = pd.Timedelta(hours=24)
POST_REMOVE_RISK_PERIODS = 2


def map_ethnicity_group(value):
    # Collapse detailed race strings into a small set of analysis groups.
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
    # Merge overlapping Foley rows within each stay into one continuous episode.
    episodes = []

    for stay_id, stay_events in df.groupby("stay_id"):
        # Sort procedure rows by insertion time within the stay.
        stay_events = stay_events.sort_values("inserted")

        current_start = None
        current_end = None

        for event in stay_events.itertuples():
            if current_start is None:
                # Initialise the first collapsed episode.
                current_start = event.inserted
                current_end = event.removed
                continue

            if event.inserted <= current_end:
                # If the next row starts before the current episode ends, both rows are
                # part of the same episode, so keep the earliest start and extend the end.
                if pd.isna(current_end):
                    current_end = event.removed
                elif pd.notna(event.removed) and event.removed > current_end:
                    current_end = event.removed
            else:
                # If the next row starts after the current episode ends, the overlap is
                # broken, so save the finished episode and start a new one.
                episodes.append((stay_id, current_start, current_end))
                current_start = event.inserted
                current_end = event.removed

        if current_start is not None:
            # Append the final episode for the stay.
            episodes.append((stay_id, current_start, current_end))

    return pd.DataFrame(episodes, columns=["stay_id", "inserted", "removed"])


def build_required_catheter_episodes(mimic_dir: Path) -> pd.DataFrame:
    # Read ICU stay rows.
    icu = pd.read_csv(
        mimic_dir / "icu/icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )

    # Parse ICU timestamps and drop incomplete stays.
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["stay_id", "intime", "outtime"]).copy()

    # Read patient rows and calculate age at ICU admission.
    patients = pd.read_csv(
        mimic_dir / "hosp/patients.csv",
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

    # Read admission rows and collapse race values to analysis groups.
    admissions = pd.read_csv(
        mimic_dir / "hosp/admissions.csv",
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

    # Read procedure event rows.
    procedure_events = pd.read_csv(
        mimic_dir / "icu/procedureevents.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"],
    )

    # Keep Foley rows and parse procedure timestamps.
    procedure_events = procedure_events[procedure_events["itemid"] == FOLEY_ITEMID].copy()
    procedure_events["starttime"] = pd.to_datetime(procedure_events["starttime"], errors="coerce")
    procedure_events["endtime"] = pd.to_datetime(procedure_events["endtime"], errors="coerce")

    # Join ICU stay columns onto procedure rows.
    procedure_events = procedure_events.merge(
        icu[["stay_id", "subject_id", "hadm_id", "intime", "outtime"]],
        on="stay_id",
        how="left",
    )

    # Rename ICU time columns and create episode timestamps.
    procedure_events = procedure_events.rename(columns={"intime": "ICU_in", "outtime": "ICU_out"})
    procedure_events["inserted"] = procedure_events["starttime"]
    procedure_events["removed"] = procedure_events["endtime"]
    # Replace missing removal times with ICU discharge.
    procedure_events.loc[procedure_events["removed"].isna(), "removed"] = procedure_events["ICU_out"]
    # Drop rows without complete episode timing.
    procedure_events = procedure_events.dropna(subset=["inserted", "removed", "ICU_in", "ICU_out"]).copy()
    # Sort rows by stay and insertion time.
    procedure_events = procedure_events.sort_values(["stay_id", "inserted"])

    # Collapse overlapping procedure rows into episodes.
    collapsed = merge_overlapping_foley_events(procedure_events)

    # Join identifiers, demographics, and ICU times onto episode rows.
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

    # Mark the next catheter insertion within each stay.
    catheterised = catheterised.sort_values(["stay_id", "inserted"]).reset_index(drop=True)
    catheterised["reinsertion_time"] = catheterised.groupby("stay_id")["inserted"].shift(-1)

    # Filter to episodes lasting at least the minimum retained duration.
    episode_duration = catheterised["removed"] - catheterised["inserted"]
    catheterised = catheterised[episode_duration >= MIN_EPISODE_DURATION].copy()

    # Read microbiology rows used for CAUTI episode matching.
    micro = pd.read_csv(
        mimic_dir / "hosp/microbiologyevents.csv",
        usecols=["subject_id", "hadm_id", "charttime", "spec_type_desc", "org_name"],
    )
    micro["charttime"] = pd.to_datetime(micro["charttime"], errors="coerce")
    micro = micro[
        micro["spec_type_desc"].str.contains("urine", case=False, na=False) &
        micro["org_name"].notna()
    ].copy()

    # Match urine microbiology rows to catheter episodes within the CAUTI time window.
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

    # Join the first matching CAUTI time onto each episode row.
    catheterised = catheterised.merge(
        micro_matched,
        on=["stay_id", "inserted"],
        how="left",
    )

    # Return rows sorted by stay and insertion time.
    return catheterised.sort_values(["stay_id", "inserted"]).reset_index(drop=True)


def make_state_windows(state_start, state_end):
    # Split one state interval into consecutive windows up to PERIOD_DURATION long.
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
    # Expand each catheter episode into in-state and out-state panel periods.
    rows = []

    for episode in catheterised.itertuples():
        in_windows = make_state_windows(episode.inserted, episode.removed)

        for state_idx, period_start, period_end, interval_hours in in_windows:
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
        out_windows = make_state_windows(episode.removed, out_state_end)

        for state_idx, period_start, period_end, interval_hours in out_windows:
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
    panel.loc[
        (panel["icu_end_in_period"] == 1) & (panel["death_in_period"] == 0),
        "next_state",
    ] = "ICU_EXIT_ALIVE"
    panel.loc[panel["death_in_period"] == 1, "next_state"] = "DEATH"
    panel.loc[panel["removed_in_period"] == 1, "next_state"] = "REMOVAL"
    panel.loc[panel["reinsertion_in_period"] == 1, "next_state"] = "REINSERTION"
    panel.loc[panel["cauti_in_period"] == 1, "next_state"] = "CAUTI"

    panel["at_risk_cauti"] = (
        (panel["catheter_state"] == "in") |
        (
            (panel["catheter_state"] == "out") &
            (panel["periods_in_state"] <= POST_REMOVE_RISK_PERIODS)
        )
    ).astype(int)

    panel["at_risk_reinsertion"] = (
        panel["catheter_state"] == "out"
    ).astype(int)

    episode_keys = ["stay_id", "inserted"]
    panel["is_last_period_of_episode"] = 0
    last_row_index = panel.groupby(episode_keys)["period_end"].idxmax()
    panel.loc[last_row_index, "is_last_period_of_episode"] = 1

    panel["episode_end_reason"] = pd.NA
    panel.loc[
        (panel["is_last_period_of_episode"] == 1) & (panel["reinsertion_in_period"] == 1),
        "episode_end_reason"
    ] = "reinsertion"
    panel.loc[
        (panel["is_last_period_of_episode"] == 1) &
        (panel["episode_end_reason"].isna()) &
        (panel["icu_end_in_period"] == 1),
        "episode_end_reason"
    ] = "icu_end"

    panel["sex_M"] = (panel["gender"] == "M").astype(int)
    panel["sex_missing"] = panel["gender"].isna().astype(int)
    eth_dummies = pd.get_dummies(panel["ethnicity_group"], prefix="ethnicity")
    panel = pd.concat([panel, eth_dummies], axis=1)

    panel["cov_start"] = panel["period_start"] - LOOKBACK_DURATION
    panel["cov_end"] = panel["period_start"]
    panel["cov_start"] = panel[["cov_start", "intime"]].max(axis=1)
    panel["row_id"] = np.arange(1, len(panel) + 1)

    # Keep ethnicity dummy columns grouped together near the demographics.
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
    panel = panel[[c for c in ordered_cols if c in non_ethnicity_cols or c in ethnicity_cols]]
    return panel


def main() -> None:
    # Create the output directory.
    OUTDIR.mkdir(exist_ok=True, parents=True)

    # Build the episode cohort and the base panel.
    episodes = build_required_catheter_episodes(MIMIC_DIR)
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

    # Save the episode-level and row-level outputs.
    episode_export.to_csv(OUTFILE, index=False)
    base_panel.to_csv(BASE_PANEL_OUTFILE, index=False)

    # Print the run summary.
    print("[SAVE]", OUTFILE)
    print("[SAVE]", BASE_PANEL_OUTFILE)
    print("Episodes:", len(episodes))
    print("Stays:", episodes["stay_id"].nunique())
    print("Base panel rows:", len(base_panel))
    

if __name__ == "__main__":
    main()
