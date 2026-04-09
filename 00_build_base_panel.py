# Identify the catheter episodes that define the downstream master-panel cohort.

from pathlib import Path

import numpy as np
import pandas as pd

MIMIC_DIR = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1")
OUTDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
OUTFILE = OUTDIR / "required_catheter_episodes.csv"
BASE_PANEL_OUTFILE = OUTDIR / "base_panel.csv"
# Minimum retained episode duration in seconds.
PATIENT_DAY_SECONDS = 86400
# Foley catheter procedure itemid.
FOLEY_ITEMID = 229351
LOOKBACK_HOURS = 24


def collapse_ethnicity(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).upper()
    if "WHITE" in s:
        return "White"
    if "BLACK" in s:
        return "Black"
    if "ASIAN" in s:
        return "Asian"
    if "HISPANIC" in s or "LATIN" in s:
        return "Hispanic"
    if "DECLINED" in s or "UNKNOWN" in s or "UNABLE" in s:
        return "Unknown"
    return "Other"


def collapse_foley_events(df: pd.DataFrame) -> pd.DataFrame:
    # Collapse overlapping Foley procedure rows within each ICU stay.
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
                # Extend the current episode end when rows overlap.
                if pd.isna(current_end):
                    current_end = event.removed
                elif pd.notna(event.removed) and event.removed > current_end:
                    current_end = event.removed
            else:
                # Append the completed episode and start a new one.
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
        usecols=["subject_id", "hadm_id", "race"],
        low_memory=False,
    ).rename(columns={"race": "ethnicity"})

    icu = icu.merge(
        admissions[["subject_id", "hadm_id", "ethnicity"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    icu["ethnicity_group"] = icu["ethnicity"].apply(collapse_ethnicity)

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
    collapsed = collapse_foley_events(procedure_events)

    # Join identifiers, demographics, and ICU times onto episode rows.
    catheterised = collapsed.merge(
        icu[["stay_id", "subject_id", "hadm_id", "intime", "outtime", "gender", "age", "ethnicity_group"]],
        on="stay_id",
        how="inner",
    ).rename(columns={"intime": "ICU_in", "outtime": "ICU_out"})

    # Initialise the reinsertion-time column.
    catheterised["reinsertion_time"] = pd.NaT

    # Assign the next insertion time within each stay.
    for stay_id, stay_episodes in catheterised.groupby("stay_id"):
        # Collect row indices and insertion timestamps.
        episode_indices = stay_episodes.index.tolist()
        insertion_times = stay_episodes["inserted"].tolist()

        # Write the next insertion time back to the current row.
        for i in range(len(episode_indices) - 1):
            catheterised.loc[episode_indices[i], "reinsertion_time"] = insertion_times[i + 1]

    # Filter to episodes lasting at least 24 hours.
    episode_duration_seconds = (catheterised["removed"] - catheterised["inserted"]).dt.total_seconds()
    catheterised = catheterised[episode_duration_seconds >= PATIENT_DAY_SECONDS].copy()

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
    if pd.isna(state_start) or pd.isna(state_end) or state_end <= state_start:
        return []

    rows = []
    window_start = state_start
    state_idx = 0

    while window_start < state_end:
        window_end = min(window_start + pd.Timedelta(hours=24), state_end)
        interval_hours = round((window_end - window_start).total_seconds() / 3600.0, 2)
        rows.append((state_idx, window_start, window_end, interval_hours))
        window_start = window_end
        state_idx += 1

    return rows


def build_base_panel(catheterised: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for episode in catheterised.itertuples():
        in_windows = make_state_windows(episode.inserted, episode.removed)

        for state_idx, day_start, day_end, interval_hours in in_windows:
            rows.append({
                "subject_id": episode.subject_id,
                "hadm_id": episode.hadm_id,
                "stay_id": episode.stay_id,
                "inserted": episode.inserted,
                "removed": episode.removed,
                "reinsertion_time": episode.reinsertion_time,
                "catheter_state": "in",
                "state_index": state_idx,
                "day_start": day_start,
                "day_end": day_end,
                "interval_hours": interval_hours,
                "cauti_time": episode.cauti_time,
                "ICU_out": episode.ICU_out,
                "intime": episode.ICU_in,
                "gender": episode.gender,
                "age": episode.age,
                "ethnicity_group": episode.ethnicity_group,
            })

        out_state_end = episode.reinsertion_time if pd.notna(episode.reinsertion_time) else episode.ICU_out
        out_windows = make_state_windows(episode.removed, out_state_end)

        for state_idx, day_start, day_end, interval_hours in out_windows:
            rows.append({
                "subject_id": episode.subject_id,
                "hadm_id": episode.hadm_id,
                "stay_id": episode.stay_id,
                "inserted": episode.inserted,
                "removed": episode.removed,
                "reinsertion_time": episode.reinsertion_time,
                "catheter_state": "out",
                "state_index": state_idx,
                "day_start": day_start,
                "day_end": day_end,
                "interval_hours": interval_hours,
                "cauti_time": episode.cauti_time,
                "ICU_out": episode.ICU_out,
                "intime": episode.ICU_in,
                "gender": episode.gender,
                "age": episode.age,
                "ethnicity_group": episode.ethnicity_group,
            })

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(
        ["stay_id", "inserted", "day_start", "day_end", "catheter_state"]
    ).reset_index(drop=True)

    panel["episode_index"] = panel.groupby(["stay_id", "inserted"]).cumcount()
    panel["days_in_state"] = panel["state_index"] + 1
    panel = panel.drop(columns=["state_index"])

    panel["removed_today"] = (
        (panel["catheter_state"] == "in") &
        (panel["removed"] > panel["day_start"]) &
        (panel["removed"] <= panel["day_end"])
    ).astype(int)

    panel["reinsertion_today"] = (
        (panel["catheter_state"] == "out") &
        panel["reinsertion_time"].notna() &
        (panel["reinsertion_time"] > panel["day_start"]) &
        (panel["reinsertion_time"] <= panel["day_end"])
    ).astype(int)

    panel["icu_end_today"] = (
        panel["ICU_out"].notna() &
        (panel["ICU_out"] > panel["day_start"]) &
        (panel["ICU_out"] <= panel["day_end"])
    ).astype(int)

    panel["cauti_today"] = (
        panel["cauti_time"].notna() &
        (panel["cauti_time"] > panel["day_start"]) &
        (panel["cauti_time"] <= panel["day_end"])
    ).astype(int)

    panel["at_risk_in"] = (panel["catheter_state"] == "in").astype(int)
    panel["at_risk_out"] = (panel["catheter_state"] == "out").astype(int)

    episode_keys = ["stay_id", "inserted"]
    panel["is_last_day_of_episode"] = 0
    last_row_index = panel.groupby(episode_keys)["day_end"].idxmax()
    panel.loc[last_row_index, "is_last_day_of_episode"] = 1

    panel["episode_end_reason"] = pd.NA
    panel.loc[
        (panel["is_last_day_of_episode"] == 1) & (panel["reinsertion_today"] == 1),
        "episode_end_reason"
    ] = "reinsertion"
    panel.loc[
        (panel["is_last_day_of_episode"] == 1) &
        (panel["episode_end_reason"].isna()) &
        (panel["icu_end_today"] == 1),
        "episode_end_reason"
    ] = "icu_end"

    panel["sex_M"] = (panel["gender"] == "M").astype(int)
    panel["sex_missing"] = panel["gender"].isna().astype(int)
    eth_dummies = pd.get_dummies(panel["ethnicity_group"], prefix="ethnicity", dummy_na=True)
    panel = pd.concat([panel, eth_dummies], axis=1)

    panel["cov_start"] = panel["day_start"] - pd.Timedelta(hours=LOOKBACK_HOURS)
    panel["cov_end"] = panel["day_start"]
    panel["cov_start"] = panel[["cov_start", "intime"]].max(axis=1)
    panel["row_id"] = np.arange(1, len(panel) + 1)

    panel = panel.drop(columns=["cauti_time", "gender", "ethnicity_group", "intime", "ICU_out"])
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
        "day_start",
        "day_end",
        "interval_hours",
        "days_in_state",
        "removed_today",
        "reinsertion_today",
        "cauti_today",
        "icu_end_today",
        "is_last_day_of_episode",
        "episode_end_reason",
        "at_risk_in",
        "at_risk_out",
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
