"""
Build Causal Temporal Panel for CAUTI Analysis.

Creates a patient-interval panel from MIMIC-IV with catheter episodes, state-specific
intervals, CAUTI events, and time-varying covariates for downstream causal modelling.

Key timing variables
--------------------
- episode_index: absolute interval index within the catheter episode; does NOT reset
  when the state changes from IN to OUT.
- days_in_state: interval count within the current state block; resets separately
  within IN and OUT.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

# Configuration constants

MIMIC_DIR = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1")
CLEANED_CHART_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\cleaned_chart_covariates.csv")
PATIENT_DAY_SECONDS = 86400
FOLEY_ITEMID = 229351
CHUNK_ROWS_CHARTEVENTS = 1_000_000
LOOKBACK_HOURS = 24


def collapse_foley_events(df):
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

    out = pd.DataFrame(episodes, columns=["stay_id", "inserted", "removed"])
    return out


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


def main():
    print("[CONFIG]", MIMIC_DIR)

    icu = pd.read_csv(
        MIMIC_DIR / "icu/icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
    )

    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["stay_id", "intime", "outtime"])

    print("[ICU stays]", len(icu))

    patients = pd.read_csv(
        MIMIC_DIR / "hosp/patients.csv",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        low_memory=False
    )
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")
    patients["anchor_year"] = pd.to_numeric(patients["anchor_year"], errors="coerce")

    icu["icu_year"] = icu["intime"].dt.year
    icu = icu.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left"
    )
    icu["age"] = icu["anchor_age"] + (icu["icu_year"] - icu["anchor_year"])
    icu = icu.drop(columns=["icu_year", "anchor_age", "anchor_year"])

    admissions = pd.read_csv(
        MIMIC_DIR / "hosp/admissions.csv",
        usecols=["subject_id", "hadm_id", "race"],
        low_memory=False
    ).rename(columns={"race": "ethnicity"})

    icu = icu.merge(
        admissions[["subject_id", "hadm_id", "ethnicity"]],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    icu["ethnicity_group"] = icu["ethnicity"].apply(collapse_ethnicity)

    demo = icu[
        ["stay_id", "gender", "age", "ethnicity_group", "intime"]
    ].drop_duplicates("stay_id")

    procedure_events = pd.read_csv(
        MIMIC_DIR / "icu/procedureevents.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"]
    )

    procedure_events = procedure_events[procedure_events["itemid"] == FOLEY_ITEMID]

    procedure_events["starttime"] = pd.to_datetime(procedure_events["starttime"], errors="coerce")
    procedure_events["endtime"] = pd.to_datetime(procedure_events["endtime"], errors="coerce")

    procedure_events = procedure_events.merge(
        icu[["stay_id", "subject_id", "hadm_id", "intime", "outtime"]],
        on="stay_id",
        how="left"
    )

    procedure_events = procedure_events.rename(columns={"intime": "ICU_in", "outtime": "ICU_out"})

    procedure_events["inserted"] = procedure_events["starttime"]
    procedure_events["removed"] = procedure_events["endtime"]
    procedure_events.loc[procedure_events["removed"].isna(), "removed"] = procedure_events["ICU_out"]

    procedure_events = procedure_events.dropna(subset=["inserted", "removed", "ICU_out"])
    procedure_events = procedure_events.sort_values(["stay_id", "inserted"])

    collapsed = collapse_foley_events(procedure_events)

    catheterised = collapsed.merge(
        icu[["stay_id", "subject_id", "hadm_id", "outtime"]],
        on="stay_id"
    )
    catheterised = catheterised.rename(columns={"outtime": "ICU_out"})

    catheterised["reinsertion_time"] = pd.NaT

    for stay_id, stay_episodes in catheterised.groupby("stay_id"):
        episode_indices = stay_episodes.index.tolist()
        insertion_times = stay_episodes["inserted"].tolist()

        for i in range(len(episode_indices) - 1):
            catheterised.loc[episode_indices[i], "reinsertion_time"] = insertion_times[i + 1]

    episode_duration_seconds = (catheterised["removed"] - catheterised["inserted"]).dt.total_seconds()
    catheterised = catheterised[episode_duration_seconds >= PATIENT_DAY_SECONDS]

    print("[Catheter episodes]", len(catheterised))

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
                "ICU_out": episode.ICU_out,
                "catheter_state": "in",
                "state_index": state_idx,
                "day_start": day_start,
                "day_end": day_end,
                "interval_hours": interval_hours
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
                "ICU_out": episode.ICU_out,
                "catheter_state": "out",
                "state_index": state_idx,
                "day_start": day_start,
                "day_end": day_end,
                "interval_hours": interval_hours
            })

    panel = pd.DataFrame(rows)

    # Sort in true chronological order within episode.
    panel = panel.sort_values(
        ["stay_id", "inserted", "day_start", "day_end", "catheter_state"]
    ).reset_index(drop=True)

    # Absolute interval index across the whole episode; does not reset at state change.
    panel["episode_index"] = (
        panel.groupby(["stay_id", "inserted"])
        .cumcount()
    )

    # State-local interval index converted to 1-based days_in_state.
    panel["days_in_state"] = panel["state_index"] + 1

    # No longer needed once days_in_state has been created.
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

    micro = pd.read_csv(
        MIMIC_DIR / "hosp/microbiologyevents.csv",
        usecols=["subject_id", "hadm_id", "charttime", "spec_type_desc", "org_name"]
    )

    micro["charttime"] = pd.to_datetime(micro["charttime"], errors="coerce")

    micro = micro[
        micro["spec_type_desc"].str.contains("urine", case=False, na=False) &
        micro["org_name"].notna()
    ]

    micro_matched = micro.merge(
        catheterised[["subject_id", "hadm_id", "stay_id", "inserted", "removed"]],
        on=["subject_id", "hadm_id"]
    )

    micro_matched = micro_matched[
        (micro_matched["charttime"] >= micro_matched["inserted"]) &
        (micro_matched["charttime"] <= micro_matched["removed"] + pd.Timedelta(hours=48))
    ]

    micro_matched = micro_matched.sort_values("charttime").drop_duplicates(["stay_id", "inserted"])

    cauti_events = micro_matched[["stay_id", "inserted", "charttime"]]

    panel["cauti_today"] = 0

    panel_with_cauti = panel.merge(cauti_events, on=["stay_id", "inserted"], how="left")

    cauti_mask = (
        panel_with_cauti["charttime"].notna() &
        (panel_with_cauti["charttime"] > panel_with_cauti["day_start"]) &
        (panel_with_cauti["charttime"] <= panel_with_cauti["day_end"])
    )

    panel_with_cauti.loc[cauti_mask, "cauti_today"] = 1
    panel = panel_with_cauti.drop(columns=["charttime"])

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

    panel = panel.merge(
        demo[["stay_id", "gender", "age", "ethnicity_group", "intime"]],
        on="stay_id",
        how="left"
    )

    # Use one explicit sex indicator plus a missingness flag, rather than two redundant dummies.
    panel["sex_M"] = (panel["gender"] == "M").astype(int)
    panel["sex_missing"] = panel["gender"].isna().astype(int)
    sex_cols = ["sex_M", "sex_missing"]

    eth_dummies = pd.get_dummies(panel["ethnicity_group"], prefix="ethnicity", dummy_na=True)

    panel["cov_start"] = panel["day_start"] - pd.Timedelta(hours=LOOKBACK_HOURS)
    panel["cov_end"] = panel["day_start"]
    panel["cov_start"] = panel[["cov_start", "intime"]].max(axis=1)

    panel["row_id"] = np.arange(1, len(panel) + 1)

    windows = panel[["row_id", "stay_id", "cov_start", "cov_end"]].copy()
    stay_ids = set(windows["stay_id"].dropna().astype(int).unique())

    print("[EHR] Aggregating chartevents covariates...")

    usecols_ce = ["stay_id", "itemid", "charttime", "valuenum"]
    partial_stats = []

    kept_rows_total = 0
    chunk_idx = 0
    t0 = time.time()

    for chunk in pd.read_csv(
        CLEANED_CHART_FILE,
        usecols=usecols_ce,
        chunksize=CHUNK_ROWS_CHARTEVENTS,
        low_memory=False,
    ):
        chunk_idx += 1
        t_chunk0 = time.time()

        chunk_filtered = chunk[chunk["stay_id"].isin(stay_ids)].copy()
        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} keep=0 dt={time.time()-t_chunk0:.1f}s")
            continue

        chunk_filtered["charttime"] = pd.to_datetime(chunk_filtered["charttime"], errors="coerce")
        chunk_filtered["valuenum"] = pd.to_numeric(chunk_filtered["valuenum"], errors="coerce")

        chunk_filtered = chunk_filtered.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"])
        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} numeric_keep=0 dt={time.time()-t_chunk0:.1f}s")
            continue

        merged = chunk_filtered.merge(windows, on="stay_id", how="inner")
        merged = merged[
            (merged["charttime"] >= merged["cov_start"]) &
            (merged["charttime"] < merged["cov_end"])
        ].copy()

        kept = len(merged)
        kept_rows_total += kept

        if kept > 0:
            item_summary = (
                merged.groupby(["row_id", "itemid"])["valuenum"]
                .agg(count="count", total="sum", min="min", max="max")
                .reset_index()
            )
            partial_stats.append(item_summary)

        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} stay_filtered={len(chunk_filtered):,} "
            f"matched={kept:,} cum_matched={kept_rows_total:,} dt={time.time()-t_chunk0:.1f}s"
        )

    if partial_stats:
        agg = pd.concat(partial_stats, ignore_index=True)
        agg = (
            agg.groupby(["row_id", "itemid"])
            .agg(
                count=("count", "sum"),
                total=("total", "sum"),
                min=("min", "min"),
                max=("max", "max")
            )
            .reset_index()
        )
        agg["mean"] = agg["total"] / agg["count"]

        wide = agg.pivot_table(
            index="row_id",
            columns="itemid",
            values=["mean", "min", "max"],
            aggfunc="first"
        )
        wide.columns = [f"itemid_{itemid}__{stat}" for stat, itemid in wide.columns]
        wide = wide.reset_index()

        panel = panel.merge(wide, on="row_id", how="left")
        print(f"[EHR] Covariates aggregated. Total time: {time.time()-t0:.1f}s")
    else:
        print(f"[EHR] No chartevents matched windows. Total time: {time.time()-t0:.1f}s")

    panel = pd.concat(
        [
            panel.drop(columns=["gender", "ethnicity_group", "intime", "cov_start", "cov_end", "ICU_out"]),
            eth_dummies,
        ],
        axis=1,
    )

    covariate_cols = sorted([c for c in panel.columns if c.startswith("itemid_")])

    panel = panel[[
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
        *sex_cols,
        *sorted(eth_dummies.columns),
        *covariate_cols
    ]]

    outdir = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
    outdir.mkdir(exist_ok=True)

    outfile = outdir / "master_panel.csv"
    panel.to_csv(outfile, index=False)

    print("[SAVE]", outfile)
    print("Removals:", panel["removed_today"].sum())
    print("Reinsertions:", panel["reinsertion_today"].sum())
    print("CAUTI:", panel["cauti_today"].sum())
    print("ICU end rows:", panel["icu_end_today"].sum())
    print("Last episode days:", panel["is_last_day_of_episode"].sum())
    print("Rows:", len(panel))
    print("[DONE]")


if __name__ == "__main__":
    main()
