"""
Extract raw chart-event covariates for the CAUTI analysis cohort.

Creates a raw long-format chart-events file for the ICU stays that contribute
catheter episodes to the downstream panel.
"""

from pathlib import Path
import time
import pandas as pd

# Configuration constants

MIMIC_DIR = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1")
PATIENT_DAY_SECONDS = 86400
FOLEY_ITEMID = 229351
CHUNK_ROWS_CHARTEVENTS = 1_000_000


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

    episode_duration_seconds = (catheterised["removed"] - catheterised["inserted"]).dt.total_seconds()
    catheterised = catheterised[episode_duration_seconds >= PATIENT_DAY_SECONDS]

    print("[Catheter episodes]", len(catheterised))

    stay_ids = set(catheterised["stay_id"].dropna().astype(int).unique())

    print("[EHR] Extracting raw chartevents covariates...")

    usecols_ce = ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "storetime", "valuenum", "value", "valueuom"]

    outdir = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
    outdir.mkdir(exist_ok=True)

    outfile = outdir / "raw_chart_covariates.csv"
    if outfile.exists():
        outfile.unlink()

    p_chartevents = MIMIC_DIR / "icu/chartevents.csv"
    kept_rows_total = 0
    chunk_idx = 0
    t0 = time.time()
    first_write = True

    for chunk in pd.read_csv(
        p_chartevents,
        usecols=usecols_ce,
        chunksize=CHUNK_ROWS_CHARTEVENTS,
        low_memory=False,
    ):
        chunk_idx += 1
        t_chunk0 = time.time()

        chunk_filtered = chunk[chunk["stay_id"].isin(stay_ids)].copy()
        kept = len(chunk_filtered)
        kept_rows_total += kept

        if kept > 0:
            chunk_filtered["charttime"] = pd.to_datetime(chunk_filtered["charttime"], errors="coerce")
            if "storetime" in chunk_filtered.columns:
                chunk_filtered["storetime"] = pd.to_datetime(chunk_filtered["storetime"], errors="coerce")

            chunk_filtered.to_csv(
                outfile,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False

        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} keep={kept:,} "
            f"cum_keep={kept_rows_total:,} dt={time.time()-t_chunk0:.1f}s"
        )

    print(f"[EHR] Raw chart covariates extracted. Total time: {time.time()-t0:.1f}s")
    print("[SAVE]", outfile)
    print("Rows:", kept_rows_total)
    print("[DONE]")


if __name__ == "__main__":
    main()
