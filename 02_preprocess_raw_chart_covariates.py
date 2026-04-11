"""
Preprocess raw chart-event covariates before downstream validation/cleaning.

This step converts Fahrenheit temperatures to Celsius, remaps those rows onto
the Celsius temperature itemid, and saves a separate preprocessed file.
"""

from pathlib import Path

import pandas as pd

DATADIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
INFILE = DATADIR / "raw_chart_covariates.csv"
OUTFILE = DATADIR / "preprocessed_raw_chart_covariates.csv"
SAMPLE_OUTFILE = DATADIR / "preprocessed_raw_chart_covariates__first_1000_rows.csv"
CHUNK_ROWS = 1_000_000
SAMPLE_ROWS = 1000
FAHRENHEIT_UNITS = {"F", "DEG F", "DEGREES F", "°F", "° F"}
CELSIUS_UNIT = "°C"
TEMP_F_ITEMID = 223761
TEMP_C_ITEMID = 223762


def fahrenheit_to_celsius(values: pd.Series) -> pd.Series:
    # Convert Fahrenheit temperatures to Celsius.
    return (values - 32.0) * (5.0 / 9.0)


def main() -> None:
    DATADIR.mkdir(exist_ok=True, parents=True)

    # Replace any previous outputs so this step always writes a fresh file.
    if OUTFILE.exists():
        OUTFILE.unlink()
    if SAMPLE_OUTFILE.exists():
        SAMPLE_OUTFILE.unlink()

    first_write = True
    chunk_idx = 0
    converted_rows_total = 0
    sample_rows_written = 0

    # Stream the raw chart file so temperature conversion scales to large extracts.
    for chunk in pd.read_csv(INFILE, chunksize=CHUNK_ROWS, low_memory=False):
        chunk_idx += 1

        # Standardise units and itemids before identifying Fahrenheit rows.
        unit_clean = chunk["valueuom"].fillna("").astype(str).str.strip().str.upper()
        itemids = pd.to_numeric(chunk["itemid"], errors="coerce")
        fahrenheit_mask = unit_clean.isin(FAHRENHEIT_UNITS) | itemids.eq(TEMP_F_ITEMID)

        if fahrenheit_mask.any():
            # Convert numeric Fahrenheit values to Celsius.
            chunk.loc[fahrenheit_mask, "valuenum"] = pd.to_numeric(
                chunk.loc[fahrenheit_mask, "valuenum"],
                errors="coerce",
            )
            chunk.loc[fahrenheit_mask, "valuenum"] = fahrenheit_to_celsius(
                chunk.loc[fahrenheit_mask, "valuenum"]
            )

            # Convert string temperature values to Celsius when they are numeric.
            numeric_value = pd.to_numeric(chunk.loc[fahrenheit_mask, "value"], errors="coerce")
            numeric_mask = numeric_value.notna()
            if numeric_mask.any():
                converted_value = fahrenheit_to_celsius(numeric_value.loc[numeric_mask]).round(3)
                chunk.loc[numeric_value.loc[numeric_mask].index, "value"] = converted_value.astype(str)

            # Reassign Fahrenheit rows onto the Celsius itemid and unit.
            chunk.loc[fahrenheit_mask, "itemid"] = TEMP_C_ITEMID
            chunk.loc[fahrenheit_mask, "valueuom"] = CELSIUS_UNIT
            converted_rows_total += int(fahrenheit_mask.sum())

        # Save the full converted chunk.
        chunk.to_csv(
            OUTFILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        # Save the first SAMPLE_ROWS rows as a lightweight inspection file.
        if sample_rows_written < SAMPLE_ROWS:
            sample_chunk = chunk.head(SAMPLE_ROWS - sample_rows_written).copy()
            sample_chunk.to_csv(
                SAMPLE_OUTFILE,
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

    print("[SAVE]", OUTFILE)
    print("[SAVE SAMPLE]", SAMPLE_OUTFILE)
    print("Converted rows:", converted_rows_total)
    print("[DONE]")


if __name__ == "__main__":
    main()
