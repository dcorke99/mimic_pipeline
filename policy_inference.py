"""Estimator-independent patient-bootstrap reporting helpers."""

import numpy as np
import pandas as pd

import policy_eval_common as pec

CURRENT_PRACTICE = "current_practice"


def paired_differences(values, entries):
    baseline = {
        (row.estimator, row.outcome): i
        for i, row in enumerate(entries.itertuples(index=False))
        if row.policy_name == CURRENT_PRACTICE
    }
    indices = [baseline[(row.estimator, row.outcome)] for row in entries.itertuples(index=False)]
    return np.asarray(values) - np.asarray(values)[..., indices]


def bootstrap_replicate_table(entries, values, diagnostics):
    columns = [
        f"{row.policy_name}__{row.outcome}__{estimate_type}"
        for row in entries.itertuples(index=False)
        for estimate_type in ("policy_value", "difference_vs_current_practice")
    ]
    estimates = np.stack([values, paired_differences(values, entries)], axis=2)
    table = pd.concat([
        diagnostics.reset_index(drop=True),
        pd.DataFrame(estimates.reshape(len(values), len(columns)), columns=columns),
    ], axis=1)
    table["catheter_exposure_measure"] = pec.EXPOSURE_MEASURE
    return table


def interval_table(entries, point, bootstrap, n_patients):
    table = entries.copy()
    table["point_estimate"] = point
    bounds, valid_counts = [], []
    for column in bootstrap.T:
        finite = column[np.isfinite(column)]
        valid_counts.append(len(finite))
        bounds.append(np.percentile(finite, [2.5, 97.5], method="linear") if len(finite) else [np.nan, np.nan])
    table[["ci_lower", "ci_upper"]] = np.asarray(bounds)
    table["n_bootstrap"] = len(bootstrap)
    table["n_valid_bootstrap"] = valid_counts
    table["n_undefined_bootstrap"] = len(bootstrap) - np.asarray(valid_counts)
    table["n_unique_patients"] = n_patients
    table["confidence_level"] = 0.95
    table["unit"] = np.where(table.outcome.eq("catheter_exposure_days"), "days", "probability")
    table["catheter_exposure_measure"] = np.where(
        table.outcome.eq("catheter_exposure_days"), pec.EXPOSURE_MEASURE, "",
    )
    return table
