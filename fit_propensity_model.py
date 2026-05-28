import json

import numpy as np
import pandas as pd

from model_utils import (
    ACTION_COL,
    AT_RISK_CAUTI,
    AT_RISK_REINS,
    CALIBRATION_BINS,
    COVARIATE_DICT_FILE,
    FEATURE_SPEC_FILE,
    ID_COL,
    INFILE,
    MODEL_DIR,
    MODEL_TYPE,
    OUTDIR,
    PERIODS_COL,
    POST_REMOVE_RISK_PERIODS,
    SAVE_SHAP,
    SEED,
    SHAP_SAMPLE_N,
    SPLIT_COL,
    STATE_COL,
    TIME_COL,
    TOP_FEATURES_TO_SAVE,
    Y_CAUTI,
    Y_REINS,
    add_feature_descriptions,
    add_prior_cauti_count,
    calibration_table,
    dump_joblib,
    feature_importance_series,
    fit_model,
    insert_score_columns_before_age,
    json_ready,
    load_feature_spec,
    load_panel,
    metrics_by_period,
    predict_proba,
    restore_original_order,
    save_df,
    save_shap_plots,
    scalar_binary_metrics,
    shap_importance_series,
    top_series_df,
)


PROPENSITY_SCORE_COL = "p_remove_obs"


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    feature_spec = load_feature_spec()
    covariate_dict = pd.read_csv(COVARIATE_DICT_FILE)
    df = add_prior_cauti_count(load_panel())

    feature_list = feature_spec["features"]
    remove_feature_cols = feature_spec["x_cols_remove"]

    print("Propensity model uses precomputed explicit features only.")
    print("Hidden imputer indicator columns created inside propensity fit: 0")
    print(f"Loaded features: {len(feature_list)}")

    df[PROPENSITY_SCORE_COL] = np.zeros(len(df), dtype=float)

    df_in = df[
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    train_features = df_in.loc[df_in[SPLIT_COL] == "train", remove_feature_cols]
    train_target = df_in.loc[df_in[SPLIT_COL] == "train", ACTION_COL].astype(int)
    test_features = df_in.loc[df_in[SPLIT_COL] == "test", remove_feature_cols]

    print(f"Fitting removal propensity model with {MODEL_TYPE}...", flush=True)
    remove_model = fit_model(train_features, train_target)
    df.loc[df_in.index, PROPENSITY_SCORE_COL] = predict_proba(remove_model, df_in[remove_feature_cols])

    eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    summary = scalar_binary_metrics(eval_test, ACTION_COL, PROPENSITY_SCORE_COL)
    cal = calibration_table(eval_test, ACTION_COL, PROPENSITY_SCORE_COL, bins=CALIBRATION_BINS)
    by_period = metrics_by_period(eval_test, PERIODS_COL, ACTION_COL, PROPENSITY_SCORE_COL)

    save_df(cal, OUTDIR / "propensity_calibration_test.csv")
    save_df(cal, OUTDIR / "remove_calibration_test.csv")
    save_df(by_period, OUTDIR / "propensity_by_period_test.csv")
    save_df(by_period, OUTDIR / "remove_by_period_test.csv")

    model_feature_df = top_series_df(
        "removal",
        "model_importance",
        feature_importance_series(remove_model, remove_feature_cols),
        TOP_FEATURES_TO_SAVE,
    )
    model_feature_df = add_feature_descriptions(model_feature_df, covariate_dict)
    save_df(model_feature_df, OUTDIR / "propensity_top_model_features.csv")

    shap_feature_df = top_series_df(
        "removal",
        "shap_mean_abs",
        shap_importance_series(remove_model, test_features, SHAP_SAMPLE_N),
        TOP_FEATURES_TO_SAVE,
    )
    shap_feature_df = add_feature_descriptions(shap_feature_df, covariate_dict)
    save_df(shap_feature_df, OUTDIR / "propensity_top_shap_features.csv")

    if SAVE_SHAP:
        save_shap_plots(
            pipe=remove_model,
            features=test_features,
            out_prefix=OUTDIR / f"propensity_shap_{MODEL_TYPE}",
            sample_n=SHAP_SAMPLE_N,
        )

    df = restore_original_order(df)
    df = insert_score_columns_before_age(df, [PROPENSITY_SCORE_COL])
    out_scored = OUTDIR / "propensity_scored_panel.csv"
    df.to_csv(out_scored, index=False, float_format="%.6f")

    dump_joblib(
        {
            "model_type": MODEL_TYPE,
            "remove_model": remove_model,
            "features": feature_list,
            "x_cols_remove": remove_feature_cols,
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "split_col": SPLIT_COL,
            "period_hours": feature_spec.get("period_hours"),
            "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
            "risk_set_columns": {"cauti": AT_RISK_CAUTI, "reinsertion": AT_RISK_REINS},
            "modeling_panel_file": str(INFILE),
            "feature_spec_file": str(FEATURE_SPEC_FILE),
        },
        MODEL_DIR / "propensity_model.pkl",
    )

    metrics = {
        "seed": SEED,
        "model_type": MODEL_TYPE,
        "period_hours": feature_spec.get("period_hours"),
        "split": {
            "method": "precomputed patient-level split from modeling_panel.csv",
            "train_rows": int((df[SPLIT_COL] == "train").sum()),
            "test_rows": int((df[SPLIT_COL] == "test").sum()),
            "train_patients": int(df.loc[df[SPLIT_COL] == "train", ID_COL].nunique()),
            "test_patients": int(df.loc[df[SPLIT_COL] == "test", ID_COL].nunique()),
        },
        "preprocessing": {
            "modeling_panel_file": str(INFILE),
            "feature_spec_file": str(FEATURE_SPEC_FILE),
            "hidden_indicator_columns_created_inside_propensity_fit": 0,
            "explicit_feature_count": int(len(feature_list)),
        },
        "test_performance": {"remove_in": summary},
        "auc": {"remove_in": summary["auc"]},
        "n_rows": {
            "all": int(len(df)),
            "decision_eligible": int(len(df_in)),
            "decision_eligible_train": int((df_in[SPLIT_COL] == "train").sum()),
            "decision_eligible_test": int((df_in[SPLIT_COL] == "test").sum()),
        },
        "artifacts": {
            "scored_panel": str(out_scored),
            "model_pkl": str(MODEL_DIR / "propensity_model.pkl"),
            "calibration_test_csv": str(OUTDIR / "propensity_calibration_test.csv"),
            "by_period_test_csv": str(OUTDIR / "propensity_by_period_test.csv"),
        },
        "scored_panel_schema": {
            "required_columns_present": all(col in df.columns for col in [
                ID_COL, STATE_COL, PERIODS_COL, TIME_COL, SPLIT_COL, ACTION_COL,
                Y_CAUTI, Y_REINS, AT_RISK_CAUTI, AT_RISK_REINS, PROPENSITY_SCORE_COL,
            ]),
        },
    }

    (OUTDIR / "propensity_metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2),
        encoding="utf-8",
    )

    print("\n--- SUCCESS PROPENSITY FIT ---", flush=True)
    print(f"Propensity scored panel saved: {out_scored}", flush=True)
    print(f"AUC removal: {summary['auc']}", flush=True)


if __name__ == "__main__":
    main()
