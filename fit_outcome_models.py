import json

import numpy as np
import pandas as pd

from model_utils import (
    ACTION_COL,
    AT_RISK_CAUTI,
    AT_RISK_REINS,
    CALIBRATION_BINS,
    COVARIATE_DICT_FILE,
    END_REASON_COL,
    FEATURE_SPEC_FILE,
    ID_COL,
    INFILE,
    LAST_PERIOD_COL,
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


OUTCOME_SCORE_COLS = [
    "p_cauti_if_keep",
    "p_cauti_if_remove",
    "p_cauti_if_out",
    "p_reins_if_remove",
    "p_reins_if_out",
]


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    feature_spec = load_feature_spec()
    covariate_dict = pd.read_csv(COVARIATE_DICT_FILE)
    df = add_prior_cauti_count(load_panel())

    feature_list = feature_spec["features"]
    cauti_feature_cols = feature_spec["x_cols_cauti"]
    reins_feature_cols = feature_spec["x_cols_reins"]

    print("Outcome models use precomputed explicit features only.")
    print("Hidden imputer indicator columns created inside outcome fit: 0")
    print(f"Loaded features: {len(feature_list)}")

    for col in OUTCOME_SCORE_COLS:
        df[col] = np.zeros(len(df), dtype=float)

    df_in = df[
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    df_cauti = df[
        (df["prior_cauti_count"] == 0) &
        (df[AT_RISK_CAUTI] == 1)
    ].copy()

    df_out_reins = df[df[AT_RISK_REINS] == 1].copy()
    df_reins_fit = df_out_reins[
        ~(
            (df_out_reins[LAST_PERIOD_COL] == 1) &
            (df_out_reins[END_REASON_COL] == "icu_end") &
            (df_out_reins[Y_REINS] == 0)
        )
    ].copy()

    cauti_train_features = df_cauti.loc[df_cauti[SPLIT_COL] == "train", cauti_feature_cols]
    cauti_train_target = df_cauti.loc[df_cauti[SPLIT_COL] == "train", Y_CAUTI].astype(int)
    cauti_test_features = df_cauti.loc[df_cauti[SPLIT_COL] == "test", cauti_feature_cols]

    print(f"Fitting CAUTI outcome model with {MODEL_TYPE}...", flush=True)
    cauti_model = fit_model(cauti_train_features, cauti_train_target)

    df_out_cauti = df[
        (df[STATE_COL] == "out") &
        (df[AT_RISK_CAUTI] == 1)
    ].copy()
    df.loc[df_out_cauti.index, "p_cauti_if_out"] = predict_proba(
        cauti_model,
        df_out_cauti[cauti_feature_cols],
    )

    cauti_keep_features = df_in[cauti_feature_cols].copy()
    cauti_keep_features["state_is_out"] = 0

    cauti_remove_features = df_in[cauti_feature_cols].copy()
    cauti_remove_features[PERIODS_COL] = 1
    cauti_remove_features["state_is_out"] = 1

    df.loc[df_in.index, "p_cauti_if_keep"] = predict_proba(cauti_model, cauti_keep_features)
    df.loc[df_in.index, "p_cauti_if_remove"] = predict_proba(cauti_model, cauti_remove_features)

    reins_train_features = df_reins_fit.loc[df_reins_fit[SPLIT_COL] == "train", reins_feature_cols]
    reins_train_target = df_reins_fit.loc[df_reins_fit[SPLIT_COL] == "train", Y_REINS].astype(int)
    reins_test_features = df_reins_fit.loc[df_reins_fit[SPLIT_COL] == "test", reins_feature_cols]

    print(f"Fitting reinsertion outcome model with {MODEL_TYPE}...", flush=True)
    reins_model = fit_model(reins_train_features, reins_train_target)

    df.loc[df_out_reins.index, "p_reins_if_out"] = predict_proba(
        reins_model,
        df_out_reins[reins_feature_cols],
    )

    reins_remove_features = df_in[reins_feature_cols].copy()
    reins_remove_features[PERIODS_COL] = 1
    df.loc[df_in.index, "p_reins_if_remove"] = predict_proba(reins_model, reins_remove_features)

    cauti_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df["prior_cauti_count"] == 0) &
        (df[AT_RISK_CAUTI] == 1)
    ].copy()
    cauti_eval_test["p_cauti_obs_eval"] = np.where(
        cauti_eval_test[STATE_COL] == "in",
        cauti_eval_test["p_cauti_if_keep"],
        cauti_eval_test["p_cauti_if_out"],
    )

    reins_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df[AT_RISK_REINS] == 1)
    ].copy()
    reins_eval_test = reins_eval_test[
        ~(
            (reins_eval_test[LAST_PERIOD_COL] == 1) &
            (reins_eval_test[END_REASON_COL] == "icu_end") &
            (reins_eval_test[Y_REINS] == 0)
        )
    ].copy()

    cauti_summary = scalar_binary_metrics(cauti_eval_test, Y_CAUTI, "p_cauti_obs_eval")
    reins_summary = scalar_binary_metrics(reins_eval_test, Y_REINS, "p_reins_if_out")

    cauti_cal = calibration_table(cauti_eval_test, Y_CAUTI, "p_cauti_obs_eval", bins=CALIBRATION_BINS)
    reins_cal = calibration_table(reins_eval_test, Y_REINS, "p_reins_if_out", bins=CALIBRATION_BINS)
    cauti_by_state_period = metrics_by_period(
        cauti_eval_test,
        PERIODS_COL,
        Y_CAUTI,
        "p_cauti_obs_eval",
        group_cols=[STATE_COL],
    )
    reins_by_period = metrics_by_period(reins_eval_test, PERIODS_COL, Y_REINS, "p_reins_if_out")

    save_df(cauti_cal, OUTDIR / "outcome_cauti_calibration_test.csv")
    save_df(cauti_cal, OUTDIR / "cauti_calibration_test.csv")
    save_df(reins_cal, OUTDIR / "outcome_reinsertion_calibration_test.csv")
    save_df(reins_cal, OUTDIR / "reinsertion_calibration_test.csv")
    save_df(cauti_by_state_period, OUTDIR / "outcome_cauti_by_state_period_test.csv")
    save_df(cauti_by_state_period, OUTDIR / "cauti_by_state_period_test.csv")
    save_df(reins_by_period, OUTDIR / "outcome_reinsertion_by_period_test.csv")
    save_df(reins_by_period, OUTDIR / "reinsertion_by_period_test.csv")

    model_feature_df = pd.concat(
        [
            top_series_df("cauti", "model_importance", feature_importance_series(cauti_model, cauti_feature_cols), TOP_FEATURES_TO_SAVE),
            top_series_df("reinsertion", "model_importance", feature_importance_series(reins_model, reins_feature_cols), TOP_FEATURES_TO_SAVE),
        ],
        ignore_index=True,
    )
    model_feature_df = add_feature_descriptions(model_feature_df, covariate_dict)
    save_df(model_feature_df, OUTDIR / "outcome_top_model_features.csv")

    shap_feature_df = pd.concat(
        [
            top_series_df("cauti", "shap_mean_abs", shap_importance_series(cauti_model, cauti_test_features, SHAP_SAMPLE_N), TOP_FEATURES_TO_SAVE),
            top_series_df("reinsertion", "shap_mean_abs", shap_importance_series(reins_model, reins_test_features, SHAP_SAMPLE_N), TOP_FEATURES_TO_SAVE),
        ],
        ignore_index=True,
    )
    shap_feature_df = add_feature_descriptions(shap_feature_df, covariate_dict)
    save_df(shap_feature_df, OUTDIR / "outcome_top_shap_features.csv")

    if SAVE_SHAP:
        save_shap_plots(
            pipe=cauti_model,
            features=cauti_test_features,
            out_prefix=OUTDIR / f"outcome_cauti_shap_{MODEL_TYPE}",
            sample_n=SHAP_SAMPLE_N,
        )
        save_shap_plots(
            pipe=reins_model,
            features=reins_test_features,
            out_prefix=OUTDIR / f"outcome_reinsertion_shap_{MODEL_TYPE}",
            sample_n=SHAP_SAMPLE_N,
        )

    df = restore_original_order(df)
    df = insert_score_columns_before_age(df, OUTCOME_SCORE_COLS)
    out_scored = OUTDIR / "outcome_scored_panel.csv"
    df.to_csv(out_scored, index=False, float_format="%.6f")

    dump_joblib(
        {
            "model_type": MODEL_TYPE,
            "cauti_model": cauti_model,
            "reins_model": reins_model,
            "features": feature_list,
            "x_cols_cauti": cauti_feature_cols,
            "x_cols_reins": reins_feature_cols,
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "split_col": SPLIT_COL,
            "period_hours": feature_spec.get("period_hours"),
            "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
            "risk_set_columns": {"cauti": AT_RISK_CAUTI, "reinsertion": AT_RISK_REINS},
            "modeling_panel_file": str(INFILE),
            "feature_spec_file": str(FEATURE_SPEC_FILE),
        },
        MODEL_DIR / "outcome_models.pkl",
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
            "hidden_indicator_columns_created_inside_outcome_fit": 0,
            "explicit_feature_count": int(len(feature_list)),
        },
        "auc": {
            "cauti_transition": cauti_summary["auc"],
            "reins_out": reins_summary["auc"],
        },
        "test_performance": {
            "cauti_transition": cauti_summary,
            "reins_out": reins_summary,
        },
        "n_rows": {
            "all": int(len(df)),
            "in_no_prior_cauti": int(len(df_in)),
            "at_risk_cauti": int((df[AT_RISK_CAUTI] == 1).sum()),
            "at_risk_reinsertion": int((df[AT_RISK_REINS] == 1).sum()),
            "cauti_fit": int(len(df_cauti)),
            "reinsertion_fit": int(len(df_reins_fit)),
        },
        "artifacts": {
            "scored_panel": str(out_scored),
            "model_pkl": str(MODEL_DIR / "outcome_models.pkl"),
            "cauti_calibration_test_csv": str(OUTDIR / "outcome_cauti_calibration_test.csv"),
            "reinsertion_calibration_test_csv": str(OUTDIR / "outcome_reinsertion_calibration_test.csv"),
        },
        "scored_panel_schema": {
            "required_columns_present": all(col in df.columns for col in [
                ID_COL, STATE_COL, PERIODS_COL, TIME_COL, SPLIT_COL, ACTION_COL,
                Y_CAUTI, Y_REINS, AT_RISK_CAUTI, AT_RISK_REINS, *OUTCOME_SCORE_COLS,
            ]),
        },
    }

    (OUTDIR / "outcome_metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2),
        encoding="utf-8",
    )

    print("\n--- SUCCESS OUTCOME FIT ---", flush=True)
    print(f"Outcome scored panel saved: {out_scored}", flush=True)
    print(f"AUC CAUTI: {cauti_summary['auc']}", flush=True)
    print(f"AUC Reinsertion: {reins_summary['auc']}", flush=True)


if __name__ == "__main__":
    main()
