"""Validate run choices and derive paths; no editable run choices live here."""

from pathlib import Path

import pipeline_config as run_config


REPO_ROOT = Path(__file__).resolve().parent
PANEL_NAMES = ("real", "semi_synthetic_with_confounding", "confounder_omitted", "randomised_action")
MODEL_NAMES = ("logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost", "superlearner")
STAGE_NAMES = (
    "create_data_panel.py", "create_semi_synthetic_panel.py", "fit_nuisance_models.py",
    "panel_analysis.py", "build_policy_panels.py", "evaluate_gformula_policies.py",
    "evaluate_ipw_policies.py", "evaluate_aipw_policies.py", "check_policy_results.py",
)


def validate_selection(name, values, allowed):
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{name} must be a list of names")
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown choices in {name}: {sorted(unknown)}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate choices in {name}")


def validate_run_choices():
    for name, allowed in (
        ("PIPELINE_STAGES", STAGE_NAMES), ("MODEL_TYPES", MODEL_NAMES),
        ("SUPERLEARNER_BASE_MODELS", MODEL_NAMES[:-1]),
        ("NUISANCE_PANELS", PANEL_NAMES), ("POLICY_PANELS", PANEL_NAMES),
        ("EVALUATION_PANELS", PANEL_NAMES), ("BOOTSTRAP_MODES", ("fixed", "refit")),
    ):
        validate_selection(name, getattr(run_config, name), allowed)
    if not run_config.SUPERLEARNER_BASE_MODELS:
        raise ValueError("SUPERLEARNER_BASE_MODELS must contain at least one learner")
    if run_config.NUISANCE_MODEL_TYPE not in MODEL_NAMES:
        raise ValueError(f"Unknown NUISANCE_MODEL_TYPE: {run_config.NUISANCE_MODEL_TYPE!r}")
    for name in ("ANALYSIS_PANEL", "RESULT_CHECK_PANEL"):
        if getattr(run_config, name) not in PANEL_NAMES:
            raise ValueError(f"Unknown {name}: {getattr(run_config, name)!r}")
    if run_config.RESULT_CHECK_BOOTSTRAP_MODE not in ("fixed", "refit"):
        raise ValueError(f"Unknown RESULT_CHECK_BOOTSTRAP_MODE: {run_config.RESULT_CHECK_BOOTSTRAP_MODE!r}")
    if run_config.AIPW_RESIDUAL_NORMALISATION not in ("hajek", "ht"):
        raise ValueError(f"Unknown AIPW_RESIDUAL_NORMALISATION: {run_config.AIPW_RESIDUAL_NORMALISATION!r}")
    for name in ("RUN_LEARNING_CURVES", "ANALYSIS_INCLUDE_AGE_SPLIT"):
        if not isinstance(getattr(run_config, name), bool):
            raise ValueError(f"{name} must be True or False")
    days = run_config.POLICY_DAYS
    if (not isinstance(days, (list, tuple)) or not days
            or any(type(day) is not int or day < 1 for day in days)
            or len(days) != len(set(days))):
        raise ValueError("POLICY_DAYS must be a nonempty list of unique positive integers")


def get_data_dir():
    return REPO_ROOT / "data"


def get_artefact_root():
    return REPO_ROOT / "artefacts"


def get_validation_dir():
    return get_artefact_root() / "semi-synthetic_validation/semi_synthetic_measured_confounding"


def get_panel_run(name):
    if name not in PANEL_NAMES:
        raise ValueError(f"Unknown panel: {name!r}")
    if name == "real":
        return name, get_data_dir() / "modelling_panel.csv", get_artefact_root() / "nuisance_models"
    filename = {
        "semi_synthetic_with_confounding": "semi_synthetic_panel.csv",
        "confounder_omitted": "semi_synthetic_panel_confounder_omitted.csv",
        "randomised_action": "semi_synthetic_panel_randomised_action.csv",
    }[name]
    validation = get_validation_dir()
    return name, validation / filename, validation / "pipeline_runs" / name / "nuisance_models"


def get_panel_runs(names):
    validate_selection("panel selection", names, PANEL_NAMES)
    return tuple(get_panel_run(name) for name in names)


def first_panel_run(runs):
    # Empty stage selections still allow modules to import for shared helpers.
    return runs[0] if runs else get_panel_run("real")
