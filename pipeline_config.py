"""Edit run choices here. Shared paths are listed below; technical settings stay in their scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
ARTEFACT_ROOT = REPO_ROOT / "artefacts"
VALIDATION_DIR = ARTEFACT_ROOT / "semi-synthetic_validation/semi_synthetic_measured_confounding"

# Each panel's input CSV and nuisance-model output directory.
PANEL_PATHS = {
    "real": (DATA_DIR / "modelling_panel.csv", ARTEFACT_ROOT / "nuisance_models"),
    "semi_synthetic_with_confounding": (
        VALIDATION_DIR / "semi_synthetic_panel.csv",
        VALIDATION_DIR / "pipeline_runs/semi_synthetic_with_confounding/nuisance_models",
    ),
    "confounder_omitted": (
        VALIDATION_DIR / "semi_synthetic_panel_confounder_omitted.csv",
        VALIDATION_DIR / "pipeline_runs/confounder_omitted/nuisance_models",
    ),
    "randomised_action": (
        VALIDATION_DIR / "semi_synthetic_panel_randomised_action.csv",
        VALIDATION_DIR / "pipeline_runs/randomised_action/nuisance_models",
    ),
}

# Pipeline stages to execute, in order. Comment out stages to skip them.
# Running a Python stage directly still uses the choices below.
PIPELINE_STAGES = [
    "create_data_panel.py",
    "create_semi_synthetic_panel.py",
    "fit_nuisance_models.py",
    "panel_analysis.py",
    "build_policy_panels.py",
    "evaluate_gformula_policies.py",
    "evaluate_ipw_policies.py",
    "evaluate_aipw_policies.py",
    "check_policy_results.py",
]

# Learners to fit, in order. An empty list skips nuisance fitting.
MODEL_TYPES = [
    # "logistic_regression",
    # "random_forest",
    # "extra_trees",
    "xgboost",
    # "lightgbm",
    # "catboost",
    # "superlearner",
]
RUN_LEARNING_CURVES = False

# Super Learner membership is independent of MODEL_TYPES.
SUPERLEARNER_BASE_MODELS = [
    "logistic_regression", "random_forest", "extra_trees",
    "xgboost", "lightgbm", "catboost",
]

# Panels to use at each stage. Comment out entries to skip them.
# The semi-synthetic builder generates all three validation variants when run.
NUISANCE_PANELS = [
    "real",
    # "semi_synthetic_with_confounding",
    # "confounder_omitted",
    # "randomised_action",
]
POLICY_PANELS = [
    "real",
    # "semi_synthetic_with_confounding",
    # "confounder_omitted",
    # "randomised_action",
]
EVALUATION_PANELS = [
    "real",
    # "semi_synthetic_with_confounding",
    # "confounder_omitted",
    # "randomised_action",
]

# Learner used by all evaluators and descriptive panel analysis.
# For "fixed" evaluation, fit this learner on each selected panel first.
NUISANCE_MODEL_TYPE = "xgboost"
# Select "fixed", "refit", or both; an empty list skips policy evaluation.
BOOTSTRAP_MODES = ["fixed"]

# Shared by policy construction and semi-synthetic known-truth calculations.
POLICY_DAYS = [1, 2, 3, 4, 5]

# Descriptive analysis and cross-estimator result audit each use one panel.
ANALYSIS_PANEL = "real"
ANALYSIS_INCLUDE_AGE_SPLIT = True
RESULT_CHECK_PANEL = "real"
RESULT_CHECK_BOOTSTRAP_MODE = "fixed"
