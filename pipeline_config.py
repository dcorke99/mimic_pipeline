from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Each panel's input CSV and nuisance-model output directory.
PANEL_PATHS = {
    "observed": (
        REPO_ROOT / "data/modelling_panel.csv",
        REPO_ROOT / "artefacts/nuisance_models/observed",
    ),
    "semi_synthetic_with_confounding": (
        REPO_ROOT / "artefacts/validation/with_confounding.csv",
        REPO_ROOT / "artefacts/nuisance_models/semi_synthetic_with_confounding",
    ),
    "confounder_omitted": (
        REPO_ROOT / "artefacts/validation/confounder_omitted.csv",
        REPO_ROOT / "artefacts/nuisance_models/confounder_omitted",
    ),
    "randomised_action": (
        REPO_ROOT / "artefacts/validation/randomised_action.csv",
        REPO_ROOT / "artefacts/nuisance_models/randomised_action",
    ),
}

# Intermediate policy panels and final results have the same panel-first layout.
POLICY_DIRS = {
    "observed": REPO_ROOT / "artefacts/policy_panels/observed",
    "semi_synthetic_with_confounding": REPO_ROOT / "artefacts/policy_panels/semi_synthetic_with_confounding",
    "confounder_omitted": REPO_ROOT / "artefacts/policy_panels/confounder_omitted",
    "randomised_action": REPO_ROOT / "artefacts/policy_panels/randomised_action",
}
RESULT_DIRS = {
    "observed": REPO_ROOT / "artefacts/results/observed",
    "semi_synthetic_with_confounding": REPO_ROOT / "artefacts/results/semi_synthetic_with_confounding",
    "confounder_omitted": REPO_ROOT / "artefacts/results/confounder_omitted",
    "randomised_action": REPO_ROOT / "artefacts/results/randomised_action",
}

# Pipeline stages to execute, in order. Comment out stages to skip them.
# Running a Python stage directly still uses the choices below.
PIPELINE_STAGES = [
    "create_observed_panel.py",
    "create_semi_synthetic_panel.py",
    "fit_nuisance_models.py",
    "panel_analysis.py",
    "build_policy_panels.py",
    "evaluate_gformula_policies.py",
    "evaluate_policy_dml_policies.py",
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

# Super Learner membership is independent of MODEL_TYPES.
SUPERLEARNER_BASE_MODELS = [
    "logistic_regression", "random_forest", "extra_trees",
    "xgboost", "lightgbm", "catboost",
]

# Panels to use at each stage. Comment out entries to skip them.
# The semi-synthetic builder generates all three validation variants when run.
NUISANCE_PANELS = [
    "observed",
    "semi_synthetic_with_confounding",
    "confounder_omitted",
    "randomised_action",
]
POLICY_PANELS = [
    "observed",
    "semi_synthetic_with_confounding",
    "confounder_omitted",
    "randomised_action",
]
EVALUATION_PANELS = [
    "observed",
    "semi_synthetic_with_confounding",
    "confounder_omitted",
    "randomised_action",
]

# Learner used by all evaluators and descriptive panel analysis.
# For "fixed" evaluation, fit this learner on each selected panel first.
NUISANCE_MODEL_TYPE = "xgboost"
# Select "fixed", "refit", or both; an empty list skips policy evaluation.
BOOTSTRAP_MODES = ["fixed"]

# Shared by policy construction and semi-synthetic known-truth calculations.
POLICY_DAYS = [1, 2, 3, 4, 5]

# Policy-DML follow-up hours from insertion, by panel.
POLICY_DML_HORIZON_HOURS = {
    "observed": 30 * 24,
    "semi_synthetic_with_confounding": 14 * 24,
    "confounder_omitted": 14 * 24,
    "randomised_action": 14 * 24,
}

# Policy-DML continuation learner: "ridge", "extra_trees", "random_forest" or "xgboost".
POLICY_DML_CONTINUATION_LEARNER = "xgboost"
POLICY_DML_N_BOOTSTRAP = 100
POLICY_DML_CLIPPING_SENSITIVITY = ()

# Descriptive analysis uses one panel.
ANALYSIS_PANEL = "observed"
ANALYSIS_INCLUDE_AGE_SPLIT = True
