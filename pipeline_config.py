"""Run choices grouped by consuming script; shared settings are defined once."""

from pathlib import Path


# ============================================================================
# Shared paths
# ============================================================================
# Used by nuisance fitting, policy construction, analysis and evaluation.

REPO_ROOT = Path(__file__).resolve().parent

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


# ============================================================================
# Shared evaluation choices
# ============================================================================
# Evaluation panels, learners and bootstrap modes: both policy evaluators.
# Fixed evaluation requires saved nuisance predictions for each selected learner.
# Empty panel, learner or bootstrap-mode selections skip the corresponding work.

EVALUATION_PANELS = [
    # "observed",
    "semi_synthetic_with_confounding",
    # "confounder_omitted",
    # "randomised_action",
]

EVALUATION_MODEL_TYPES = [
    # "xgboost",
    "catboost",
    # "superlearner",
]

BOOTSTRAP_MODES = ["refit"]


# ============================================================================
# run_pipeline.py
# ============================================================================
# Stages execute in order. Comment out stages to skip them.
# Running a stage directly still uses the settings in this file.

PIPELINE_STAGES = [
    # "create_observed_panel.py",
    # "create_semi_synthetic_panel.py",
    # "fit_nuisance_models.py",
    # "panel_analysis.py",
    # "build_policy_panels.py",
    # "evaluate_gformula_policies.py",
    "evaluate_policy_dml_policies.py",
]


# ============================================================================
# create_semi_synthetic_panel.py
# ============================================================================
# Uses POLICY_DAYS from the policy construction section and POLICY_DML_HORIZON_HOURS below.
# Always creates all three validation variants; simulation settings stay in the script.


# ============================================================================
# fit_nuisance_models.py
# ============================================================================
# Learners and panels to fit, in order; empty lists skip fitting.
# Super Learner membership is independent of MODEL_TYPES.

MODEL_TYPES = [
    # "logistic_regression",
    # "random_forest",
    # "extra_trees",
    # "xgboost",
    # "lightgbm",
    "catboost",
    # "superlearner",
]

SUPERLEARNER_BASE_MODELS = [
    "logistic_regression", "random_forest", "extra_trees",
    "xgboost", "lightgbm", "catboost",
]

NUISANCE_PANELS = [
    "observed",
    # "semi_synthetic_with_confounding",
    # "confounder_omitted",
    # "randomised_action",
]


# ============================================================================
# panel_analysis.py
# ============================================================================
# Descriptive analysis uses one panel and one nuisance learner.

ANALYSIS_PANEL = "observed"

NUISANCE_MODEL_TYPE = "catboost"

ANALYSIS_INCLUDE_AGE_SPLIT = True


# ============================================================================
# build_policy_panels.py
# ============================================================================
# Policy construction: panels and one-based catheter-removal days.
# POLICY_DAYS is also used by create_semi_synthetic_panel.py for known-truth values.
# Output paths use POLICY_DIRS above.

POLICY_DAYS = [1, 2, 3, 4, 5]

POLICY_PANELS = [
    "observed",
    # "semi_synthetic_with_confounding",
    # "confounder_omitted",
    # "randomised_action",
]


# ============================================================================
# evaluate_gformula_policies.py
# ============================================================================
# Uses shared EVALUATION_PANELS, EVALUATION_MODEL_TYPES and BOOTSTRAP_MODES.
# Number of patient bootstrap draws for each g-formula run.

GFORMULA_N_BOOTSTRAP = 100


# ============================================================================
# evaluate_policy_dml_policies.py
# ============================================================================
# Uses shared EVALUATION_PANELS, EVALUATION_MODEL_TYPES and BOOTSTRAP_MODES.
# Runs every behaviour / continuation learner combination; an empty continuation
# list skips DML. Horizon hours from insertion are also used by the synthetic builder.

POLICY_DML_HORIZON_HOURS = {
    "observed": 30 * 24,
    "semi_synthetic_with_confounding": 14 * 24,
    "confounder_omitted": 14 * 24,
    "randomised_action": 14 * 24,
}

POLICY_DML_CONTINUATION_LEARNERS = [
    # "ridge",
    # "extra_trees",
    # b"random_forest",
    # "xgboost",
    "catboost",
]

POLICY_DML_N_BOOTSTRAP = 1

POLICY_DML_CLIPPING_SENSITIVITY = ()
