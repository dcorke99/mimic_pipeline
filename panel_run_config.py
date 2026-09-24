"""Shared real/validation panel paths for the causal-evaluation pipeline."""

from dataclasses import dataclass
from pathlib import Path


PANEL_CHOICES = (
    "real",
    "validation",
    "validation-omitted",
    "validation-randomised",
)

VALIDATION_DIRNAME = "semi_synthetic_measured_confounding"
VALIDATION_PANELS = {
    "validation": (
        "semi_synthetic_panel.csv",
        "semi_synthetic_with_confounding",
    ),
    "validation-omitted": (
        "semi_synthetic_panel_confounder_omitted.csv",
        "confounder_omitted",
    ),
    "validation-randomised": (
        "semi_synthetic_panel_randomised_action.csv",
        "randomised_action",
    ),
}


@dataclass(frozen=True)
class PanelRunPaths:
    panel_name: str
    panel_path: Path
    artefact_root: Path


def add_panel_argument(parser):
    """Add the common panel selector to a script's argument parser."""
    parser.add_argument(
        "--panel",
        choices=PANEL_CHOICES,
        default="real",
        help=(
            "Source panel and isolated artefact tree to use. The default 'real' "
            "preserves the existing production paths."
        ),
    )
    return parser


def resolve_panel_run(repo_root, panel_name):
    """Resolve one panel choice to its source file and run-specific artefact root."""
    repo_root = Path(repo_root)
    if panel_name == "real":
        return PanelRunPaths(
            panel_name=panel_name,
            panel_path=repo_root / "data" / "modelling_panel.csv",
            artefact_root=repo_root / "artefacts",
        )

    panel_filename, run_directory = VALIDATION_PANELS[panel_name]

    validation_root = (
        repo_root / "artefacts" / "validation" / VALIDATION_DIRNAME
    )
    return PanelRunPaths(
        panel_name=panel_name,
        panel_path=validation_root / panel_filename,
        artefact_root=validation_root / "pipeline_runs" / run_directory,
    )
