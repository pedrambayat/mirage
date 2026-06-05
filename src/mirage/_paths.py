"""Filesystem anchors shared across the package.

`repo_root()` resolves the mirage project root from this module's location,
so default data/prediction paths are derived in one place rather than being
recomputed (with magic `parents[...]` indices) in every module that needs them.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the mirage repo root (the directory containing ``src/``)."""
    return Path(__file__).resolve().parents[2]


def default_af2m_predictions_root() -> Path:
    """Where the AF2-M wrapper writes per-example outputs by default.

    The single source of truth for this layout: both the AF2-M predictor and
    the scorers that read its outputs derive their default from here, rather
    than re-hardcoding ``data/raw/predictions/af2m``. Lives here (not in the
    af2m predictor module) so scorers can import it without taking a
    scorers→pose_predictors dependency, which would close an import cycle
    against ``pose_predictors.af2m``'s use of ``scorers.base``.
    """
    return repo_root() / "data" / "raw" / "predictions" / "af2m"


def default_protenix_predictions_root() -> Path:
    """Where Protenix writes per-example outputs by default.

    Mirrors ``default_af2m_predictions_root``: the single source of truth for
    the Protenix prediction layout so both the scorer and any staging scripts
    agree on the path without re-hardcoding it.
    """
    return repo_root() / "data" / "raw" / "predictions" / "protenix"
