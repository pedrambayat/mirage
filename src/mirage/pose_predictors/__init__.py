"""Pose-prediction wrappers.

Each predictor produces a predicted complex PDB from a `BenchmarkExample`'s
binder + target sequences by shelling out to an external GPU pipeline
(ColabFold, Protenix, Boltz, AF3 …). Wrappers stay pure-Python so they fit
inside the mirage uv env; GPU libraries live in separate envs.
"""

from mirage.pose_predictors.af2m import AF2MPosePredictor
from mirage.pose_predictors.base import AbstractPosePredictor, StagedManifest
from mirage.pose_predictors.protenix import ProtenixPosePredictor

__all__ = [
    "AF2MPosePredictor",
    "AbstractPosePredictor",
    "ProtenixPosePredictor",
    "StagedManifest",
]
