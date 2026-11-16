"""Невязки, регуляризация, чувствительность и оптимизация обратных задач."""

from .objectives import compose_maps, make_objective, evaluate_score
from .misfit import QuadraticMisfit, GaussianLikelihood, covariance_metric, working_precision
from .regularization import QuadraticPenalty, CoefficientMetric, p1_matrices, build_penalty
from .optimizers import solve_lbfgsb
from .problem import InverseProblem
from .constraints import analyze_nonnegative_observations, project_nonnegative_equalities
from .functionals import (
    scalar_from_functions,
    sum_functionals,
    scale_functional,
    pullback_functional,
)

__all__ = [
    "compose_maps",
    "make_objective",
    "evaluate_score",
    "QuadraticMisfit",
    "GaussianLikelihood",
    "covariance_metric",
    "working_precision",
    "QuadraticPenalty",
    "CoefficientMetric",
    "p1_matrices",
    "build_penalty",
    "solve_lbfgsb",
    "InverseProblem",
    "project_nonnegative_equalities",
    "analyze_nonnegative_observations",
    "scalar_from_functions",
    "sum_functionals",
    "scale_functional",
    "pullback_functional",
]
