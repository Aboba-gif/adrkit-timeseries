"""Невязки, регуляризация, чувствительность и оптимизация обратных задач."""

from .objectives import compose_maps, make_objective, evaluate_score
from .misfit import (
    QuadraticMisfit,
    GaussianLikelihood,
    covariance_metric,
    working_precision,
)
from .regularization import (
    QuadraticPenalty,
    CoefficientMetric,
    p1_matrices,
    build_penalty,
)
from .optimizers import solve_lbfgsb
from .problem import InverseProblem
from .constraints import (
    NonnegativeAnalysisLinearization,
    StationaryBoxInverse,
    analyze_nonnegative_map,
    analyze_nonnegative_observations,
    nonnegative_analysis_linear_operator,
    project_nonnegative_equalities,
    stationary_box_inverse,
)
from .design import (
    CovarianceFactor,
    DesignCandidate,
    InnovationBudget,
    JointMomentRisk,
    affine_goal_risk,
    innovation_covariance_budget,
    joint_moment_risk,
    profile_information,
    robust_ball_maximum,
    select_designs,
)
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
    "analyze_nonnegative_map",
    "NonnegativeAnalysisLinearization",
    "nonnegative_analysis_linear_operator",
    "StationaryBoxInverse",
    "stationary_box_inverse",
    "CovarianceFactor",
    "DesignCandidate",
    "InnovationBudget",
    "JointMomentRisk",
    "affine_goal_risk",
    "innovation_covariance_budget",
    "joint_moment_risk",
    "profile_information",
    "robust_ball_maximum",
    "select_designs",
    "scalar_from_functions",
    "sum_functionals",
    "scale_functional",
    "pullback_functional",
]
