"""Компоненты задач адвекции, диффузии и реакции."""

from .errors import CapabilityError, ConfigError
from .coefficients import ADRUnits, ReactionLaw, ScalarField, VectorField
from .maps import map_from_functions
from .pde import ADRProblem
from .spaces import ArraySpace, matrix_map
from .state_maps import state_map, forcing_state_map, state_map_from_solver
from .state_residuals import StateResidual

__all__ = ["CapabilityError", "ConfigError", "ArraySpace", "ADRProblem",
           "ADRUnits", "ScalarField", "VectorField", "ReactionLaw",
           "map_from_functions", "matrix_map", "StateResidual",
           "state_map", "forcing_state_map", "state_map_from_solver"]
