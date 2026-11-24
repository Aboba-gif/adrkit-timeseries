"""Trajectory factories for a rectangular ADR discretization and interval-mean point loads."""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .backends.bounded import BoundedSolver
from ._numeric import real_array, real_scalar
from .coefficients import ReactionLaw
from .time_integration import PositiveAffineAction, poisson_degree, solve_strang


def _transport_coefficients(theta):
    values = real_array(theta, name="diffusion, velocity and growth")
    if values.shape != (4,) or values[0] <= 0:
        raise ValueError("Expected (D, vx, vy, alpha) with positive diffusion")
    return values


@dataclass(frozen=True)
class StrangAffine:
    """Second-order splitting with a fixed uniformization plan over the coefficient box."""

    tolerance: float
    device: str
    method = "strang-affine"
    version = 1
    devices = frozenset(("cpu", "cuda"))
    capabilities = frozenset(("forward", "joint_jvp", "joint_vjp", "joint_hvp"))

    def plan(self, space, bounds):
        """Fix the polynomial degree independently of the active coefficients."""
        bounds = real_array(bounds, name="transport bounds")
        if bounds.shape != (3, 2) or np.any(bounds[:, 0] > bounds[:, 1]) or bounds[0, 0] <= 0:
            raise ValueError("Expected finite ordered bounds for diffusion and the two velocities")
        velocity = np.max(np.abs(bounds[1:3]), axis=1)
        rate = float(
            2 * bounds[0, 1] * (space.hx**-2 + space.hy**-2)
            + velocity[0] / space.hx
            + velocity[1] / space.hy
        )
        rate = float(np.nextafter(rate, np.inf))
        return rate, poisson_degree(rate * space.dt, self.tolerance)

    def prepare(self, space, theta, partials, plan):
        """Assemble the affine step on this grid for one coefficient vector."""
        theta = _transport_coefficients(theta)
        rate, degree = plan
        operator = sum(
            theta[j] * partials[name] for j, name in enumerate(("diffusion", "v_x", "v_y"))
        )
        return PositiveAffineAction(
            operator,
            theta[3],
            step=space.dt,
            rate=rate,
            degree=degree,
            backend="cupy" if self.device == "cuda" else "numpy",
        )

    solve = staticmethod(solve_strang)

    def diagnostics(self, trajectory):
        action = trajectory.affine
        return {
            "uniformization": {
                "rate": action.rate,
                "degree": action.degree,
                "tail": action.transport_tail,
                "operator_tolerance": self.tolerance,
            },
            "polynomial_truncation_bound": float(trajectory.polynomial_error_bound),
            "polynomial_truncation_scope": "state_linf; polynomial truncation only; excludes rounding, time splitting, source averaging, spatial approximation and parameter error",
        }

    def valid(self, summary):
        return (
            np.isfinite(summary["polynomial_truncation_bound"])
            and summary["uniformization"]["tail"] <= self.tolerance
        )


@dataclass(frozen=True)
class _ImplicitOperators:
    space: object
    coefficients: tuple
    array_module = np
    to_host = staticmethod(np.asarray)


@dataclass(frozen=True)
class BackwardEuler:
    """Fully implicit first-order CPU solver for transport, growth and quadratic loss."""

    tolerance: float
    device: str
    method = "backward-euler"
    version = 1
    devices = frozenset(("cpu",))
    capabilities = frozenset(("forward",))

    def plan(self, space, bounds):
        """Newton steps do not require a bounding polynomial plan."""
        return None

    def prepare(self, space, theta, partials, plan):
        """Bind the grid and the transport and growth coefficients."""
        return _ImplicitOperators(space, tuple(_transport_coefficients(theta)))

    def solve(self, action, initial, forcing, *, beta, load):
        """Advance with nonnegative interval-mean intensities at the assembled point source."""
        space = action.space
        diffusion, vx, vy, alpha = action.coefficients
        if space.load is None or not np.array_equal(real_array(load, name="load"), space.load):
            raise ValueError(
                "Backward Euler requires the point load assembled on its spatial grid"
            )
        if alpha * space.dt >= 1:
            raise ValueError("Backward Euler with positive growth requires step * alpha < 1")
        initial = real_array(initial, name="initial")
        forcing = real_array(forcing, name="forcing")
        beta = real_scalar(beta, name="beta")
        if initial.shape != (space.size,) or np.any(initial < 0):
            raise ValueError("Backward Euler requires a nonnegative nodal initial field")
        if forcing.ndim != 1 or not forcing.size or np.any(forcing < 0) or beta < 0:
            raise ValueError("Backward Euler requires nonnegative interval intensities and beta")
        reaction = ReactionLaw(
            lambda u, xy, time: u * (beta * u - alpha),
            lambda u, xy, time: 2 * beta * u - alpha,
            pure=True,
        )
        solver = BoundedSolver(
            {
                "diffusion": diffusion,
                "velocity": (vx, vy),
                "linear_loss": 0.0,
                "horizon": len(forcing) * space.dt,
                "initial_state": initial,
                "boundary_condition": space.boundary_condition,
                "source_position": space.xy[space.source_index],
                "background": 0.0,
                "reaction": reaction,
            },
            domain=space.bounds,
            interior_points=space.interior_points,
            time_steps=len(forcing),
            residual_tolerance=self.tolerance,
        )
        return solver.solve_controls(np.r_[0.0, forcing])

    def diagnostics(self, trajectory):
        return {
            "max_scaled_residual": trajectory.max_scaled_residual,
            "residual_tolerance": self.tolerance,
            "nonlinear_iterations": trajectory.nonlinear_iterations,
        }

    def valid(self, summary):
        return 0 <= summary["max_scaled_residual"] <= self.tolerance


def time_integrator(settings, *, device, required=()):
    """Resolve method-specific accuracy settings and check the capabilities a caller needs."""
    methods = {
        StrangAffine.method: (StrangAffine, "operator_tolerance"),
        BackwardEuler.method: (BackwardEuler, "residual_tolerance"),
    }
    if not isinstance(settings, Mapping):
        raise ValueError("Time integration settings must be a mapping")
    method = settings.get("method")
    if not isinstance(method, str) or method not in methods:
        raise ValueError(f"Unknown time integrator {method!r}")
    cls, option = methods[method]
    if device not in cls.devices or not set(required) <= cls.capabilities:
        raise ValueError(f"{method} does not support {', '.join(sorted(required))} on {device}")
    if set(settings) != {"method", option}:
        raise ValueError(f"{method} requires the accuracy setting {option!r}")
    value = real_scalar(settings[option], name=option)
    if not 0 < value < 1:
        raise ValueError(f"{option} must lie strictly between zero and one")
    return cls(float(value), device)
