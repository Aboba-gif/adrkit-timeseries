"""Rectangular spatial transfer and the fully implicit time integrator."""

import numpy as np
import pytest

from adrkit.integrators import time_integrator
from adrkit.spatial import spatial_discretization


def medium(initial, *, horizon=1.0):
    return dict(
        diffusion=0.1,
        velocity=(0.0, 0.0),
        linear_loss=0.0,
        initial_state=initial,
        horizon=horizon,
        source_position=(0.0, 0.0),
        boundary_condition="neumann",
    )


def test_dual_cell_transfer_preserves_substance_positivity_constants_and_covectors():
    coarse, fine = [
        spatial_discretization("rectangle-upwind", (-3, 3, -6, 6), h) for h in (1, 0.75)
    ]
    rng = np.random.default_rng(975)
    for source, target in ((coarse, fine), (fine, coarse)):
        old, new = [grid.assemble(medium(0.0), time_steps=2) for grid in (source, target)]
        matrix = source.neumann_transfer_to(target)
        field, dual = rng.random(old.size), rng.normal(size=new.size)
        assert matrix.data.min() >= 0
        np.testing.assert_allclose(matrix @ np.ones(old.size), 1.0, atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            new.spatial_quadrature_weights @ matrix,
            old.spatial_quadrature_weights,
            atol=1e-14,
            rtol=0,
        )
        np.testing.assert_allclose(dual @ (matrix @ field), (matrix.T @ dual) @ field)
        for grid in (old, new):
            np.testing.assert_allclose(grid.spatial_quadrature_weights @ grid.load, 1.0)
    other = spatial_discretization("rectangle-upwind", (-4, 2, -6, 6), 1)
    with pytest.raises(ValueError, match="identical rectangular bounds"):
        coarse.neumann_transfer_to(other)


def test_backward_euler_converges_at_first_order_for_quadratic_decay():
    method = time_integrator(
        {"method": "backward-euler", "residual_tolerance": 1e-13},
        device="cpu",
        required=("forward",),
    )
    mesh = spatial_discretization("rectangle-upwind", (-1, 1, -1, 1), 0.5)
    errors = []
    for steps in (8, 16, 32):
        space = mesh.assemble(medium(0.7), time_steps=steps)
        action = method.prepare(space, (0.1, 0.0, 0.0, 0.0), {}, None)
        result = method.solve(
            action, space.initial_state, np.zeros(steps), beta=0.5, load=space.load
        )
        assert method.valid(method.diagnostics(result))
        errors.append(np.max(np.abs(result.states[-1] - 0.7 / (1 + 0.5 * 0.7))))
    assert np.all((np.asarray(errors[:-1]) / errors[1:]) > 1.85)
    assert np.all((np.asarray(errors[:-1]) / errors[1:]) < 2.1)


def test_discretization_checks_unknown_or_masked_physical_inputs():
    with pytest.raises(ValueError, match="masked"):
        spatial_discretization(
            "rectangle-upwind", np.ma.array([0.0, 1.0, 0.0, 1.0], mask=[0, 1, 0, 0]), 0.1
        )
    with pytest.raises(ValueError, match="real scalar"):
        time_integrator(
            {"method": "backward-euler", "residual_tolerance": np.ma.masked}, device="cpu"
        )
    with pytest.raises(ValueError, match="does not support"):
        time_integrator(
            {"method": "backward-euler", "residual_tolerance": 1e-11},
            device="cuda",
            required=("forward",),
        )
