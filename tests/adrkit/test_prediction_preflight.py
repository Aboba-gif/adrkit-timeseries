"""Проверки перед созданием снимков и вычислением состояния."""
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.loads import IntervalAverageLoad
from adrkit.observations.grid import GridObservation
import adrkit.predictions as predictions
from adrkit.predictions import GridPrediction, SelectedPrediction, solver_state_space
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace, MatrixMap


def vector_space(unit="1"):
    return ArraySpace((2,), axes=("entry",), coordinates=((0, 1),), units=unit)


@pytest.mark.parametrize("space_name", ["domain", "codomain"])
@pytest.mark.parametrize("kind", ["missing", "none", "wrong_type"])
def test_selection_requires_spaces_before_selection_or_model_calls(monkeypatch, space_name, kind):
    layout = vector_space()
    base = SimpleNamespace(domain=layout, codomain=layout)
    if kind == "missing":
        delattr(base, space_name)
    else:
        setattr(base, space_name, None if kind == "none" else "array space")
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid spaces must fail before selection or model calls")
    base.predict = base.jvp = base.vjp = base.jacobian = forbidden
    monkeypatch.setattr(predictions, "Selection", forbidden)
    with pytest.raises(TypeError, match="domain and codomain must be ArraySpace"):
        SelectedPrediction(base, [0])


def test_selection_constructor_only_reads_declared_spaces():
    layout = vector_space()
    base = SimpleNamespace(domain=layout, codomain=layout)
    selected = SelectedPrediction(base, [1])
    assert selected.base is base and selected.domain is layout
    assert selected.codomain.coordinates == ((1,),)


def test_selected_prediction_preserves_actions_metadata_and_row_order(monkeypatch):
    domain = ArraySpace((2,), axes=("control",), coordinates=(("east", "west"),),
                        units="dimensionless", dtype=np.float32)
    codomain = ArraySpace((4,), axes=("measurement",),
        coordinates=(("A", "B", "C", "D"),), units="mol/s", scale=2.5,
        dtype=np.float32, binding="custom_observation")
    matrix = np.array([[1., 2.], [3., 5.], [7., 11.], [13., 17.]], dtype=np.float32)
    base = MatrixMap(matrix, domain, codomain)
    def jacobian(at):
        return np.column_stack([base.jvp(at, d) for d in np.eye(domain.size, dtype=domain.dtype)])
    monkeypatch.setattr(base, "jacobian", jacobian, raising=False)
    base.trajectory = SimpleNamespace(max_scaled_residual=0.)
    rows = np.array([3, 1])
    selected = SelectedPrediction(base, rows)
    rows[0] = 0
    selected.rows[0] = 2
    expected_rows = np.array([3, 1])
    np.testing.assert_array_equal(selected.rows, expected_rows)
    assert selected.domain is domain
    assert selected.codomain == ArraySpace((2,), axes=codomain.axes,
        coordinates=(("D", "B"),), units=codomain.units, scale=codomain.scale,
        pairing=codomain.pairing, dtype=codomain.dtype, binding=codomain.binding)
    point = np.array([2., -1.], dtype=np.float32)
    direction = np.array([1., 2.], dtype=np.float32)
    np.testing.assert_array_equal(selected.predict(point), (matrix@point)[expected_rows])
    np.testing.assert_array_equal(selected.jvp(point, direction), (matrix@direction)[expected_rows])
    np.testing.assert_array_equal(selected.jacobian(point), matrix[expected_rows])
    cotangent = np.array([2., -3.], dtype=np.float32)
    scattered = []
    transpose = base.vjp
    def vjp(at, dual):
        scattered.append(dual.copy())
        return transpose(at, dual)
    monkeypatch.setattr(base, "vjp", vjp)
    np.testing.assert_array_equal(selected.vjp(point, cotangent), matrix[expected_rows].T@cotangent)
    np.testing.assert_array_equal(scattered[0], np.array([0., -3., 0., 2.], dtype=np.float32))
    assert scattered[0].dtype == codomain.dtype
    assert selected.trajectory is base.trajectory
    monkeypatch.setattr(base, "invalidate", lambda: setattr(base, "trajectory", None), raising=False)
    selected.invalidate()
    assert selected.trajectory is None


@pytest.mark.parametrize("component,method", [
    ("load", "predict"), ("load", "jvp"), ("load", "vjp"),
    ("solver", "solve_controls"), ("solver", "tangent_controls"),
    ("solver", "adjoint_controls"),
])
def test_grid_prediction_requires_derivatives_before_snapshot(component, method):
    solver = BoundedSolver(dict(diffusion=.1, velocity=[.2, -.1],
        linear_loss=.01, horizon=1.), interior_points=3, time_steps=2)
    load = IntervalAverageLoad(P1Basis([0., 1.], time_unit="h"), solver.times,
        origin=0., q_reference=1., history_integral=lambda a, b: 0.,
        history_initial=0., source_unit="C*km^2/hour")
    output = ArraySpace((1,), axes=("observation",), coordinates=((0,),), units="ug/m^3")
    observation = GridObservation(np.ones((1, solver.size)), [[0., 0., 1.]],
                                  domain=solver_state_space(solver, state_unit="ug/m^3"), codomain=output)
    solver, load = copy(solver), copy(load)
    setattr(load if component == "load" else solver, method, None)
    solver.snapshot = lambda: pytest.fail("Invalid derivatives must fail before snapshot")
    with pytest.raises(TypeError, match=method):
        GridPrediction(solver, load, observation)
