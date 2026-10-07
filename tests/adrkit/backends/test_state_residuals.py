"""Полные якобианы нелинейной невязки состояния для аддитивного входа ADR."""
from dataclasses import replace
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from scipy import sparse

from adrkit.backends.bounded import BoundedSolver
from adrkit.backends.cached import CachedSolver
from adrkit.coefficients import ADRUnits, ReactionLaw, ScalarField
from adrkit.errors import CapabilityError
from adrkit.pde import ADRProblem
from adrkit.spaces import ArraySpace
from adrkit.state_maps import forcing_state_map, state_map
from adrkit.state_residuals import StateResidual


def settings(**changes):
    result = dict(horizon=.2, diffusion=.13, velocity=(.15, -.1),
                  linear_loss=.04, reaction_gamma=0., reaction_c_star=.7,
                  source_position=(0., 0.))
    result.update(changes)
    return result


def nonlinear_coefficient(beta=1.5, *, representation="dense", unit=None, derivative_unit=None):
    r"""Создать нелокальный член невязки с полным якобианом.

    Parameters
    ----------
    beta : float, default: 1.5
        Множитель нелинейного члена.
    representation : {'dense', 'csr', 'array'}, default: 'dense'
        Представление якобиана: плотный массив, CSR-матрица или CSC-массив.
    unit : str or None, optional
        Единица значения невязки.
    derivative_unit : str or None, optional
        Единица производной по состоянию.

    Returns
    -------
    StateResidual
        Добавочный член для девяти координат состояния.

    Notes
    -----
    Для :math:`w_i=(i+1)/45`, :math:`i=0,\ldots,8`, и несимметричной
    трёхдиагональной матрицы :math:`K` вычисляются

    .. math::

        m = w^\mathsf{T}c,\qquad N(c)=\beta m^2 Kc,
        \qquad DN(c)=\beta\left(m^2K+2m(Kc)w^\mathsf{T}\right).

    Главная, верхняя и нижняя диагонали :math:`K` равны 4, −1 и −0.3.
    """
    weights = np.arange(1., 10.)/45.
    matrix = np.diag(np.full(9, 4.))
    matrix += np.diag(np.full(8, -1.), 1)+np.diag(np.full(8, -.3), -1)
    weights.flags.writeable = matrix.flags.writeable = False

    def value(state, points, time):
        moment = weights @ state
        return beta*moment**2*(matrix @ state)

    def jacobian(state, points, time):
        moment = weights @ state
        result = beta*(moment**2*matrix+2*moment*np.outer(matrix @ state, weights))
        if representation == "csr":
            return sparse.csr_matrix(result)
        if representation == "array":
            return sparse.csc_array(result)
        return result

    return StateResidual(value, jacobian, pure=True, unit=unit, derivative_unit=derivative_unit)


def layouts(solver, *, forcing):
    coordinates = (solver.times, tuple(map(tuple, solver.xy)))
    states = ArraySpace((solver.nt+1, solver.size), axes=("time", "node"),
                        coordinates=coordinates, units="concentration")
    domain = (ArraySpace(states.shape, axes=states.axes, coordinates=coordinates,
                          units="concentration/time") if forcing else
              ArraySpace((solver.nt+1,), axes=("time",), coordinates=(solver.times,),
                         units="source/time"))
    return domain, states


@pytest.mark.parametrize("representation", ["dense", "csr", "array"])
def test_full_nonsymmetric_state_jacobian_matches_residual_derivative(representation):
    law = nonlinear_coefficient(representation=representation)
    solver = BoundedSolver(settings(state_residual=law), interior_points=3, time_steps=4,
                           half_width=1.)
    assert solver.linear_lu is None  # При gamma=0 член N(c) также должен вычисляться.
    at, h = np.linspace(.2, 1.1, 9), np.sin(np.arange(9)+.1)
    step = 2
    jacobian = solver.jacobian(at, time_index=step)
    eps = 1e-6
    derivative = (solver.residual_forcing(at+eps*h, np.zeros(9), np.ones(9), step)
                  -solver.residual_forcing(at-eps*h, np.zeros(9), np.ones(9), step))/(2*eps)
    np.testing.assert_allclose(jacobian @ h, derivative, rtol=3e-9, atol=5e-10)
    full = law.evaluate(at, solver.xy, solver.times[step+1], derivative=True).toarray()
    assert np.linalg.norm(full-np.diag(np.diag(full))) > .1
    assert np.linalg.norm(full-full.T) > .1
    np.testing.assert_allclose(jacobian.toarray(), solver._base_at(step).toarray()+solver.dt*full,
                               rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("forcing", [False, True])
@pytest.mark.parametrize("representation", ["dense", "csr"])
def test_nonlinear_nonlocal_forward_jvp_vjp_and_fixed_operator(forcing, representation):
    solver = BoundedSolver(settings(state_residual=nonlinear_coefficient(representation=representation)),
                           interior_points=3, time_steps=7, half_width=1.)
    domain, codomain = layouts(solver, forcing=forcing)
    mapping = (forcing_state_map if forcing else state_map)(solver, domain=domain, codomain=codomain)
    rng = np.random.default_rng(813)
    at = 1.+rng.random(domain.shape)
    h, z = rng.normal(size=domain.shape), rng.normal(size=codomain.shape)
    tangent, pullback = mapping.jvp.jvp(at, h), mapping.vjp.vjp(at, z)
    eps = 3e-6
    numerical = (mapping.predict.predict(at+eps*h)-mapping.predict.predict(at-eps*h))/(2*eps)
    np.testing.assert_allclose(tangent, numerical, rtol=4e-7, atol=2e-9)
    np.testing.assert_allclose(np.vdot(tangent, z), np.vdot(h, pullback), rtol=2e-12, atol=2e-12)
    linear = mapping.predict.linear_operator(at)
    mapping.predict.predict(at*1.7)
    np.testing.assert_array_equal(linear.matvec(h.ravel()).reshape(codomain.shape), tangent)
    np.testing.assert_array_equal(linear.rmatvec(z.ravel()).reshape(domain.shape), pullback)
    trajectory = solver.solve_forcing(at) if forcing else solver.solve_controls(at)
    assert trajectory.max_scaled_residual <= 1e-11
    assert np.all(pullback[0] == 0.)


def test_additive_term_composes_with_local_reaction_and_prescribed_fields():
    local = ReactionLaw(lambda c, xy, t: .04*c**3, lambda c, xy, t: .12*c**2, pure=True)
    problem = ADRProblem.from_model(settings(diffusion=ScalarField(
        lambda xy, t: .13+.01*xy[:, 0]**2, time_dependent=False),
        reaction=local, state_residual=nonlinear_coefficient()))
    assert problem.to_model()["state_residual"] is problem.state_residual
    solver = problem.make_solver(BoundedSolver, interior_points=3, time_steps=4, half_width=1.)
    at = np.linspace(.1, .9, 9)
    expected = .04*at**3+problem.state_residual.evaluate(at, solver.xy, solver.times[1])
    np.testing.assert_array_equal(solver.reaction(at, 0), expected)
    matrix = solver._base_at(0)+sparse.diags(solver.dt*.12*at**2)
    matrix += solver.dt*problem.state_residual.evaluate(at, solver.xy, solver.times[1], derivative=True)
    np.testing.assert_allclose(solver.jacobian(at, time_index=0).toarray(), matrix.toarray(),
                               rtol=2e-15, atol=2e-15)


def test_callback_arguments_and_value_jacobian_results_have_independent_ownership():
    state, points = np.ones(3), np.zeros((3, 2))
    vector, matrix = np.arange(3.), sparse.eye(3, format="csc")

    def value(c, xy, t):
        c[:] = 12.
        xy[:] = 13.
        return vector

    def jacobian(c, xy, t):
        c[:] = 14.
        xy[:] = 15.
        return matrix

    law = StateResidual(value, jacobian, pure=True)
    actual_vector = law.evaluate(state, points, .5)
    actual_matrix = law.evaluate(state, points, .5, derivative=True)
    np.testing.assert_array_equal(state, np.ones(3))
    np.testing.assert_array_equal(points, np.zeros((3, 2)))
    actual_vector[:] = 99.
    actual_matrix.data[:] = 100.
    np.testing.assert_array_equal(vector, np.arange(3.))
    np.testing.assert_array_equal(matrix.toarray(), np.eye(3))


def test_user_snapshot_freezes_numeric_setup_and_retains_nondeepcopyable_resources():
    outside = {"beta": 1.5}
    lock = threading.Lock()
    calls = []

    def snapshot():
        calls.append(outside["beta"])
        fixed = nonlinear_coefficient(outside["beta"])

        class LockedValue:
            def __init__(self):
                self.lock = lock  # Вызов deepcopy для этого функционального объекта завершается
                                  # ошибкой.

            def __call__(self, c, xy, t):
                with self.lock:
                    return fixed.value(c, xy, t)

        return StateResidual(LockedValue(), fixed.jacobian, pure=True)

    law = StateResidual(lambda c, xy, t: nonlinear_coefficient(outside["beta"]).value(c, xy, t),
        lambda c, xy, t: nonlinear_coefficient(outside["beta"]).jacobian(c, xy, t),
        pure=True, snapshot_factory=snapshot)
    problem = ADRProblem.from_model(settings(state_residual=law))
    assert not calls
    solver = BoundedSolver(problem, interior_points=3, time_steps=4, half_width=1.)
    assert calls == [1.5]
    domain, codomain = layouts(solver, forcing=False)
    mapping = state_map(solver, domain=domain, codomain=codomain)
    # Снимок решателя не должен вызывать deepcopy ресурса добавочного члена невязки.
    expected = mapping.predict.predict(np.ones(5))
    outside["beta"] = 20.
    np.testing.assert_array_equal(mapping.predict.predict(np.ones(5)*1.1),
                                  solver.solve_controls(np.ones(5)*1.1).states)
    np.testing.assert_array_equal(mapping.predict.predict(np.ones(5)), expected)


def test_cached_rejects_full_law_before_value_jacobian_or_snapshot():
    def forbidden(*args):
        raise AssertionError("CachedSolver must reject unsupported law before execution")

    law = StateResidual(forbidden, forbidden, pure=True, snapshot_factory=forbidden)
    with pytest.raises(CapabilityError, match="StateResidual"):
        CachedSolver(settings(state_residual=law), interior_points=3, time_steps=4)


@pytest.mark.parametrize("backend", [BoundedSolver, CachedSolver])
def test_constant_transport_without_state_residual_preserves_solution(backend):
    spec = json.loads(((Path(__file__).resolve().parent.parent / "fixtures/constant_transport.json")).read_text(encoding="utf-8"))
    names = ("diffusion", "velocity", "linear_loss", "reaction_c_star",
             "source_position", "background", "horizon", "reaction_gamma")
    model = {name: spec[name] for name in names}
    reference_solver = backend(model, interior_points=3, time_steps=4)
    explicit = backend(ADRProblem.from_model(model), interior_points=3, time_steps=4)
    assert explicit._state_residual_law is None and not explicit._variable_fields
    controls = np.linspace(.1, 1., 5)
    expected, actual = reference_solver.solve_controls(controls), explicit.solve_controls(controls)
    np.testing.assert_array_equal(explicit.base_matrix.toarray(), reference_solver.base_matrix.toarray())
    np.testing.assert_array_equal(actual.states, expected.states)
    assert actual.nonlinear_iterations == expected.nonlinear_iterations
    assert actual.max_scaled_residual == expected.max_scaled_residual


@pytest.mark.parametrize("bad", [1., np.ones(2), np.ones((3, 1)),
                                  np.array([0., np.nan, 0.]), np.ones(3)*1j, np.ones(3, dtype=bool)])
def test_invalid_value_shape_or_numeric_data_are_rejected(bad):
    law = StateResidual(lambda c, xy, t: bad, lambda c, xy, t: np.eye(3), pure=True)
    with pytest.raises(ValueError, match="value"):
        law.evaluate(np.ones(3), np.zeros((3, 2)), 0.)


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((2, 2)), np.full((3, 3), np.inf),
    np.eye(3, dtype=complex), np.eye(3, dtype=bool), sparse.eye(2),
    sparse.eye(3, dtype=complex), sparse.eye(3, dtype=bool),
    sparse.csr_matrix(np.full((3, 3), np.nan))])
def test_invalid_full_jacobian_is_rejected(bad):
    law = StateResidual(lambda c, xy, t: c, lambda c, xy, t: bad, pure=True)
    with pytest.raises(ValueError, match="Jacobian"):
        law.evaluate(np.ones(3), np.zeros((3, 2)), 0., derivative=True)


@pytest.mark.parametrize("state,points,time", [
    (np.ones(2), np.zeros((3, 2)), 0.), (np.ones(3), np.zeros((3, 3)), 0.),
    (np.ones(3)*1j, np.zeros((3, 2)), 0.), (np.ones(3), np.zeros((3, 2)), np.inf),
    (np.ones(3), np.zeros((3, 2)), True),
])
def test_invalid_arguments_fail_before_calling_law(state, points, time):
    def forbidden(*args):
        raise AssertionError("arguments must be validated before callbacks")

    law = StateResidual(forbidden, forbidden, pure=True)
    with pytest.raises(ValueError):
        law.evaluate(state, points, time)


def test_explicit_callbacks_purity_and_snapshot_result_are_required():
    value, jacobian = lambda c, xy, t: c, lambda c, xy, t: np.eye(len(c))
    with pytest.raises(TypeError, match="Jacobian"):
        StateResidual(value, None, pure=True)
    with pytest.raises(ValueError, match="pure"):
        StateResidual(value, jacobian, pure=False)
    law = StateResidual(value, jacobian, pure=True, snapshot_factory=lambda: object())
    with pytest.raises(TypeError, match="self-contained"):
        BoundedSolver(settings(state_residual=law), interior_points=3, time_steps=4)
    with pytest.raises(TypeError, match="StateResidual"):
        ADRProblem.from_model(settings(state_residual=value))
    with pytest.raises(TypeError, match="StateResidual"):
        BoundedSolver(settings(state_residual=value), interior_points=3, time_steps=4)


@pytest.mark.parametrize("labels", [{"unit": "wrong"}, {"derivative_unit": "wrong"}])
def test_state_residual_units_are_checked_for_problem_and_bound_snapshot(labels):
    units = ADRUnits("km", "h", "ug/m^3")
    law = nonlinear_coefficient(**labels)
    with pytest.raises(ValueError, match="StateResidual unit"):
        ADRProblem.from_model(settings(units=units, state_residual=law))
    with pytest.raises(ValueError, match="StateResidual unit"):
        BoundedSolver(settings(units=units, state_residual=law), interior_points=3, time_steps=4)
    matching = nonlinear_coefficient(unit=units.reaction, derivative_unit=units.loss)
    declared = replace(matching, snapshot_factory=lambda: law)
    problem = ADRProblem.from_model(settings(units=units, state_residual=declared))
    with pytest.raises(ValueError, match="StateResidual unit"):
        BoundedSolver(problem, interior_points=3, time_steps=4)
