"""Построение данных ADR, их привязка и сохранение численных значений по умолчанию."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.backends.cached import CachedSolver
from adrkit.coefficients import ADRUnits, ReactionLaw, ScalarField, VectorField
from adrkit.errors import CapabilityError
from adrkit.pde import ADRProblem


def model(**changes):
    result = dict(horizon=.2, diffusion=.13, velocity=(.15, -.1),
                  linear_loss=.04, reaction_gamma=.3, reaction_c_star=.7,
                  source_position=(0., 0.))
    result.update(changes)
    return result


@pytest.mark.parametrize("backend", [BoundedSolver, CachedSolver])
@pytest.mark.parametrize("gamma", [0., .3])
def test_numeric_roundtrip_preserves_constant_matrix_state_and_derivatives(backend, gamma):
    data = model(reaction_gamma=gamma)
    problem = ADRProblem.from_model(data)
    direct = backend(data, interior_points=5, time_steps=9)
    adapted = problem.make_solver(backend, interior_points=5, time_steps=9)
    assert not adapted._variable_fields
    np.testing.assert_array_equal(adapted.base_matrix.toarray(), direct.base_matrix.toarray())
    controls = 1+.2*np.sin(direct.times)
    expected, actual = direct.solve_controls(controls), adapted.solve_controls(controls)
    np.testing.assert_array_equal(actual.states, expected.states)
    assert actual.nonlinear_iterations == expected.nonlinear_iterations
    assert actual.max_scaled_residual == expected.max_scaled_residual
    direction = np.cos(direct.times)
    np.testing.assert_array_equal(adapted.tangent_controls(actual, direction),
                                  direct.tangent_controls(expected, direction))
    z = np.random.default_rng(62).normal(size=expected.states.shape)
    np.testing.assert_array_equal(adapted.adjoint_controls(actual, z),
                                  direct.adjoint_controls(expected, z))


def test_constant_transport_model_roundtrip_preserves_solution():
    path = (Path(__file__).resolve().parent / "fixtures/constant_transport.json")
    spec = json.loads(path.read_text(encoding="utf-8"))
    names = ("diffusion", "velocity", "linear_loss", "reaction_c_star",
             "source_position", "background", "horizon", "reaction_gamma")
    data = {name: spec[name] for name in names}
    adapted = ADRProblem.from_model(data).to_model()
    for name in names:
        np.testing.assert_array_equal(adapted[name], data[name])
    original = CachedSolver(data, interior_points=5, time_steps=9)
    constructed = CachedSolver(adapted, interior_points=5, time_steps=9)
    controls = np.linspace(0., 1., 10)
    np.testing.assert_array_equal(constructed.solve_controls(controls).states,
                                  original.solve_controls(controls).states)


def test_immutable_numeric_ownership_and_fresh_model_dicts():
    velocity = np.array([.15, -.1])
    source = [0., 0.]
    diffusion = np.asarray(.13)
    problem = ADRProblem.from_model(model(velocity=velocity, source_position=source,
                                         diffusion=diffusion))
    velocity[:] = 300.
    source[:] = [100., 100.]
    diffusion[...] = 99.
    assert problem.velocity == (.15, -.1)
    assert problem.source_position == (0., 0.)
    assert problem.diffusion == .13
    with pytest.raises(FrozenInstanceError):
        problem.diffusion = .5
    first = problem.to_model()
    first["velocity"] = (9., 9.)
    assert problem.to_model()["velocity"] == (.15, -.1)


def test_fields_are_declarations_until_a_solver_samples_them():
    calls = []

    class Diffusion:
        def evaluate(self, xy, t):
            calls.append((xy.copy(), t))
            return .13+.01*xy[:, 0]**2

    units = ADRUnits("km", "h", "ug/m^3")
    diffusion = ScalarField(Diffusion(), unit=units.diffusion, time_dependent=False)
    velocity = VectorField(lambda xy, t: (.15+.02*t, -.1), unit=units.velocity)
    loss = lambda xy, t: .04+.01*t
    problem = ADRProblem.from_model(model(diffusion=diffusion, velocity=velocity,
        linear_loss=loss, units=units, time_origin=10.))
    assert not calls
    exported = problem.to_model()
    assert exported["diffusion"] is diffusion
    assert exported["velocity"] is velocity
    assert exported["linear_loss"] is loss
    assert not calls
    solver = problem.make_solver(BoundedSolver, interior_points=5, time_steps=4)
    assert len(calls) == 1  # Поле стационарно; объявление не должно вызывать его вычисление.
    assert calls[0][1] == pytest.approx(10.05)
    assert solver.solve_controls(np.ones(5)).max_scaled_residual <= 1e-11


def test_reaction_snapshot_is_bound_by_solver_not_by_problem():
    outside = {"coefficient": .2}
    snapshot_calls = []

    def snapshot():
        snapshot_calls.append(outside["coefficient"])
        coefficient = outside["coefficient"]
        return ReactionLaw(lambda u, xy, t: coefficient*u**3,
                           lambda u, xy, t: 3*coefficient*u**2, pure=True)

    reaction = ReactionLaw(lambda u, xy, t: outside["coefficient"]*u**3,
        lambda u, xy, t: 3*outside["coefficient"]*u**2,
        pure=True, snapshot_factory=snapshot)
    problem = ADRProblem.from_model(model(reaction=reaction))
    assert not snapshot_calls
    assert problem.to_model()["reaction"] is reaction
    assert not snapshot_calls
    solver = problem.make_solver(BoundedSolver, interior_points=5, time_steps=4)
    assert snapshot_calls == [.2]
    controls = np.ones(5)
    expected = solver.solve_controls(controls)
    outside["coefficient"] = 100.
    actual = solver.solve_controls(controls)
    np.testing.assert_array_equal(actual.states, expected.states)


def test_replace_local_reaction_keeps_the_existing_additive_forward_problem():
    base = ADRProblem.from_model(model())
    reaction = ReactionLaw(lambda u, xy, t: .2*u**3,
                          lambda u, xy, t: .6*u**2, pure=True)
    changed = replace(base, reaction=reaction)
    solver = changed.make_solver(BoundedSolver, interior_points=5, time_steps=6)
    reference = BoundedSolver(model(reaction=reaction), interior_points=5, time_steps=6)
    controls = np.linspace(.1, .3, 7)
    np.testing.assert_array_equal(solver.solve_controls(controls).states,
                                  reference.solve_controls(controls).states)
    assert base.reaction is None


@pytest.mark.parametrize("name,value", [
    ("diffusion", 0.), ("diffusion", -.1), ("diffusion", np.inf),
    ("diffusion", True), ("diffusion", 1j), ("diffusion", ".13"),
    ("diffusion", [.13]), ("diffusion", np.array(.13, dtype=object)),
    ("linear_loss", -.1), ("reaction_gamma", -.1), ("reaction_c_star", 0.),
    ("velocity", (1.,)), ("velocity", (1., 2., 3.)),
    ("velocity", np.array([1., np.nan])), ("velocity", (1j, 0.)),
    ("velocity", (True, .1)), ("source_position", [False, 0.]),
    ("velocity", "(0, 1)"), ("source_position", (1.,)),
    ("horizon", 0.), ("horizon", True), ("horizon", "1.0"),
    ("time_origin", np.inf), ("initial_state", np.ones(2)),
])
def test_invalid_numeric_data_fail_before_any_solve(name, value):
    with pytest.raises((TypeError, ValueError)):
        ADRProblem.from_model(model(**{name: value}))


@pytest.mark.parametrize("name,value", [
    ("diffusion_form", "laplacian"), ("advection_form", "conservative"),
    ("initial_state", 1.), ("background", 1.),
])
def test_unsupported_equation_forms_and_known_data_fail_explicitly(name, value):
    with pytest.raises(CapabilityError):
        ADRProblem.from_model(model(**{name: value}))


@pytest.mark.parametrize("changes", [
    {"diffusion": VectorField(lambda xy, t: (1., 1.))},
    {"velocity": ScalarField(lambda xy, t: 1.)},
    {"reaction": lambda u, xy, t: u},
    {"units": "km/h"},
])
def test_component_type_errors_fail_without_evaluating_callbacks(changes):
    with pytest.raises(TypeError):
        ADRProblem.from_model(model(**changes))


@pytest.mark.parametrize("name,field", [
    ("diffusion", ScalarField(lambda xy, t: .1, unit="m^2/s")),
    ("velocity", VectorField(lambda xy, t: (0., 0.), unit="m/s")),
    ("linear_loss", ScalarField(lambda xy, t: .1, unit="1/s")),
    ("reaction_gamma", ScalarField(lambda xy, t: .1, unit="1/s")),
    ("reaction_c_star", ScalarField(lambda xy, t: 1., unit="ppm")),
])
def test_declared_field_units_are_checked_before_binding(name, field):
    with pytest.raises(ValueError, match="unit"):
        ADRProblem.from_model(model(units=ADRUnits("km", "h", "ug/m^3"), **{name: field}))


@pytest.mark.parametrize("labels", [
    {"unit": "ppm/s"}, {"derivative_unit": "1/s"},
])
def test_declared_reaction_units_are_checked_before_binding(labels):
    law = ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1., pure=True, **labels)
    with pytest.raises(ValueError, match="unit"):
        ADRProblem.from_model(model(units=ADRUnits("km", "h", "ug/m^3"), reaction=law))


def test_reaction_requires_explicit_purity_and_state_derivative():
    with pytest.raises(ValueError, match="pure"):
        ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1., pure=False)
    with pytest.raises(TypeError, match="derivative"):
        ReactionLaw(lambda u, xy, t: u, None, pure=True)


def test_reaction_snapshot_factory_must_return_an_independent_pure_law():
    calls = []

    def invalid_snapshot():
        calls.append(True)
        return object()

    law = ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1.,
                      pure=True, snapshot_factory=invalid_snapshot)
    problem = ADRProblem.from_model(model(reaction=law))
    assert not calls
    with pytest.raises(TypeError, match="self-contained pure ReactionLaw"):
        problem.make_solver(BoundedSolver, interior_points=5, time_steps=4)
    assert calls == [True]


def test_reaction_units_are_rechecked_after_user_snapshot_binding():
    units = ADRUnits("km", "h", "ug/m^3")

    def mismatched_snapshot():
        return ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1.,
                           pure=True, unit="ppm/s", derivative_unit=units.loss)

    law = ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1., pure=True,
        snapshot_factory=mismatched_snapshot, unit=units.reaction, derivative_unit=units.loss)
    problem = ADRProblem.from_model(model(reaction=law, units=units))
    with pytest.raises(ValueError, match="Reaction unit"):
        problem.make_solver(BoundedSolver, interior_points=5, time_steps=4)


def test_solver_factory_is_explicit_and_receives_separate_numerical_options():
    problem = ADRProblem.from_model(model(source_position=None))
    received = []
    sentinel = object()

    def factory(data, **options):
        received.append((data, options))
        return sentinel

    assert problem.make_solver(factory, interior_points=(5, 7), time_steps=9) is sentinel
    assert received == [(problem.to_model(), {"interior_points": (5, 7), "time_steps": 9})]
    assert "interior_points" not in problem.to_model()
    with pytest.raises(TypeError, match="factory"):
        problem.make_solver(None)


def test_from_model_rejects_unknown_keys_instead_of_ignoring_a_typo():
    with pytest.raises(ValueError, match="Unknown ADR model keys"):
        ADRProblem.from_model(model(diffuson=.5))
    with pytest.raises(TypeError, match="mapping"):
        ADRProblem.from_model("diffusion=.1")


def test_cached_factory_rejects_custom_reaction_before_value_callback():
    def forbidden(u, xy, t):
        raise AssertionError("unsupported reaction must not be evaluated")

    problem = ADRProblem.from_model(model(reaction=ReactionLaw(forbidden, forbidden, pure=True)))
    with pytest.raises(CapabilityError, match="constant coefficients"):
        problem.make_solver(CachedSolver, interior_points=5, time_steps=4)
