"""Проверки PDE с заданными коэффициентами, производных и владения данными.

Проверяются прямая PDE и производные по аддитивному входу.
Задача восстановления коэффициентов и её идентифицируемость здесь не рассматриваются."""
import gc
import threading
import weakref

import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.state_maps import forcing_state_map, state_map
from adrkit.backends.cached import CachedSolver
from adrkit.coefficients import ADRUnits, ReactionLaw, ScalarField, VectorField
from adrkit.errors import CapabilityError
from adrkit.spaces import ArraySpace


def settings(**changes):
    result = dict(horizon=.2, diffusion=.13, velocity=(.15, -.1),
                  linear_loss=.04, reaction_gamma=.3, reaction_c_star=.7,
                  source_position=(0., 0.))
    result.update(changes)
    return result


def variable_settings(**changes):
    return settings(diffusion=lambda xy, t: .13+.01*xy[:, 0]+.02*t,
                    velocity=lambda xy, t: np.column_stack((.15+.02*xy[:, 0],
                                                            -.1+.01*xy[:, 1]+.03*t)),
                    linear_loss=lambda xy, t: .04+.01*t,
                    **changes)


def layouts(solver, *, forcing=False):
    coordinates = (tuple(solver.times), tuple(map(tuple, solver.xy)))
    state = ArraySpace((solver.nt+1, solver.size), axes=('time', 'node'),
                       coordinates=coordinates, units='concentration')
    if forcing:
        domain = ArraySpace(state.shape, axes=state.axes, coordinates=coordinates,
                            units='concentration/time')
    else:
        domain = ArraySpace((solver.nt+1,), axes=('time',),
                            coordinates=(coordinates[0],), units='source/time')
    return domain, state


@pytest.mark.parametrize('gamma', [0., .3])
def test_constant_field_equivalence_and_point_load_forcing(gamma):
    model = settings(reaction_gamma=gamma)
    wrapped = dict(model)
    for name in ('diffusion', 'linear_loss', 'reaction_gamma', 'reaction_c_star'):
        value = model[name]
        wrapped[name] = ScalarField(lambda xy, t, value=value: value,
                                    time_dependent=False)
    wrapped['velocity'] = VectorField(lambda xy, t: model['velocity'], time_dependent=False)
    reference_solver = BoundedSolver(model, interior_points=(5, 7), time_steps=9)
    fields = BoundedSolver(wrapped, interior_points=(5, 7), time_steps=9)
    controls = 1+.2*np.sin(reference_solver.times)
    expected, actual = reference_solver.solve_controls(controls), fields.solve_controls(controls)
    np.testing.assert_allclose(actual.states, expected.states, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(fields.operator_at(0).toarray(), reference_solver.operator.toarray(),
                               atol=2e-15, rtol=2e-15)
    forcing = controls[:, None]*fields.load[None, :]
    np.testing.assert_allclose(fields.solve_forcing(forcing).states, actual.states, atol=2e-12)
    h = np.cos(reference_solver.times)
    np.testing.assert_allclose(fields.tangent_controls(actual, h),
                               reference_solver.tangent_controls(expected, h), atol=2e-12)
    z = np.random.default_rng(41).normal(size=actual.states.shape)
    np.testing.assert_allclose(fields.adjoint_controls(actual, z),
                               reference_solver.adjoint_controls(expected, z), atol=2e-12)


def test_scalar_numpy_arrays_preserve_constant_fast_path_and_ownership():
    model = settings()
    wrapped = dict(model)
    for name in ('diffusion', 'linear_loss', 'reaction_gamma', 'reaction_c_star'):
        wrapped[name] = np.asarray(model[name])
    solver = BoundedSolver(wrapped, interior_points=5, time_steps=4)
    reference = BoundedSolver(model, interior_points=5, time_steps=4)
    for name in ('diffusion', 'linear_loss', 'reaction_gamma', 'reaction_c_star'):
        wrapped[name][...] = 99.
    assert not solver._variable_fields
    np.testing.assert_array_equal(solver.solve_controls(np.ones(5)).states,
                                  reference.solve_controls(np.ones(5)).states)


def test_external_rectangle_mutation_cannot_change_geometry_or_solution():
    bounds = np.asarray([-2., 2., -2., 2.])
    solver = BoundedSolver(variable_settings(), interior_points=5, time_steps=4, domain=bounds)
    expected_bounds, expected_xy = solver.bounds.copy(), solver.xy.copy()
    controls = np.ones(5)
    expected = solver.solve_controls(controls).states
    bounds[:] = [-20., 20., -10., 10.]
    np.testing.assert_array_equal(solver.bounds, expected_bounds)
    np.testing.assert_array_equal(solver.xy, expected_xy)
    np.testing.assert_array_equal(solver.solve_controls(controls).states, expected)


def test_time_varying_fields_match_discrete_eigenmode_no_stale_lu():
    """Аналитическая рекуррентная формула для собственного вектора дискретного лапласиана Дирихле."""
    origin = -.4
    model = settings(horizon=.3, source_position=None, time_origin=origin,
                     diffusion=lambda xy, t: .2+.15*t,
                     velocity=(0., 0.), linear_loss=lambda xy, t: .07+.03*t,
                     reaction_gamma=0.)
    solver = BoundedSolver(model, interior_points=(7, 5), time_steps=12,
                           domain=(0., 1., 0., 2.))
    x, y = solver.xy.T
    mode = np.sin(np.pi*x)*np.sin(np.pi*y/2.)
    forcing = (1+solver.times[:, None]+origin)*mode[None, :]
    result = solver.solve_forcing(forcing)
    eigenvalue = (4*np.sin(np.pi/(2*(solver.nx+1)))**2/solver.hx**2
                  +4*np.sin(np.pi/(2*(solver.ny+1)))**2/solver.hy**2)
    amplitude = np.zeros(solver.nt+1)
    for n, relative_time in enumerate(solver.times[1:]):
        time = origin+relative_time
        amplitude[n+1] = ((amplitude[n]+solver.dt*(1+time))
            /(1+solver.dt*((.2+.15*time)*eigenvalue+.07+.03*time)))
    np.testing.assert_allclose(result.states, amplitude[:, None]*mode, atol=2e-13, rtol=2e-12)
    assert result.max_scaled_residual < 1e-11


def test_divergence_diffusion_contains_spatial_gradient_term():
    r"""Проверить вклад пространственного градиента коэффициента диффузии.

    Notes
    -----
    Для :math:`D=1+0.2x` и :math:`u=x^2+y^2` ожидается

    .. math::

        -\nabla\!\cdot(D\nabla u)=-4D-0.4x.
    """
    solver = BoundedSolver(settings(diffusion=lambda xy, t: 1+.2*xy[:, 0],
                           velocity=(0., 0.), linear_loss=0., reaction_gamma=0.),
                           interior_points=7, time_steps=2)
    x, y = solver.xy.T
    polynomial = x*x+y*y
    actual = (solver.operator_at(0) @ polynomial).reshape(solver.ny, solver.nx)
    expected = (-4*(1+.2*x)-.4*x).reshape(solver.ny, solver.nx)
    np.testing.assert_allclose(actual[1:-1, 1:-1], expected[1:-1, 1:-1], atol=3e-14)


def test_velocity_is_advective_not_conservative():
    r"""Различить адвективную и консервативную формы переноса.

    Notes
    -----
    Для :math:`v=(x,2y)` постоянное состояние различает две формы:

    .. math::

        v\cdot\nabla 1=0,\qquad \nabla\!\cdot(v\,1)=3.
    """
    solver = BoundedSolver(settings(diffusion=.1, linear_loss=0., reaction_gamma=0.,
        velocity=lambda xy, t: np.column_stack((xy[:, 0], 2*xy[:, 1]))),
        interior_points=7, time_steps=2)
    result = (solver.operator_at(0) @ np.ones(solver.size)).reshape(solver.ny, solver.nx)
    np.testing.assert_allclose(result[1:-1, 1:-1], 0., atol=1e-14)


def cubic_reaction(coefficient=.05):
    """Создать покомпонентную кубическую реакцию и её производную."""
    return ReactionLaw(lambda u, xy, t: coefficient*u**3,
                       lambda u, xy, t: 3*coefficient*u**2, pure=True)


def manufactured_error(nodes):
    r"""Вычислить дискретную ошибку заданного решения и невязку шагов.

    Parameters
    ----------
    nodes : int
        Число внутренних узлов по каждой пространственной оси.

    Returns
    -------
    error : numpy.floating
        Норма ошибки в последний момент: квадратный корень из суммы
        квадратов узловых ошибок, умноженной на площадь ячейки.
    max_scaled_residual : float
        Наибольшая нормированная невязка решённых временных шагов.

    Notes
    -----
    Правая часть построена по точному решению на единичном квадрате
    с нулевыми начальными и граничными значениями:

    .. math::

        u(t,x,y) = t\sin(\pi x)\sin(\pi y).
    """
    model = settings(horizon=.2, source_position=None,
        diffusion=lambda xy, t: .1+.02*xy[:, 0]+.01*t,
        velocity=lambda xy, t: np.column_stack((.4+.1*xy[:, 0], -.3-.15*xy[:, 1])),
        linear_loss=lambda xy, t: .1+.1*t, reaction=cubic_reaction())
    solver = BoundedSolver(model, interior_points=nodes, time_steps=2*(nodes+1),
                           domain=(0., 1., 0., 1.))
    x, y = solver.xy.T
    t = solver.times[:, None]
    phi = np.sin(np.pi*x)*np.sin(np.pi*y)
    exact = t*phi
    ux = t*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
    uy = t*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    diffusion = .1+.02*x+.01*t
    forcing = (phi+2*np.pi**2*diffusion*exact-.02*ux
               +(.4+.1*x)*ux+(-.3-.15*y)*uy+(.1+.1*t)*exact+.05*exact**3)
    computed = solver.solve_forcing(forcing)
    error = np.sqrt(solver.cell_area*np.sum((computed.states[-1]-exact[-1])**2))
    return error, computed.max_scaled_residual


def test_manufactured_variable_field_solution_converges():
    results = [manufactured_error(nodes) for nodes in (7, 15, 31)]
    errors = np.array([value[0] for value in results])
    # Аппроксимация переноса разностями против потока имеет первый порядок; проверяется уточнение
    # сетки, а не теорема сходимости.
    assert np.all(errors[:-1]/errors[1:] > 1.5), errors
    assert max(value[1] for value in results) < 1e-11


@pytest.mark.parametrize('forcing', [False, True])
def test_variable_field_jvp_vjp_and_fixed_linearization(forcing):
    solver = BoundedSolver(variable_settings(reaction=cubic_reaction(.2)),
                           interior_points=5, time_steps=7)
    domain, codomain = layouts(solver, forcing=forcing)
    caps = (forcing_state_map if forcing else state_map)(solver, domain=domain, codomain=codomain)
    rng = np.random.default_rng(37)
    at = .5+rng.random(domain.shape)
    h, z = rng.normal(size=domain.shape), rng.normal(size=codomain.shape)
    jh = caps.jvp.jvp(at, h)
    hz = caps.vjp.vjp(at, z)
    eps = 2e-5
    numerical = (caps.predict.predict(at+eps*h)-caps.predict.predict(at-eps*h))/(2*eps)
    np.testing.assert_allclose(jh, numerical, atol=3e-9, rtol=2e-7)
    np.testing.assert_allclose(np.vdot(jh, z), np.vdot(h, hz), atol=2e-12, rtol=2e-12)
    linearization = caps.jvp.linear_operator(at)
    caps.predict.predict(at*1.7)
    np.testing.assert_allclose(linearization @ h.ravel(), jh.ravel(), atol=2e-12)
    np.testing.assert_allclose(linearization.rmatvec(z.ravel()), hz.ravel(), atol=2e-12)
    assert np.all(hz[0] == 0)  # Вход в t0 не изменяет фиксированное начальное условие.


def test_fields_are_sampled_once_on_physical_clock_and_frozen():
    settings_outside = {'value': .2}
    calls = []
    def field(points, time):
        calls.append(time)
        points[:] = 99.  # Изменение переданных координат не должно повреждать сетку.
        return settings_outside['value']+time*.01
    solver = BoundedSolver(settings(diffusion=field, time_origin=-.3),
                           interior_points=5, time_steps=4)
    np.testing.assert_allclose(calls, solver.times[1:]-.3)
    assert np.max(np.abs(solver.xy)) < 2
    controls = np.ones(5)
    expected = solver.solve_controls(controls).states
    settings_outside['value'] = 999.
    clone = solver.snapshot()
    np.testing.assert_array_equal(clone.solve_controls(controls).states, expected)
    np.testing.assert_array_equal(solver.solve_controls(controls).states, expected)
    assert len(calls) == 4
    assert all(not frame.flags.writeable for frame in clone._diffusion_frames.values)


def test_evaluate_object_and_stationary_wrapper():
    class Diffusion:
        def evaluate(self, points, time):
            return .2+.01*points[:, 0]
    field = ScalarField(Diffusion(), time_dependent=False)
    solver = BoundedSolver(settings(diffusion=field), interior_points=5, time_steps=4)
    assert len(solver._diffusion_frames.values) == 1
    solver.solve_controls(np.ones(5))


def test_resource_owning_field_is_sampled_and_discarded_not_deepcopied():
    class LockedDiffusion:
        def __init__(self):
            self.lock = threading.Lock()

        def evaluate(self, points, time):
            with self.lock:
                return .2+.01*points[:, 0]

    field = LockedDiffusion()
    reference = weakref.ref(field)
    solver = BoundedSolver(settings(diffusion=field), interior_points=5, time_steps=4)
    del field
    gc.collect()
    assert reference() is None  # Снимок использует сохранённые значения поля.
    controls = np.ones(5)
    expected = solver.solve_controls(controls).states
    np.testing.assert_array_equal(solver.snapshot().solve_controls(controls).states, expected)


def test_pure_reaction_with_resource_is_not_deepcopied():
    class LockedReaction:
        def __init__(self):
            self.lock = threading.Lock()

        def value(self, state, points, time):
            with self.lock:
                return .2*state**3

        def derivative(self, state, points, time):
            with self.lock:
                return .6*state**2

    component = LockedReaction()
    law = ReactionLaw(component.value, component.derivative, pure=True)
    solver = BoundedSolver(settings(reaction=law), interior_points=5, time_steps=4)
    controls = np.ones(5)
    expected = solver.solve_controls(controls).states
    np.testing.assert_array_equal(solver.snapshot().solve_controls(controls).states, expected)


@pytest.mark.parametrize('forcing', [False, True])
def test_state_map_rejects_reversed_or_unknown_spatial_coordinates(forcing):
    solver = BoundedSolver(variable_settings(), interior_points=5, time_steps=4)
    domain, codomain = layouts(solver, forcing=forcing)
    factory = forcing_state_map if forcing else state_map
    for nodes in (codomain.coordinates[1][::-1], tuple(f'node-{i}' for i in range(solver.size))):
        wrong_output = ArraySpace(codomain.shape, axes=codomain.axes,
            coordinates=(codomain.coordinates[0], nodes), units=codomain.units)
        wrong_input = (ArraySpace(domain.shape, axes=domain.axes,
            coordinates=wrong_output.coordinates, units=domain.units) if forcing else domain)
        with pytest.raises(ValueError, match='node coordinates'):
            factory(solver, domain=wrong_input, codomain=wrong_output)
    coordinate_output = ArraySpace(codomain.shape, axes=codomain.axes,
        coordinates=(codomain.coordinates[0], tuple(range(solver.size))), units=codomain.units)
    coordinate_input = (ArraySpace(domain.shape, axes=domain.axes,
        coordinates=coordinate_output.coordinates, units=domain.units) if forcing else domain)
    factory(solver, domain=coordinate_input, codomain=coordinate_output)


def test_reaction_snapshot_factory_copies_mutable_configuration():
    outside = {'coefficient': .2}
    def factory():
        return cubic_reaction(outside['coefficient'])
    law = ReactionLaw(lambda u, xy, t: outside['coefficient']*u**3,
                      lambda u, xy, t: 3*outside['coefficient']*u**2,
                      pure=True, snapshot_factory=factory)
    solver = BoundedSolver(settings(reaction=law), interior_points=5, time_steps=4)
    controls = np.ones(5)
    expected = solver.solve_controls(controls).states
    outside['coefficient'] = 300.
    np.testing.assert_array_equal(solver.snapshot().solve_controls(controls).states, expected)


def test_pure_reaction_contract_and_required_derivative():
    with pytest.raises(ValueError, match='pure=True'):
        ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1., pure=False)
    with pytest.raises(TypeError, match='derivative'):
        ReactionLaw(lambda u, xy, t: u, None, pure=True)
    law = ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1., pure=True,
                      snapshot_factory=lambda: 'not a law')
    with pytest.raises(TypeError, match='self-contained'):
        BoundedSolver(settings(reaction=law), interior_points=5, time_steps=4)


@pytest.mark.parametrize('name,field,message', [
    ('diffusion', lambda xy, t: 0., 'strictly positive'),
    ('diffusion', lambda xy, t: -.1, 'strictly positive'),
    ('linear_loss', lambda xy, t: -.1, 'nonnegative'),
    ('reaction_gamma', lambda xy, t: -.1, 'nonnegative'),
    ('reaction_c_star', lambda xy, t: 0., 'strictly positive'),
    ('velocity', lambda xy, t: (np.inf, 0.), 'finite'),
    ('diffusion', lambda xy, t: np.nan, 'finite'),
    ('diffusion', lambda xy, t: np.ones((len(xy), 1)), 'shape'),
    ('diffusion', lambda xy, t: 1+0j, 'real numeric'),
    ('diffusion', lambda xy, t: True, 'real numeric'),
])
def test_invalid_sampled_fields_fail_before_solving(name, field, message):
    with pytest.raises(ValueError, match=message):
        BoundedSolver(settings(**{name: field}), interior_points=5, time_steps=4)


def test_units_and_component_declarations_are_checked():
    units = ADRUnits('m', 's', 'kg/m^3')
    good = ScalarField(lambda xy, t: .2, unit='m^2/s')
    BoundedSolver(settings(diffusion=good, units=units), interior_points=5, time_steps=4)
    with pytest.raises(ValueError, match='unit'):
        BoundedSolver(settings(diffusion=ScalarField(lambda xy, t: .2, unit='km^2/h'), units=units))
    with pytest.raises(TypeError, match='vector field'):
        BoundedSolver(settings(velocity=ScalarField(lambda xy, t: (1., 0.))))
    with pytest.raises(TypeError, match='scalar field'):
        BoundedSolver(settings(diffusion=VectorField(lambda xy, t: (1., 0.))))
    bad_reaction = ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1.,
                               pure=True, unit='kg', derivative_unit='1/s')
    with pytest.raises(ValueError, match='Reaction unit'):
        BoundedSolver(settings(reaction=bad_reaction, units=units))


def test_state_map_checks_declared_point_source_dimensions():
    units = ADRUnits('m', 's', 'kg/m^3')
    solver = BoundedSolver(settings(units=units), interior_points=5, time_steps=4)
    coordinates = (tuple(solver.times), tuple(map(tuple, solver.xy)))
    codomain = ArraySpace((5, solver.size), axes=('time', 'node'),
                          coordinates=coordinates, units=units.concentration)
    for unit in ('K', units.source):
        domain = ArraySpace((5,), axes=('time',), coordinates=(coordinates[0],), units=unit)
        if unit == 'K':
            with pytest.raises(ValueError, match='Point-source control unit'):
                state_map(solver, domain=domain, codomain=codomain)
        else:
            state_map(solver, domain=domain, codomain=codomain)


@pytest.mark.parametrize('setting,value', [('advection_form', 'conservative'),
                                         ('diffusion_form', 'nondivergence')])
def test_unsupported_pde_forms_fail_explicitly(setting, value):
    with pytest.raises(CapabilityError, match='div'):
        BoundedSolver(settings(**{setting: value}))


def test_cached_backend_rejects_variable_fields_and_custom_reaction():
    for changes in (dict(diffusion=lambda xy, t: .2), dict(reaction=cubic_reaction())):
        with pytest.raises(CapabilityError, match='constant coefficients'):
            CachedSolver(settings(**changes), interior_points=5, time_steps=4)


def test_cached_rejects_before_invoking_unsupported_callback():
    def forbidden(points, time):
        raise AssertionError('capability check must precede coefficient evaluation')
    with pytest.raises(CapabilityError, match='constant coefficients'):
        CachedSolver(settings(diffusion=forbidden))


def test_reaction_output_shape_and_reality_checked():
    law = ReactionLaw(lambda u, xy, t: np.ones((len(u), 1)),
                      lambda u, xy, t: 1., pure=True)
    solver = BoundedSolver(settings(reaction=law), interior_points=5, time_steps=4)
    with pytest.raises(ValueError, match='shape'):
        solver.solve_controls(np.ones(5))
    law = ReactionLaw(lambda u, xy, t: u, lambda u, xy, t: 1+0j, pure=True)
    solver = BoundedSolver(settings(reaction=law), interior_points=5, time_steps=4)
    with pytest.raises(ValueError, match='real numeric'):
        solver.solve_controls(np.ones(5))


def test_distributed_forcing_does_not_require_a_point_source():
    solver = BoundedSolver(variable_settings(source_position=None), interior_points=5, time_steps=4)
    forcing = np.ones((solver.nt+1, solver.size))
    forward = solver.solve_forcing(forcing)
    assert forward.max_scaled_residual < 1e-11
    with pytest.raises(CapabilityError, match='No point source'):
        solver.solve_controls(np.ones(5))
    with pytest.raises(CapabilityError, match='No point source'):
        solver.tangent_controls(forward, np.ones(5))
    with pytest.raises(ValueError, match='real'):
        solver.solve_forcing(forcing.astype(complex))


def test_cached_distributed_forcing_matches_newton():
    model = settings(source_position=None)
    newton = BoundedSolver(model, interior_points=5, time_steps=4)
    cached = CachedSolver(model, interior_points=5, time_steps=4)
    forcing = np.random.default_rng(7).random((newton.nt+1, newton.size))
    a, b = newton.solve_forcing(forcing), cached.solve_forcing(forcing)
    np.testing.assert_allclose(a.states, b.states, atol=5e-11, rtol=5e-11)
    h = np.ones_like(forcing)
    np.testing.assert_allclose(newton.tangent_forcing(a, h), cached.tangent_forcing(b, h),
                               atol=5e-11, rtol=5e-11)
