"""Second initial-state derivatives of the fixed discrete Strang composition."""

from math import exp, expm1

import numpy as np
import pytest
from scipy import sparse

from adrkit.time_integration import PositiveAffineAction, solve_strang


@pytest.fixture(scope="module")
def cupy():
    module = pytest.importorskip("cupy")
    try:
        count = module.cuda.runtime.getDeviceCount()
    except module.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    if not count:
        pytest.skip("CUDA device unavailable")
    return module


def _generator():
    return sparse.csr_matrix([[-2, 2, 0], [1, -4, 3], [0, 2, -2]], dtype=float)


def _path(*, distributed=True, backend="numpy", beta=0.3, initial=None):
    action = PositiveAffineAction(_generator(), -0.2, step=0.1, rate=5, degree=14, backend=backend)
    forcing = np.full((6, 3), 0.2) if distributed else np.linspace(0.1, 0.4, 6)
    load = None if distributed else np.array([0.5, 0.0, 0.2])
    initial = [0.3, 0.5, 0.4] if initial is None else initial
    return solve_strang(action, initial, forcing, beta=beta, load=load)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_scalar_hessian_matches_independent_fractional_linear_closed_form(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    step, alpha, beta, initial, direction = 0.2, -0.2, 0.7, 0.4, -0.6
    forcing = np.array([0.2, 0.6, 0.0, 0.1])
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), alpha, step=step, rate=1, degree=12, backend=backend
    )
    trajectory = solve_strang(action, [initial], forcing[:, None], beta=beta)
    cotangent = np.array([[1.2], [0.7], [-0.4], [0.8], [0.2]])
    # R E R is a Mobius map (a*x+b)/(c*x+d), det=exp(alpha*step).
    # Its second derivative is -2*c*det/(c*x+d)^3. Matrix products give
    # the closed form at every layer independently of trajectory derivatives.
    k, growth = beta * step / 2, exp(alpha * step)
    composition = np.eye(2)
    determinant, expected = 1.0, 0.0
    expected_states = [initial]
    for n, source in enumerate(forcing, start=1):
        load = source * expm1(alpha * step) / alpha
        one_step = np.array(
            [[growth + k * load, load], [k * (growth + 1 + k * load), 1 + k * load]]
        )
        composition = one_step @ composition
        determinant *= growth
        c, d = composition[1]
        second = -2 * c * determinant / (c * initial + d) ** 3
        expected += cotangent[n, 0] * second * direction
        expected_states.append(
            (composition[0, 0] * initial + composition[0, 1]) / (c * initial + d)
        )
    actual = action.to_host(trajectory.initial_hvp(cotangent, [direction]))
    np.testing.assert_allclose(actual, [expected], rtol=3e-12, atol=2e-14)
    np.testing.assert_allclose(
        action.to_host(trajectory.states).ravel(), expected_states, rtol=3e-13, atol=2e-15
    )


@pytest.mark.parametrize("distributed", [True, False])
def test_hessian_action_matches_difference_of_initial_pullbacks_at_positive_state(distributed):
    initial = np.array([0.3, 0.5, 0.4])
    direction = np.array([0.2, -0.1, 0.05])
    path = _path(distributed=distributed, initial=initial)
    cotangent = np.random.default_rng(2934).normal(size=path.states.shape)
    before = path.states.copy()
    result = path.initial_hvp(cotangent, direction)
    increment = 1e-5
    varied = [
        _path(distributed=distributed, initial=initial + sign * increment * direction).vjp(
            cotangent
        )["initial"]
        for sign in (1, -1)
    ]
    np.testing.assert_allclose(
        result, (varied[0] - varied[1]) / (2 * increment), rtol=3e-7, atol=2e-10
    )
    np.testing.assert_array_equal(path.states, before)
    np.testing.assert_array_equal(initial, [0.3, 0.5, 0.4])
    np.testing.assert_array_equal(direction, [0.2, -0.1, 0.05])


def test_mixed_initial_derivatives_are_symmetric_and_linear_in_direction():
    path = _path()
    rng = np.random.default_rng(3821)
    cotangent = rng.normal(size=path.states.shape)
    left, right = rng.normal(size=(2, 3))
    h_left = path.initial_hvp(cotangent, left)
    h_right = path.initial_hvp(cotangent, right)
    np.testing.assert_allclose(left @ h_right, right @ h_left, rtol=3e-12, atol=2e-14)
    np.testing.assert_allclose(
        path.initial_hvp(cotangent, left + right),
        h_left + h_right,
        rtol=3e-12,
        atol=2e-14,
    )


def test_nonnegative_state_cotangent_has_nonpositive_initial_curvature():
    path = _path(distributed=False)
    cotangent = np.linspace(0.1, 1.0, path.states.size).reshape(path.states.shape)
    direction = np.array([0.3, -0.5, 0.2])
    action = path.initial_hvp(cotangent, direction)
    assert direction @ action < 0


def test_no_time_steps_leave_only_a_linear_initial_pairing():
    action = PositiveAffineAction(_generator(), -0.2, step=0.1, rate=5, degree=14)
    path = solve_strang(action, [0.3, 0.5, 0.4], np.empty((0, 3)), beta=0.3)
    result = path.initial_hvp([[1.0, -2.0, 0.5]], [-0.4, 0.2, 0.1])
    np.testing.assert_array_equal(result, np.zeros(3))


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_zero_state_has_the_independent_full_flow_boundary_curvature(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    beta, step, direction, terminal = 0.7, 0.2, -0.4, 1.3
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=step, rate=1, degree=14, backend=backend
    )
    path = solve_strang(action, [0], [[0]], beta=beta)
    # Q=alpha=f=0 gives u(t)=u0/(1+beta*t*u0); at u0=0,
    # d(vjp(w)['initial'])[v] = -2*beta*step*w_end*v.
    expected = -2 * beta * step * terminal * direction
    result = path.initial_hvp([[2.0], [terminal]], [direction])
    np.testing.assert_allclose(action.to_host(result), [expected], rtol=3e-13, atol=2e-15)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("direction, terminal", [(0.0, 1.0), (1.0, 0.0), (1e308, 0.0)])
def test_huge_reaction_curvature_has_exact_zero_weighted_actions(
    request, backend, direction, terminal
):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=2, rate=1, degree=14, backend=backend
    )
    path = solve_strang(action, [0], [[0]], beta=1e308)
    result = path.initial_hvp([[2.0], [terminal]], [direction])
    np.testing.assert_array_equal(action.to_host(result), [0.0])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize(
    "step, direction, terminal, expected",
    [(2.0, 1e-308, 1.0, -4.0), (2.0, 1.0, 1e-308, -4.0), (4.0, 1e-308, 0.25, -2.0)],
)
def test_tiny_weights_rescue_an_unrepresentable_unweighted_curvature(
    request, backend, step, direction, terminal, expected
):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=step, rate=1, degree=14, backend=backend
    )
    path = solve_strang(action, [0], [[0]], beta=1e308)
    result = path.initial_hvp([[0.0], [terminal]], [direction])
    np.testing.assert_allclose(action.to_host(result), [expected], rtol=4e-12, atol=2e-14)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_linear_reaction_and_initial_layer_pairing_have_exact_zero_curvature(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    direction = np.array([-0.3, 0.5, 0.2])
    path = _path(backend=backend, beta=0.0)
    cotangent = np.ones(path.states.shape)
    value = path.initial_hvp(cotangent, direction)
    np.testing.assert_array_equal(path.affine.to_host(value), np.zeros(3))
    nonlinear = _path(backend=backend)
    cotangent[1:] = 0
    value = nonlinear.initial_hvp(cotangent, direction)
    np.testing.assert_array_equal(nonlinear.affine.to_host(value), np.zeros(3))


@pytest.mark.parametrize("distributed", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.3])
def test_device_hessian_matches_cpu_and_stays_on_device(cupy, distributed, beta):
    cpu = _path(distributed=distributed, beta=beta)
    gpu = _path(distributed=distributed, beta=beta, backend="cupy")
    rng = np.random.default_rng(4728)
    cotangent = rng.normal(size=cpu.states.shape)
    direction = rng.normal(size=3)
    expected = cpu.initial_hvp(cotangent, direction)
    actual = gpu.initial_hvp(cupy.asarray(cotangent), cupy.asarray(direction))
    assert isinstance(actual, cupy.ndarray) and actual.dtype == cupy.float64
    np.testing.assert_allclose(cupy.asnumpy(actual), expected, rtol=3e-11, atol=3e-13)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_unrepresentable_curvature_is_reported_instead_of_returned(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=2, rate=1, degree=14, backend=backend
    )
    path = solve_strang(action, [0], [[0]], beta=1e308)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(FloatingPointError):
        path.initial_hvp([[0], [1]], [1])
