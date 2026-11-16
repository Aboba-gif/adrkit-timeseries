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


def _path(*, distributed=True, backend="numpy", beta=0.3, initial=None, forcing=None):
    action = PositiveAffineAction(_generator(), -0.2, step=0.1, rate=5, degree=14, backend=backend)
    if forcing is None:
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


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("distributed", [True, False])
def test_initial_increment_matches_independent_shifted_solve_and_owns_output(
    request, backend, distributed
):
    if backend == "cupy":
        module = request.getfixturevalue("cupy")
    initial = np.array([0.3, 0.5, 0.4])
    direction = np.array([0.05, -0.07, 0.1])
    base = _path(backend=backend, distributed=distributed, initial=initial)
    shifted = _path(backend=backend, distributed=distributed, initial=initial + direction)
    original = base.affine.to_host(base.states)
    actual = base.initial_increment(direction)
    if backend == "cupy":
        assert isinstance(actual, module.ndarray)
    else:
        assert not np.shares_memory(actual, base.states)
    expected = shifted.affine.to_host(shifted.states) - original
    np.testing.assert_allclose(base.affine.to_host(actual), expected, rtol=4e-12, atol=2e-15)
    actual[0, 0] = 7.0
    np.testing.assert_array_equal(base.affine.to_host(base.states), original)
    np.testing.assert_array_equal(direction, [0.05, -0.07, 0.1])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("direction", [2.0**-60, -(2.0**-60)])
def test_sub_ulp_increment_remains_nonzero_when_forward_difference_is_zero(
    request, backend, direction
):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    initial, beta, step = 1.0, 1.0, 1.0
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=step, rate=1, degree=14, backend=backend
    )
    base = solve_strang(action, [initial], [[0]], beta=beta)
    shifted = solve_strang(action, [initial + direction], [[0]], beta=beta)
    np.testing.assert_array_equal(action.to_host(shifted.states), action.to_host(base.states))
    actual = action.to_host(base.initial_increment([direction]))
    expected = direction / (
        (1 + beta * step * initial) * (1 + beta * step * (initial + direction))
    )
    assert actual[1, 0] != 0.0
    np.testing.assert_allclose(actual[:, 0], [direction, expected], rtol=5e-13, atol=0)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_increment_log_denominators_do_not_replace_overflowing_products_by_zero(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=4, rate=1, degree=14, backend=backend
    )
    base = solve_strang(action, [0], [[0]], beta=1e308)
    actual = action.to_host(base.initial_increment([1e-308]))
    # Full reaction flow: delta/(1+beta*h*delta) = 1e-308/5.
    np.testing.assert_allclose(actual[1], [2e-309], rtol=2e-11, atol=0)


def test_initial_increment_has_the_declared_tangent_as_its_small_step_limit():
    path = _path()
    direction = np.array([0.1, -0.2, 0.3])
    tangent = path.jvp(d_initial=direction)
    first = path.initial_increment(1e-4 * direction) / 1e-4
    second = path.initial_increment(1e-5 * direction) / 1e-5
    first_error = np.linalg.norm(first - tangent)
    second_error = np.linalg.norm(second - tangent)
    assert second_error < 0.2 * first_error


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_zero_beta_increment_is_the_fixed_affine_tangent(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    path = _path(beta=0.0, backend=backend)
    direction = np.array([-0.1, 0.2, -0.05])
    actual = path.initial_increment(direction)
    expected = path.jvp(d_initial=direction)
    np.testing.assert_array_equal(path.affine.to_host(actual), path.affine.to_host(expected))


def test_increment_uses_the_actual_truncated_polynomial_plan():
    step, rate = 0.2, 5.0
    generator = _generator()
    action = PositiveAffineAction(generator, -0.2, step=step, rate=rate, degree=2)
    path = solve_strang(action, [0.3, 0.5, 0.4], [[0.1, 0.2, 0.3]], beta=0)
    direction = np.array([-0.1, 0.2, 0.05])
    transition = np.eye(3) + generator.toarray() / rate
    power, expected = np.eye(3), np.zeros(3)
    for weight in action.transport_weights:
        expected += exp(-0.2 * step) * weight * (power @ direction)
        power = transition @ power
    np.testing.assert_allclose(
        path.initial_increment(direction)[1], expected, rtol=5e-14, atol=2e-16
    )


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_feasible_negative_increment_can_reach_an_exact_zero_shifted_state(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    action = PositiveAffineAction(_generator(), -0.2, step=0.1, rate=5, degree=14, backend=backend)
    initial = np.array([0.3, 0.5, 0.4])
    path = solve_strang(action, initial, np.zeros((4, 3)), beta=0.3)
    actual = action.to_host(path.initial_increment(-initial))
    np.testing.assert_allclose(actual, -action.to_host(path.states), rtol=5e-13, atol=2e-15)


def test_no_steps_and_zero_initial_increment_return_independent_arrays():
    action = PositiveAffineAction(_generator(), -0.2, step=0.1, rate=5, degree=14)
    path = solve_strang(action, [0.3, 0.5, 0.4], np.empty((0, 3)), beta=0.3)
    direction = np.array([-0.1, 0.2, 0.0])
    actual = path.initial_increment(direction)
    np.testing.assert_array_equal(actual, direction[None, :])
    assert not np.shares_memory(actual, direction)
    zeros = _path().initial_increment(np.zeros(3))
    np.testing.assert_array_equal(zeros, np.zeros((7, 3)))


def test_infeasible_initial_increment_is_rejected_without_projection():
    path = _path()
    with pytest.raises(ValueError, match="shifted initial state"):
        path.initial_increment([-0.31, 0, 0])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("distributed", [True, False])
def test_joint_hessian_blocks_match_independent_one_step_scalar_formulas(
    request, backend, distributed
):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    initial, intensity, step, beta, terminal = 0.4, 0.3, 0.2, 0.7, 1.3
    load = None if distributed else np.array([2.0])
    load_factor = 1.0 if distributed else 2.0
    forcing = np.array([[intensity]]) if distributed else np.array([intensity])
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=step, rate=1, degree=14, backend=backend
    )
    path = solve_strang(action, [initial], forcing, beta=beta, load=load)
    k = beta * step / 2
    first = initial / (1 + k * initial)
    middle = first + step * load_factor * intensity
    first_slope = (1 + k * initial) ** -2
    first_curvature = -2 * k * (1 + k * initial) ** -3
    middle_slope = (1 + k * middle) ** -2
    middle_curvature = -2 * k * (1 + k * middle) ** -3
    huu = terminal * (middle_curvature * first_slope**2 + middle_slope * first_curvature)
    huf = terminal * middle_curvature * first_slope * step * load_factor
    hff = terminal * middle_curvature * (step * load_factor) ** 2
    cotangent = [[0.6], [terminal]]
    forcing_unit = np.ones_like(forcing)
    initial_only = path.state_forcing_hvp(cotangent, d_initial=[1.0])
    forcing_only = path.state_forcing_hvp(cotangent, d_forcing=forcing_unit)
    both = path.state_forcing_hvp(cotangent, d_initial=[0.2], d_forcing=-0.4 * forcing_unit)
    for result, expected_initial, expected_forcing in (
        (initial_only, huu, huf),
        (forcing_only, huf, hff),
        (both, 0.2 * huu - 0.4 * huf, 0.2 * huf - 0.4 * hff),
    ):
        assert set(result) == {"initial", "forcing"}
        assert result["forcing"].shape == forcing.shape
        np.testing.assert_allclose(
            action.to_host(result["initial"]), [expected_initial], rtol=4e-12, atol=2e-14
        )
        np.testing.assert_allclose(
            action.to_host(result["forcing"]),
            np.full_like(forcing, expected_forcing),
            rtol=4e-12,
            atol=2e-14,
        )


@pytest.mark.parametrize("distributed", [True, False])
@pytest.mark.parametrize("mode", ["initial", "forcing", "both"])
def test_joint_hessian_matches_finite_differences_of_both_pullback_outputs(distributed, mode):
    initial = np.array([0.3, 0.5, 0.4])
    forcing = np.full((6, 3), 0.2) if distributed else np.linspace(0.1, 0.4, 6)
    path = _path(distributed=distributed, initial=initial, forcing=forcing)
    rng = np.random.default_rng(5021)
    cotangent = rng.normal(size=path.states.shape)
    di = rng.normal(scale=0.1, size=3) if mode != "forcing" else np.zeros(3)
    df = rng.normal(scale=0.1, size=forcing.shape) if mode != "initial" else np.zeros_like(forcing)
    result = path.state_forcing_hvp(cotangent, d_initial=di, d_forcing=df)
    epsilon = 1e-5
    varied = [
        _path(
            distributed=distributed,
            initial=initial + sign * epsilon * di,
            forcing=forcing + sign * epsilon * df,
        ).vjp(cotangent)
        for sign in (1, -1)
    ]
    for key in ("initial", "forcing"):
        difference = (varied[0][key] - varied[1][key]) / (2 * epsilon)
        np.testing.assert_allclose(result[key], difference, rtol=7e-7, atol=3e-10)


@pytest.mark.parametrize("distributed", [True, False])
def test_mixed_state_forcing_blocks_obey_bilinear_symmetry(distributed):
    path = _path(distributed=distributed)
    rng = np.random.default_rng(6422)
    cotangent = rng.normal(size=path.states.shape)
    initial_direction = rng.normal(size=3)
    forcing_direction = rng.normal(size=(6, 3) if distributed else (6,))
    initial_only = path.state_forcing_hvp(cotangent, d_initial=initial_direction)
    forcing_only = path.state_forcing_hvp(cotangent, d_forcing=forcing_direction)
    left = initial_direction @ forcing_only["initial"]
    right = np.sum(forcing_direction * initial_only["forcing"])
    np.testing.assert_allclose(left, right, rtol=5e-12, atol=3e-14)
    together = path.state_forcing_hvp(
        cotangent, d_initial=initial_direction, d_forcing=forcing_direction
    )
    for key in ("initial", "forcing"):
        np.testing.assert_allclose(
            together[key], initial_only[key] + forcing_only[key], rtol=5e-12, atol=3e-14
        )


@pytest.mark.parametrize("distributed", [True, False])
@pytest.mark.parametrize("mode", ["initial", "forcing", "both"])
def test_joint_hessian_cpu_cuda_agreement_preserves_control_shapes(cupy, distributed, mode):
    cpu = _path(distributed=distributed)
    gpu = _path(distributed=distributed, backend="cupy")
    rng = np.random.default_rng(4431)
    cotangent = rng.normal(size=cpu.states.shape)
    di = rng.normal(size=3) if mode != "forcing" else None
    df = rng.normal(size=(6, 3) if distributed else (6,)) if mode != "initial" else None
    expected = cpu.state_forcing_hvp(cotangent, d_initial=di, d_forcing=df)
    actual = gpu.state_forcing_hvp(
        cupy.asarray(cotangent),
        d_initial=None if di is None else cupy.asarray(di),
        d_forcing=None if df is None else cupy.asarray(df),
    )
    for key in ("initial", "forcing"):
        assert isinstance(actual[key], cupy.ndarray) and actual[key].dtype == cupy.float64
        assert actual[key].shape == expected[key].shape
        np.testing.assert_allclose(
            cupy.asnumpy(actual[key]), expected[key], rtol=5e-11, atol=4e-13
        )


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("distributed", [True, False])
def test_zero_beta_and_no_steps_have_zero_joint_hessian_blocks(request, backend, distributed):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    path = _path(backend=backend, distributed=distributed, beta=0)
    forcing_direction = np.full((6, 3) if distributed else (6,), -0.1)
    result = path.state_forcing_hvp(
        np.ones(path.states.shape), d_initial=[-0.2, 0.1, 0.3], d_forcing=forcing_direction
    )
    for key, shape in (("initial", (3,)), ("forcing", forcing_direction.shape)):
        np.testing.assert_array_equal(path.affine.to_host(result[key]), np.zeros(shape))
    empty_forcing = np.empty((0, 3)) if distributed else np.empty(0)
    empty_path = solve_strang(
        path.affine,
        [0.3, 0.5, 0.4],
        empty_forcing,
        beta=0.3,
        load=None if distributed else [0.5, 0, 0.2],
    )
    result = empty_path.state_forcing_hvp(
        [[1, -2, 0.5]], d_initial=[0.1, -0.2, 0.3], d_forcing=empty_forcing
    )
    np.testing.assert_array_equal(path.affine.to_host(result["initial"]), np.zeros(3))
    assert result["forcing"].shape == empty_forcing.shape


def test_initial_wrapper_uses_shared_kernel_without_allocating_forcing_output(monkeypatch):
    path = _path()
    cotangent = np.ones(path.states.shape)
    direction = [0.1, -0.2, 0.3]
    expected = path.state_forcing_hvp(cotangent, d_initial=direction)["initial"]
    original = np.zeros_like

    def guard(value, *args, **kwargs):
        assert value is not path._forcing, "Initial-only action must not allocate forcing output"
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "zeros_like", guard)
    np.testing.assert_array_equal(path.initial_hvp(cotangent, direction), expected)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_tiny_forcing_direction_has_representable_hessian_under_huge_reaction(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0, step=2, rate=1, degree=14, backend=backend
    )
    path = solve_strang(action, [0], [[0]], beta=1e308)
    result = path.state_forcing_hvp([[0], [1]], d_forcing=[[1e-308]])
    np.testing.assert_allclose(action.to_host(result["initial"]), [-4], rtol=5e-12)
    np.testing.assert_allclose(action.to_host(result["forcing"]), [[-8]], rtol=5e-12)


def _parameter_path(*, backend="numpy", distributed=True, delta=0.0, direction=None, beta=0.3):
    partials = {
        "first": sparse.csr_matrix([[-2, 1, 1], [0, -1, 1], [1, 0, -1]], dtype=float),
        "second": sparse.csr_matrix([[-1, 1, 0], [1, -2, 1], [0, 1, -1]], dtype=float),
    }
    initial = np.array([0.3, 0.5, 0.4])
    forcing = np.full((4, 3), 0.2) if distributed else np.linspace(0.1, 0.4, 4)
    coefficients, alpha = np.array([0.7, 0.4]), -0.2
    if direction is not None:
        initial += delta * direction["initial"]
        forcing += delta * direction["forcing"]
        coefficients += delta * np.array([direction["operator"][name] for name in partials])
        alpha += delta * direction["alpha"]
        beta += delta * direction["beta"]
    action = PositiveAffineAction(
        sum(coefficient * matrix for coefficient, matrix in zip(coefficients, partials.values())),
        alpha,
        step=0.1,
        rate=5,
        degree=14,
        backend=backend,
    )
    return (
        solve_strang(
            action,
            initial,
            forcing,
            beta=beta,
            load=None if distributed else [0.6, 0.2, 0.4],
        ),
        partials,
    )


def _parameter_direction(distributed, seed=9102, mode="all"):
    rng = np.random.default_rng(seed)
    result = {
        "initial": rng.normal(scale=0.1, size=3),
        "forcing": rng.normal(scale=0.1, size=(4, 3) if distributed else (4,)),
        "operator": dict(zip(("first", "second"), rng.normal(scale=0.1, size=2))),
        "alpha": float(rng.normal(scale=0.1)),
        "beta": float(rng.normal(scale=0.1)),
    }
    for key in result:
        if mode != "all" and key != mode:
            result[key] = dict.fromkeys(result[key], 0.0) if key == "operator" else result[key] * 0
    return result


def _parameter_hvp(path, partials, cotangent, direction):
    d_operator = sum(direction["operator"][name] * matrix for name, matrix in partials.items())
    return path.hvp(
        cotangent,
        operator_partials=partials,
        d_initial=direction["initial"],
        d_forcing=direction["forcing"],
        dQ=d_operator,
        dalpha=direction["alpha"],
        dbeta=direction["beta"],
    )


def _parameter_pair(direction, covector):
    return (
        np.sum(direction["initial"] * covector["initial"])
        + np.sum(direction["forcing"] * covector["forcing"])
        + sum(direction["operator"][name] * value for name, value in covector["operator"].items())
        + direction["alpha"] * covector["alpha"]
        + direction["beta"] * covector["beta"]
    )


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("distributed", [True, False])
@pytest.mark.parametrize("mode", ["initial", "forcing", "operator", "alpha", "beta", "all"])
def test_full_parameter_hvp_matches_every_output_of_the_varied_pullback(
    request, backend, distributed, mode
):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    path, partials = _parameter_path(backend=backend, distributed=distributed)
    direction = _parameter_direction(distributed, mode=mode)
    cotangent = np.random.default_rng(7503).normal(size=path.states.shape)
    actual = _parameter_hvp(path, partials, cotangent, direction)
    step = 1e-5
    varied = [
        _parameter_path(
            backend=backend, distributed=distributed, delta=sign * step, direction=direction
        )[0].vjp(cotangent, operator_partials=partials)
        for sign in (1, -1)
    ]
    for key in ("initial", "forcing", "alpha", "beta"):
        expected = (varied[0][key] - varied[1][key]) / (2 * step)
        observed = path.affine.to_host(actual[key])
        expected = path.affine.to_host(expected)
        np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=4e-10)
    for name in partials:
        expected = (varied[0]["operator"][name] - varied[1]["operator"][name]) / (2 * step)
        np.testing.assert_allclose(actual["operator"][name], expected, rtol=3e-6, atol=4e-10)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("distributed", [True, False])
def test_full_parameter_hessian_obeys_bilinear_symmetry(request, backend, distributed):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    path, partials = _parameter_path(backend=backend, distributed=distributed)
    cotangent = np.random.default_rng(3440).normal(size=path.states.shape)
    directions = [_parameter_direction(distributed, seed) for seed in (324, 531)]
    products = [_parameter_hvp(path, partials, cotangent, value) for value in directions]
    products = [
        {
            key: path.affine.to_host(value) if key in ("initial", "forcing") else value
            for key, value in product.items()
        }
        for product in products
    ]
    np.testing.assert_allclose(
        _parameter_pair(directions[0], products[1]),
        _parameter_pair(directions[1], products[0]),
        rtol=3e-11,
        atol=2e-12,
    )


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_scalar_growth_reaction_hessian_has_an_independent_closed_form(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    initial, step, alpha, beta, weight = 0.4, 0.2, -0.3, 0.7, 1.1
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), alpha, step=step, rate=1, degree=14, backend=backend
    )
    path = solve_strang(action, [initial], [[0.0]], beta=beta)
    growth, half = exp(alpha * step), step / 2
    z = beta * half * initial
    denominator = 1 + z * (1 + growth)
    constant = 1 + z
    haa = step**2 * initial * constant * growth * (denominator - 2 * z * growth) / denominator**3
    hab = (
        step
        * growth
        * half
        * initial**2
        * (denominator - 2 * constant * (1 + growth))
        / denominator**3
    )
    hbb = 2 * half**2 * initial**3 * growth * (1 + growth) ** 2 / denominator**3
    alpha_product = path.hvp([[0.3], [weight]], dalpha=1.0)
    beta_product = path.hvp([[0.3], [weight]], dbeta=1.0)
    np.testing.assert_allclose(alpha_product["alpha"], weight * haa, rtol=5e-12, atol=1e-14)
    np.testing.assert_allclose(alpha_product["beta"], weight * hab, rtol=5e-12, atol=1e-14)
    np.testing.assert_allclose(beta_product["alpha"], weight * hab, rtol=5e-12, atol=1e-14)
    np.testing.assert_allclose(beta_product["beta"], weight * hbb, rtol=5e-12, atol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("distributed", [True, False])
def test_zero_reaction_keeps_nonzero_mixed_beta_curvature(request, backend, distributed):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    path, partials = _parameter_path(backend=backend, distributed=distributed, beta=0.0)
    direction = _parameter_direction(distributed, mode="beta")
    direction["beta"] = 1.0
    cotangent = np.random.default_rng(3233).normal(size=path.states.shape)
    actual = _parameter_hvp(path, partials, cotangent, direction)
    step = 1e-5
    pulled = [
        _parameter_path(
            backend=backend,
            distributed=distributed,
            beta=0.0,
            delta=shift * step,
            direction=direction,
        )[0].vjp(cotangent, operator_partials=partials)
        for shift in (0, 1, 2)
    ]
    for key in ("initial", "forcing", "alpha", "beta"):
        expected = (-3 * pulled[0][key] + 4 * pulled[1][key] - pulled[2][key]) / (2 * step)
        np.testing.assert_allclose(
            path.affine.to_host(actual[key]),
            path.affine.to_host(expected),
            rtol=2e-5,
            atol=2e-9,
        )
    assert abs(actual["beta"]) > 1e-5
