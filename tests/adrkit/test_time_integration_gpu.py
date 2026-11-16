"""Проверка одного дискретного оператора и его производных на CPU и CUDA."""

from copy import deepcopy
import pickle

import numpy as np
import pytest
from scipy import sparse

from adrkit.time_integration import PositiveAffineAction, solve_strang


@pytest.fixture(scope="module")
def cupy():
    module = pytest.importorskip("cupy")
    try:
        available = module.cuda.runtime.getDeviceCount()
    except module.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    if not available:
        pytest.skip("CUDA device unavailable")
    return module


def generator():
    return sparse.csr_matrix([[-2.0, 2.0, 0.0], [1.0, -4.0, 3.0], [0.0, 2.0, -2.0]])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("operation", ["jvp", "jvp_many", "vjp", "trajectory_vjp"])
def test_public_derivative_rejects_overflow_for_both_backends(request, backend, operation):
    if backend == "cupy":
        request.getfixturevalue("cupy")
    affine = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), 0.0, step=1, rate=1, degree=2, backend=backend
    )
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(FloatingPointError):
        if operation == "jvp":
            affine.jvp([1.0], [0.0], du=[1e308], dalpha=1e308)
        elif operation == "jvp_many":
            affine.jvp_many([1.0], [0.0], [[1e308]], dalpha=[1e308])
        elif operation == "vjp":
            affine.vjp([2.0], [0.0], [1e308])
        else:
            trajectory = solve_strang(affine, [2.0], [[0.0]])
            trajectory.vjp([[0.0], [1e308]])


def test_device_stream_reports_nonfinite_tangent_when_fully_consumed(cupy):
    affine = action(generator(), "cupy", alpha=1.0)
    trajectory = solve_strang(
        affine, [0.1, 0.2, 0.3], np.zeros(12), beta=0.0, load=[0.0, 0.0, 0.0]
    )
    with pytest.raises(FloatingPointError, match="tangent"):
        list(trajectory.iter_jvp_many(np.full((3, 1), 1e308)))


def test_unknown_backend_is_rejected_before_optional_import():
    with pytest.raises(ValueError, match="backend must be"):
        PositiveAffineAction(generator(), -0.2, step=0.1, rate=5, degree=14, backend="other")


def action(q, backend, alpha=-0.2):
    return PositiveAffineAction(q, alpha, step=0.1, rate=5.0, degree=14, backend=backend)


@pytest.mark.parametrize("clone", [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_cpu_trajectory_roundtrip_preserves_values_and_ownership(clone):
    original = solve_strang(
        action(generator(), "numpy"), [0.3, 0.0, 0.4], np.full((3, 3), 0.2), beta=0.3
    )
    restored = clone(original)
    assert restored.affine.backend == "numpy"
    assert restored.affine.array_module is np
    assert restored.states is not original.states
    assert not restored.states.flags.writeable
    np.testing.assert_array_equal(restored.states, original.states)
    np.testing.assert_array_equal(restored.jvp(dbeta=1.0), original.jvp(dbeta=1.0))
    with pytest.raises(ValueError):
        restored.states[0, 0] = 1.0
    exported = restored.affine.to_host(restored.states)
    exported[:] = 0.0
    np.testing.assert_array_equal(restored.states, original.states)


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.3])
def test_device_trajectory_derivatives_match_cpu_and_duality(cupy, distributed, beta):
    rng = np.random.default_rng(9201)
    q, intervals, count = generator(), 5, 4
    partials = {
        "scale": q,
        "edge": sparse.csr_matrix([[-1, 1, 0], [0, 0, 0], [0, 0, 0]]),
    }
    initial = np.array([0.3, 0.0, 0.4])
    forcing = np.full((intervals, 3) if distributed else (intervals,), 0.3)
    load = None if distributed else np.array([0.4, 0.0, 0.2])
    di = rng.normal(scale=0.1, size=(3, count))
    di[1] = 0.0
    df = rng.normal(scale=0.1, size=(*forcing.shape, count))
    directions = rng.normal(scale=0.1, size=(len(partials), count))
    directions[:, -1] = 0.0
    da, db = rng.normal(scale=0.1, size=count), rng.normal(scale=0.1, size=count)
    cpu, gpu = action(q, "numpy"), action(q, "cupy")
    for name in (
        "transport_weights",
        "source_weights",
        "source_alpha_derivative",
        "source_alpha_second_derivative",
    ):
        np.testing.assert_array_equal(gpu.to_host(getattr(gpu, name)), getattr(cpu, name))
    cpu_path = solve_strang(cpu, initial, forcing, beta=beta, load=load)
    gpu_path = solve_strang(gpu, initial, forcing, beta=beta, load=load)
    assert isinstance(gpu_path.states, cupy.ndarray)
    assert gpu_path.states.dtype == cupy.float64
    np.testing.assert_allclose(
        gpu.to_host(gpu_path.states), cpu_path.states, rtol=2e-13, atol=2e-15
    )
    np.testing.assert_allclose(
        gpu_path.polynomial_error_bound, cpu_path.polynomial_error_bound, rtol=3e-13
    )
    kwargs = dict(operator_partials=partials, operator_directions=directions, dalpha=da, dbeta=db)
    expected = cpu_path.jvp_many(di, df, **kwargs)
    actual = gpu.to_host(gpu_path.jvp_many(di, df, **kwargs))
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
    streamed = gpu_path.iter_jvp_many(di, df, **kwargs)
    first = next(streamed)
    first[:] = 123.0
    rest = list(streamed)
    np.testing.assert_allclose(gpu.to_host(cupy.stack(rest)), expected[1:], rtol=3e-12, atol=3e-14)
    dq = sum(directions[j, 0] * value for j, value in enumerate(partials.values()))
    scalar = gpu_path.jvp(di[:, 0], df[..., 0], dQ=dq, dalpha=da[0], dbeta=db[0])
    np.testing.assert_allclose(gpu.to_host(scalar), expected[..., 0], rtol=3e-12, atol=3e-14)
    cotangent = rng.normal(size=cpu_path.states.shape)
    pulled = gpu_path.vjp(cotangent, partials)
    reference = cpu_path.vjp(cotangent, partials)
    for name in ("initial", "forcing"):
        np.testing.assert_allclose(
            gpu.to_host(pulled[name]), reference[name], rtol=3e-12, atol=3e-14
        )
    for name in ("alpha", "beta"):
        assert isinstance(pulled[name], float)
        np.testing.assert_allclose(pulled[name], reference[name], rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(
        list(pulled["operator"].values()),
        list(reference["operator"].values()),
        rtol=3e-12,
        atol=3e-14,
    )
    right = di.T @ gpu.to_host(pulled["initial"])
    right += np.sum(df * gpu.to_host(pulled["forcing"])[..., None], axis=tuple(range(df.ndim - 1)))
    right += directions.T @ np.asarray(list(pulled["operator"].values()))
    right += da * pulled["alpha"] + db * pulled["beta"]
    left = np.sum(actual * cotangent[..., None], axis=(0, 1))
    np.testing.assert_allclose(left, right, rtol=3e-12, atol=3e-14)
    # The states property must not let a caller invalidate future derivatives.
    external = gpu_path.states
    external[:] = 0.0
    np.testing.assert_allclose(
        gpu.to_host(gpu_path.states), cpu_path.states, rtol=2e-13, atol=2e-15
    )


def test_device_affine_public_operations_and_input_ownership(cupy):
    q = generator()
    cpu, gpu = action(q, "numpy"), action(q, "cupy")
    u, f, du, df = [
        np.array(v) for v in ([0.3, 0.2, 0.1], [0.1, 0.0, 0.2], [0.2, -0.1, 0.3], [-0.1, 0.0, 0.1])
    ]
    np.testing.assert_allclose(
        gpu.to_host(gpu.apply(u, f)), cpu.apply(u, f), rtol=2e-13, atol=2e-15
    )
    np.testing.assert_allclose(
        gpu.to_host(gpu.jvp(u, f, du, df, dQ=q, dalpha=0.1)),
        cpu.jvp(u, f, du, df, dQ=q, dalpha=0.1),
        rtol=3e-12,
        atol=3e-14,
    )
    for name, value in gpu.vjp(u, f, np.ones(3), {"q": q}).items():
        reference = cpu.vjp(u, f, np.ones(3), {"q": q})[name]
        if name == "operator":
            np.testing.assert_allclose(value["q"], reference["q"], rtol=3e-12, atol=3e-14)
        else:
            np.testing.assert_allclose(gpu.to_host(value), reference, rtol=3e-12, atol=3e-14)
    initial = cupy.asarray(u)
    forcing = cupy.full((3, 3), 0.2)
    path = solve_strang(gpu, initial, forcing, beta=0.3)
    before = gpu.to_host(path.states)
    initial[:] = 20.0
    forcing[:] = 30.0
    np.testing.assert_array_equal(gpu.to_host(path.states), before)


@pytest.mark.parametrize("beta", [0.0, 0.3])
def test_device_beta_derivative_matches_one_sided_difference(cupy, beta):
    gpu = action(generator(), "cupy")
    initial, forcing, h = np.array([0.3, 0.0, 0.4]), np.full((4, 3), 0.2), 1e-5
    path = solve_strang(gpu, initial, forcing, beta=beta)
    exact = gpu.to_host(path.jvp(dbeta=1.0))
    forward = solve_strang(gpu, initial, forcing, beta=beta + h).states
    farther = solve_strang(gpu, initial, forcing, beta=beta + 2 * h).states
    difference = gpu.to_host((-3 * path.states + 4 * forward - farther) / (2 * h))
    np.testing.assert_allclose(exact, difference, rtol=2e-7, atol=4e-11)


def test_device_decay_preserves_overflow_safe_positive_limit(cupy):
    q = sparse.csr_matrix((1, 1))
    cpu = PositiveAffineAction(q, 0.0, step=1.0, rate=1.0, degree=18)
    gpu = PositiveAffineAction(q, 0.0, step=1.0, rate=1.0, degree=18, backend="cupy")
    expected = solve_strang(cpu, [1e308], [[0.0]], beta=1e308).states[-1]
    actual = gpu.to_host(solve_strang(gpu, [1e308], [[0.0]], beta=1e308).states[-1])
    assert actual[0] > 0
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
