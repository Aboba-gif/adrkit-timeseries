"""Целевой риск, профильная информация и конечный выбор: независимые матричные случаи."""

from itertools import combinations
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from adrkit.inverse.design import (
    CenteredBudgetGoalRisk,
    CovarianceFactor,
    DesignCandidate,
    FiniteDesignResult,
    UnresolvedDesign,
    affine_goal_risk,
    centered_budget_goal_risk,
    innovation_covariance_budget,
    joint_moment_risk,
    profile_information,
    robust_ball_maximum,
    select_designs,
)
from adrkit.inverse import CertifiedDesignBound, finite_design_descent


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("physical_size", [0.1, 4.0])
def test_innovation_budget_reconstructs_spd_covariance_and_preserves_floor(
    diagonal, physical_size
):
    covariance = np.array([2.0, 1.0]) if diagonal else np.array([[2.0, 0.4], [0.4, 1.0]])
    matrix = np.diag(covariance) if diagonal else covariance
    physical = physical_size * np.eye(2)
    result = innovation_covariance_budget(physical, covariance, fraction=0.7)
    factor = result.residual_factor.factor
    residual = factor @ factor.T
    np.testing.assert_allclose(
        residual + result.physical_scale * physical @ physical.T,
        matrix,
        rtol=2e-14,
        atol=2e-14,
    )
    floor = residual - (1 - result.fraction) * matrix
    assert np.linalg.eigvalsh(floor).min() >= -2e-14
    expected_ratio = physical_size**2 / np.linalg.eigvalsh(matrix).min()
    np.testing.assert_allclose(result.max_ratio, expected_ratio, rtol=2e-14)
    if physical_size < 1:
        assert result.physical_scale == 1.0
    else:
        np.testing.assert_allclose(
            result.physical_scale * result.max_ratio, result.fraction, rtol=2e-14
        )
        assert 0 < result.physical_scale < 1


def test_innovation_budget_is_invariant_under_latent_orthogonal_rotation():
    covariance = np.array([[2.0, 0.4, -0.1], [0.4, 1.0, 0.2], [-0.1, 0.2, 1.5]])
    physical = np.array([[2.0, -0.3], [0.4, 1.2], [-0.2, 0.5]])
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    first = innovation_covariance_budget(physical, covariance, fraction=0.6)
    second = innovation_covariance_budget(physical @ rotation, covariance, fraction=0.6)
    np.testing.assert_allclose(first.max_ratio, second.max_ratio, rtol=2e-14)
    np.testing.assert_allclose(first.physical_scale, second.physical_scale, rtol=2e-14)
    a, b = first.residual_factor.factor, second.residual_factor.factor
    np.testing.assert_allclose(a @ a.T, b @ b.T, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("columns", [0, 2])
def test_zero_physical_innovations_keep_full_budget_with_valid_nonsymmetric_factor(
    columns,
):
    covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
    result = innovation_covariance_budget(np.zeros((2, columns)), covariance, fraction=0.6)
    assert result.physical_scale == 1.0 and result.max_ratio == 0.0
    factor = result.residual_factor.factor
    assert not np.allclose(factor, factor.T)
    np.testing.assert_allclose(factor @ factor.T, covariance, rtol=2e-14, atol=2e-14)


def test_zero_physical_innovation_factor_does_not_remove_unobserved_true_target():
    physical, target, estimator = (
        np.zeros((2, 1)),
        np.array([[2.0]]),
        np.array([[0.3, -0.2]]),
    )
    budget = innovation_covariance_budget(physical, [1.0, 2.0], fraction=0.6)
    error_map = np.column_stack(
        (
            np.sqrt(budget.physical_scale) * (estimator @ physical - target),
            estimator @ budget.residual_factor.factor,
        )
    )
    result = joint_moment_risk(error_map, [0.0], np.zeros(3), np.ones(3))
    noise_variance = np.linalg.norm(estimator @ budget.residual_factor.factor) ** 2
    np.testing.assert_allclose(result.value, 4.0 + noise_variance, rtol=2e-14)


def test_innovation_budget_requires_matching_physical_scaling_in_data_and_target():
    physical = np.diag([2.0, 0.5])
    result = innovation_covariance_budget(physical, np.ones(2), fraction=0.6)
    residual = result.residual_factor.factor @ result.residual_factor.factor.T
    wrong_budget = residual + physical @ physical.T
    excess = (1 - result.physical_scale) * physical @ physical.T
    np.testing.assert_allclose(wrong_budget - np.eye(2), excess, rtol=2e-14, atol=2e-14)
    assert np.linalg.eigvalsh(wrong_budget - np.eye(2)).min() > 0
    root_scale = np.sqrt(result.physical_scale)
    # Identity estimator and physical target have the same map before allocation.
    correct = np.column_stack(
        (root_scale * physical - root_scale * physical, result.residual_factor.factor)
    )
    wrong = np.column_stack((root_scale * physical - physical, result.residual_factor.factor))
    correct_risk = joint_moment_risk(correct, np.zeros(2), np.zeros(4), np.ones(4))
    wrong_risk = joint_moment_risk(wrong, np.zeros(2), np.zeros(4), np.ones(4))
    np.testing.assert_allclose(correct_risk.value, np.trace(residual), rtol=2e-14)
    assert wrong_risk.value > correct_risk.value


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, True, np.nan])
def test_innovation_budget_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        innovation_covariance_budget(np.eye(2), np.ones(2), fraction=fraction)


@pytest.mark.parametrize(
    "covariance",
    [[1.0, 0.0], [1.0, -1.0], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.5], [0.1, 1.0]]],
)
def test_innovation_budget_requires_strictly_positive_covariance(covariance):
    with pytest.raises(ValueError):
        innovation_covariance_budget(np.eye(2), covariance, fraction=0.7)


@pytest.mark.parametrize("size", [1e-200, 1e200])
def test_innovation_budget_rejects_unrepresentable_spectral_scale(size):
    with pytest.raises(ValueError):
        innovation_covariance_budget([[size]], [1.0], fraction=0.7)


def test_innovation_budget_validates_shape_mask_and_owns_factor():
    with pytest.raises(ValueError):
        innovation_covariance_budget(np.zeros((0, 1)), [], fraction=0.7)
    with pytest.raises(ValueError):
        innovation_covariance_budget(np.ma.array([[1.0]], mask=True), [1.0], fraction=0.7)
    with pytest.raises(ValueError):
        innovation_covariance_budget([[1.0 + 1j]], [1.0], fraction=0.7)
    physical, covariance = np.eye(2), np.array([[2.0, 0.4], [0.4, 1.0]])
    result = innovation_covariance_budget(physical, covariance, fraction=0.7)
    saved = result.residual_factor.factor.copy()
    physical[:] = 9
    covariance[:] = 8
    np.testing.assert_array_equal(result.residual_factor.factor, saved)
    assert not result.residual_factor.factor.flags.writeable


def test_joint_moment_risk_matches_finite_joint_population_with_nonlinear_component():
    data = np.array([[-2.0, 0.2], [-1.0, 1.1], [0.5, -0.3], [1.5, 0.7], [2.0, -0.9], [0.3, 0.5]])
    scales = np.abs(data[:, :1] + 0.2 * data[:, 1:])
    target = 0.3 * data[:, :1] - 0.1 * data[:, 1:] + 0.2 * scales
    samples = np.column_stack((data, scales, target))
    matrix = np.array([[0.7, -0.2, 0.4, -1.0], [-0.1, 0.6, 0.3, -0.5]])
    offset = np.array([0.2, -0.3])
    mean = samples.mean(axis=0)
    centered = samples - mean
    factor = CovarianceFactor(centered.T / np.sqrt(len(samples)))
    result = joint_moment_risk(matrix, offset, mean, factor)
    errors = samples @ matrix.T + offset
    error_mean = errors.mean(axis=0)
    expected_variance = np.mean(np.sum((errors - error_mean) ** 2, axis=1))
    np.testing.assert_allclose(result.mean_error, error_mean, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(result.squared_bias, error_mean @ error_mean, rtol=1e-14)
    np.testing.assert_allclose(result.variance, expected_variance, rtol=1e-14)
    np.testing.assert_allclose(result.value, np.mean(np.sum(errors**2, axis=1)), rtol=1e-14)
    assert mean[2] > 0
    independent = joint_moment_risk(matrix, offset, mean, np.diag(factor.factor @ factor.factor.T))
    assert abs(independent.value - result.value) > 0.1


def test_joint_moment_risk_keeps_correlated_cancellation_and_explicit_mean():
    result = joint_moment_risk([[1.0, -1.0]], [0.5], [2.0, 2.0], CovarianceFactor([[1.0], [1.0]]))
    np.testing.assert_array_equal(result.mean_error, [0.5])
    assert result.squared_bias == 0.25
    assert result.variance == 0.0
    assert result.value == 0.25


def test_joint_moment_factor_diagonal_and_matrix_representations_agree():
    matrix, mean, offset = (
        np.array([[0.2, -0.6, 0.5], [0.4, 0.1, -0.3]]),
        [0.7, -0.2, 0.5],
        [0.3, -0.1],
    )
    diagonal = np.array([0.25, 4.0, 0.0])
    reference = joint_moment_risk(matrix, offset, mean, diagonal)
    for covariance in (np.diag(diagonal), CovarianceFactor(np.diag(np.sqrt(diagonal)))):
        result = joint_moment_risk(matrix, offset, mean, covariance)
        np.testing.assert_array_equal(result.mean_error, reference.mean_error)
        np.testing.assert_allclose(result.value, reference.value, rtol=1e-14)


def test_joint_moment_result_owns_immutable_mean_error():
    matrix, offset, mean, diagonal = (
        np.eye(2),
        np.array([0.2, 0.4]),
        np.array([0.5, -0.1]),
        np.ones(2),
    )
    result = joint_moment_risk(matrix, offset, mean, diagonal)
    expected = result.mean_error.copy()
    matrix[:] = 0
    offset[:] = 9
    mean[:] = 8
    diagonal[:] = 7
    np.testing.assert_array_equal(result.mean_error, expected)
    assert not result.mean_error.flags.writeable
    with pytest.raises(ValueError):
        result.mean_error.flags.writeable = True


def test_joint_moment_risk_rejects_negative_joint_variance():
    with pytest.raises(ValueError):
        joint_moment_risk([[1.0, 0.0]], [0.0], [0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])


def test_centered_budget_risk_matches_joint_moments_with_correlated_low_rank_physics():
    B = np.array([[0.7, -0.2, 0.3], [-0.1, 0.5, 0.4]])
    O = np.array([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.1]])
    T = np.array([[0.6, 0.2], [-0.3, 0.7]])
    innovation = np.array([[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [-0.1, 0.3, 1.0]])
    inverse = np.linalg.solve(innovation, np.eye(3))
    innovation_factor = 2 * np.eye(3)
    physical_factor = np.array([[0.4], [0.8]])
    mean = np.array([-0.3, 0.5])
    offset = np.array([0.3, -0.2])
    physical_innovation = innovation @ O @ physical_factor
    data = B @ inverse @ innovation_factor
    observed, target = B @ O @ physical_factor, T @ physical_factor
    coefficients = []
    for scale in (0.0, 0.2, 0.7, 1.0):
        residual = innovation_factor @ innovation_factor.T - scale**2 * (
            physical_innovation @ physical_innovation.T
        )
        joint_factor = np.zeros((5, 4))
        joint_factor[:2, :1] = physical_factor
        joint_factor[2:, 1:] = np.linalg.cholesky(residual)
        # One xi enters the data and target; the residual is independent and centered jointly.
        joint_mean = np.r_[mean, -scale * innovation @ O @ mean]
        error_map = np.column_stack((scale * (B @ O - T), B @ inverse))
        reference = joint_moment_risk(
            error_map, offset, joint_mean, CovarianceFactor(joint_factor)
        )
        result = centered_budget_goal_risk(
            offset, data, observed, target, T @ mean, max_scale=scale
        )
        assert isinstance(result, CenteredBudgetGoalRisk)
        np.testing.assert_allclose(result.value_at_max_scale, reference.value, rtol=3e-14)
        np.testing.assert_allclose(reference.mean_error, offset - scale * T @ mean, atol=1e-15)
        coefficients.append([result.d, result.h, result.a])
    np.testing.assert_allclose(coefficients, np.tile(coefficients[0], (4, 1)), rtol=1e-15)
    missing_cross_term = np.linalg.norm(T @ mean) ** 2 + np.linalg.norm(target) ** 2
    assert result.a != pytest.approx(missing_cross_term)


def test_centered_budget_risk_finds_a_concave_interior_maximum():
    # O=T=B=1, total variance=1; residual variance 1-s**2 is PSD on [0,1].
    # With E[xi]=-1/2, R(s)=(1+s/2)**2+1-s**2=2+s-3*s**2/4.
    result = centered_budget_goal_risk([1.0], [[1.0]], [[1.0]], [[1.0]], [-0.5], max_scale=1)
    np.testing.assert_allclose([result.d, result.h, result.a], [2.0, -0.5, -0.75])
    assert result.value_at_zero == 2.0 and result.value_at_max_scale == 2.25
    np.testing.assert_allclose([result.vertex_scale, result.argmax], [2 / 3, 2 / 3])
    np.testing.assert_allclose([result.vertex_value, result.value], [7 / 3, 7 / 3])
    truncated = centered_budget_goal_risk([1.0], [[1.0]], [[1.0]], [[1.0]], [-0.5], max_scale=0.25)
    assert truncated.vertex_scale is None and truncated.vertex_value is None
    assert truncated.argmax == 0.25 and truncated.value == truncated.value_at_max_scale


def test_centered_budget_risk_keeps_mean_with_empty_factors_and_labels_convex_vertex():
    empty = np.empty((1, 0))
    # Deterministic physical change: R(s)=(1-2*s)**2; its vertex is a minimum.
    result = centered_budget_goal_risk([1.0], empty, empty, empty, [2.0], max_scale=1)
    assert result.vertex_scale == 0.5 and result.vertex_value == 0.0
    assert result.value_at_zero == result.value_at_max_scale == result.value == 1.0
    assert result.argmax == 0.0
    zero_interval = centered_budget_goal_risk([1.0], empty, empty, empty, [2.0], max_scale=0)
    assert zero_interval.value == 1 and zero_interval.argmax == 0
    assert zero_interval.vertex_scale is None


def test_centered_budget_risk_accepts_exact_zero_boundary_cancellation():
    result = centered_budget_goal_risk([0.0], [[1.0]], [[1.0]], [[1.0]], [0.0], max_scale=1)
    assert result.a == -1 and result.value_at_max_scale == 0.0
    assert result.value == 1 and result.argmax == 0


def test_centered_budget_risk_preserves_common_latent_rotation():
    data = np.array([[2.0, 0.0], [0.0, 2.0]])
    observed = np.array([[0.2, 0.1], [-0.3, 0.4]])
    target = np.array([[0.5, -0.2], [0.1, 0.3]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    reference = centered_budget_goal_risk(
        [0.3, -0.2], data, observed, target, [-0.1, 0.2], max_scale=1
    )
    rotated = centered_budget_goal_risk(
        [0.3, -0.2],
        data @ rotation,
        observed @ rotation,
        target @ rotation,
        [-0.1, 0.2],
        max_scale=1,
    )
    np.testing.assert_allclose(
        [reference.d, reference.h, reference.a, reference.value, reference.argmax],
        [rotated.d, rotated.h, rotated.a, rotated.value, rotated.argmax],
        rtol=1e-14,
    )


@pytest.mark.parametrize("mean", [0.0, 2.0])
def test_centered_budget_risk_rejects_negative_endpoint_or_convex_interior(mean):
    # Total data variance zero cannot fund the physical data variance 1.
    # mean=2 has nonnegative endpoints but a negative interior minimum.
    with pytest.raises(ValueError, match="negative.*incompatible.*cancellation"):
        centered_budget_goal_risk(
            [0.0 if mean == 0 else 1.0],
            np.empty((1, 0)),
            [[1.0]],
            [[1.0]],
            [mean],
            max_scale=1,
        )


def test_centered_budget_risk_requires_an_explicit_physical_mean():
    with pytest.raises(TypeError):
        centered_budget_goal_risk([1.0], [[1.0]], [[0.1]], [[0.2]], max_scale=1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("offset", [[1.0]]),
        ("offset", []),
        ("offset", [1j]),
        ("total_data_factor", [[1.0], [2.0]]),
        ("total_data_factor", [1.0]),
        ("physical_target_factor", [[0.2, 0.3]]),
        ("physical_target_mean", None),
        ("physical_target_mean", [np.nan]),
        ("physical_data_factor", [[np.inf]]),
        ("offset", np.ma.array([1.0], mask=[True])),
        ("max_scale", -0.1),
        ("max_scale", 1.1),
        ("max_scale", True),
    ],
)
def test_centered_budget_risk_rejects_invalid_inputs(key, value):
    inputs = dict(
        offset=[1.0],
        total_data_factor=[[1.0]],
        physical_data_factor=[[0.1]],
        physical_target_factor=[[0.2]],
        physical_target_mean=[0.5],
        max_scale=1,
    )
    inputs[key] = value
    with pytest.raises(ValueError):
        centered_budget_goal_risk(**inputs)


@pytest.mark.parametrize("offset", [3e154, float.fromhex("0x0.0000000000001p-1022")])
def test_centered_budget_risk_rejects_unrepresentable_required_energy(offset):
    with pytest.raises(ValueError):
        centered_budget_goal_risk([offset], [[0.0]], [[0.0]], [[0.0]], [0.0], max_scale=1)


def test_centered_budget_risk_result_is_immutable_and_independent_of_inputs():
    offset, data = np.array([1.0]), np.array([[1.0]])
    observed, target, mean = np.array([[0.1]]), np.array([[0.2]]), np.array([0.5])
    result = centered_budget_goal_risk(offset, data, observed, target, mean, max_scale=1)
    saved = (result.d, result.h, result.a, result.value)
    for array in (offset, data, observed, target, mean):
        array[:] = 9.0
    assert (result.d, result.h, result.a, result.value) == saved
    with pytest.raises(AttributeError):
        result.value = 0.0


def assert_ball_result(result, a, matrix, radius, expected, *, atol=2e-10):
    """Проверить допустимость и значение возвращённого свидетеля."""
    assert result.witness.shape == (matrix.shape[1],)
    assert np.linalg.norm(result.witness) <= radius + atol
    residual = a + matrix @ result.witness
    witnessed = float(residual @ residual)
    np.testing.assert_allclose(result.attained_estimate, witnessed, rtol=2e-12, atol=atol)
    np.testing.assert_allclose(result.attained_estimate, expected, rtol=2e-11, atol=atol)
    assert result.dual_upper_estimate + atol >= result.attained_estimate
    assert result.dual_upper_estimate + atol >= expected
    assert result.dual_upper_estimate - expected <= atol
    np.testing.assert_allclose(
        result.gap_estimate,
        result.dual_upper_estimate - result.attained_estimate,
        rtol=1e-6,
        atol=atol,
    )
    assert result.converged_estimate
    assert result.certified is False
    assert isinstance(result.status, str) and result.status
    assert result.iterations >= 0


def test_scalar_robust_maximum_is_nine_and_preserves_inputs():
    a, matrix = np.array([1.0]), np.array([[1.0]])
    original_a, original_matrix = a.copy(), matrix.copy()
    result = robust_ball_maximum(a, matrix, 2.0)
    # max_{|t|<=2} (1+t)^2 = 9; смещение входит внутрь максимума.
    assert_ball_result(result, a, matrix, 2.0, 9.0)
    np.testing.assert_allclose(result.witness, [2.0], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(matrix, original_matrix)
    witness = result.witness.copy()
    a[:] = 100
    matrix[:] = 100
    np.testing.assert_array_equal(result.witness, witness)


def test_hard_case_has_value_sixteen_thirds_and_nonzero_top_component():
    a, matrix = np.array([0.0, 1.0]), np.diag([2.0, 1.0])
    result = robust_ball_maximum(a, matrix, 1.0)
    # На сфере: 4-3*t_1^2+2*t_1+1; максимум при t_1=1/3.
    assert_ball_result(result, a, matrix, 1.0, 16.0 / 3.0)
    np.testing.assert_allclose(result.witness[1], 1.0 / 3.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(abs(result.witness[0]), np.sqrt(8.0 / 9.0), atol=1e-10)


def test_zero_top_linear_component_does_not_always_mean_hard_case():
    a, matrix = np.array([0.0, 6.0]), np.diag([2.0, 1.0])
    result = robust_ball_maximum(a, matrix, 1.0)
    # Формальный нижний компонент при lambda=4 равен 2 и вне шара.
    # Регулярный максимум достигается в (0,1): (6+1)^2=49.
    assert_ball_result(result, a, matrix, 1.0, 49.0)
    np.testing.assert_allclose(result.witness, [0.0, 1.0], rtol=0, atol=1e-10)


def test_pure_quadratic_maximum_uses_largest_singular_value():
    a, matrix = np.zeros(2), np.diag([7.0, 2.0])
    result = robust_ball_maximum(a, matrix, 1.0)
    assert_ball_result(result, a, matrix, 1.0, 49.0)
    np.testing.assert_allclose(abs(result.witness[0]), 1.0, atol=1e-10)


def test_near_hard_case_keeps_a_nonzero_top_linear_component():
    epsilon = 1e-7
    a, matrix = np.array([epsilon, 1.0]), np.diag([2.0, 1.0])
    result = robust_ball_maximum(a, matrix, 1.0, rtol=1e-12, maxiter=256)
    feasible = np.array([np.sqrt(8.0 / 9.0), 1.0 / 3.0])
    feasible_residual = a + matrix @ feasible
    feasible_value = float(feasible_residual @ feasible_residual)
    assert np.linalg.norm(result.witness) <= 1.0 + 1e-10
    assert result.witness[0] > 0.5
    assert result.attained_estimate >= feasible_value - 2e-10
    assert result.attained_estimate > 16.0 / 3.0 + 3.0 * epsilon
    assert result.dual_upper_estimate >= feasible_value - 2e-10
    assert result.gap_estimate <= 2e-10
    assert result.converged_estimate
    assert result.certified is False


@pytest.mark.parametrize("radius", [0.0, 2.0])
def test_empty_model_class_keeps_central_bias(radius):
    a, matrix = np.array([3.0, 4.0]), np.empty((2, 0))
    result = robust_ball_maximum(a, matrix, radius)
    assert_ball_result(result, a, matrix, radius, 25.0)


def test_zero_radius_keeps_bias_with_nonzero_model_directions():
    a, matrix = np.array([3.0, 4.0]), np.array([[7.0, 1.0], [2.0, -3.0]])
    result = robust_ball_maximum(a, matrix, 0.0)
    assert_ball_result(result, a, matrix, 0.0, 25.0)
    np.testing.assert_array_equal(result.witness, [0.0, 0.0])


def affine_case():
    """Полностью заданный случай с разными пространствами и фактором ранга один."""
    return {
        "E": np.array([[1.0, 0.3], [-0.2, 0.8]]),
        "offset": np.array([0.2, -0.3]),
        "O": np.array([[0.5, 1.0], [-1.0, 0.4]]),
        "T": np.array([[0.8, 0.0], [0.1, 0.6]]),
        "Cxi": CovarianceFactor(np.array([[1.0, 0.0], [0.3, 0.6]])),
        "Ce": CovarianceFactor(np.array([[0.5, 0.0], [0.1, 0.7]])),
        "L": np.array([[1.0, 0.2], [0.0, 2.0]]),
        "B": np.array([[1.0, 0.4], [0.1, -0.2]]),
        "Ceta": CovarianceFactor(np.array([[0.3], [0.7]])),
        "rho": 1.25,
    }


def assert_affine_components_equal(left, right):
    for field in (
        "parameter_variance",
        "noise_variance",
        "value_estimate",
        "upper_estimate",
    ):
        np.testing.assert_allclose(
            getattr(left, field), getattr(right, field), rtol=2e-10, atol=2e-11
        )


def test_scalar_affine_risk_is_nine_with_all_three_nonzero_contributions():
    result = affine_goal_risk(
        np.array([[2.0]]),
        np.array([1.0]),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.ones(1),
        np.ones(1),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.ones(1),
        0.5,
    )
    # (2*1-1)^2=1, Var(2*e)=4, max_{|eta|<=0.5}(1+2*eta)^2=4.
    np.testing.assert_allclose(result.parameter_variance, 1.0, atol=1e-13)
    np.testing.assert_allclose(result.noise_variance, 4.0, atol=1e-13)
    np.testing.assert_allclose(result.robust.attained_estimate, 4.0, atol=1e-12)
    np.testing.assert_allclose(result.value_estimate, 9.0, atol=1e-12)
    np.testing.assert_allclose(result.upper_estimate, 9.0, atol=1e-11)
    assert result.converged_estimate


def test_correlated_parameter_cancellation_and_noise_variance_one_half():
    result = affine_goal_risk(
        np.array([[1.0, -1.0]]),
        np.zeros(1),
        np.eye(2),
        np.zeros((1, 2)),
        np.ones((2, 2)),
        np.array([[1.0, 0.75], [0.75, 1.0]]),
        np.ones((1, 1)),
        np.zeros((2, 1)),
        CovarianceFactor(np.empty((1, 0))),
        1.0,
    )
    # Общая случайная компонента двух параметров уничтожается разностью.
    # Для ошибки измерений Var(e_0-e_1)=1+1-2*0.75=0.5.
    np.testing.assert_allclose(result.parameter_variance, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.noise_variance, 0.5, atol=1e-14)
    np.testing.assert_allclose(result.robust.attained_estimate, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.value_estimate, 0.5, atol=1e-13)
    assert result.converged_estimate


def test_zero_raw_covariance_classes_keep_only_transformed_bias():
    result = affine_goal_risk(
        np.ones((1, 1)),
        np.array([2.0]),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.zeros(1),
        np.zeros(1),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.zeros(1),
        3.0,
    )
    assert result.parameter_variance == result.noise_variance == 0.0
    np.testing.assert_allclose(result.value_estimate, 4.0, atol=1e-13)


def test_affine_offset_is_transformed_by_goal_map_inside_robust_maximum():
    result = affine_goal_risk(
        np.zeros((2, 1)),
        np.array([3.0, 4.0]),
        np.zeros((1, 1)),
        np.zeros((2, 1)),
        CovarianceFactor(np.empty((1, 0))),
        CovarianceFactor(np.empty((1, 0))),
        np.array([[2.0, 0.0]]),
        np.zeros((1, 1)),
        CovarianceFactor(np.empty((1, 0))),
        4.0,
    )
    assert result.parameter_variance == result.noise_variance == 0.0
    np.testing.assert_allclose(result.value_estimate, 36.0, atol=1e-13)
    np.testing.assert_allclose(result.upper_estimate, 36.0, atol=1e-13)


def test_rectangular_covariance_factor_owns_readonly_input():
    raw = np.array([[1.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    original = raw.copy()
    factor = CovarianceFactor(raw)
    assert factor.factor.shape == (2, 3)
    assert not np.shares_memory(factor.factor, raw)
    raw[:] = 99
    np.testing.assert_array_equal(factor.factor, original)
    with pytest.raises(ValueError):
        factor.factor[0, 0] = 99


def test_factor_rotation_and_rectangular_representations_preserve_affine_risk():
    case = affine_case()
    result = affine_goal_risk(**case)
    raw_covariances = dict(case)
    for name in ("Cxi", "Ce"):
        root = case[name].factor
        raw_covariances[name] = root @ root.T
    assert_affine_components_equal(result, affine_goal_risk(**raw_covariances))
    rotated = dict(case)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    for name in ("Cxi", "Ce"):
        root = case[name].factor
        rectangular = np.column_stack((root @ rotation, np.zeros(root.shape[0])))
        rotated[name] = CovarianceFactor(rectangular)
    rotated["Ceta"] = CovarianceFactor(-case["Ceta"].factor)
    assert_affine_components_equal(result, affine_goal_risk(**rotated))


def test_general_affine_changes_of_all_three_coordinate_spaces_preserve_risk():
    case = affine_case()
    originals = {
        name: value.copy() for name, value in case.items() if isinstance(value, np.ndarray)
    }
    result = affine_goal_risk(**case)
    output_change = np.array([[2.0, 0.5], [0.0, 1.0]])
    observation_change = np.array([[1.0, 0.2], [0.3, 1.5]])
    parameter_change = np.array([[1.2, 0.0], [0.4, 0.8]])
    inverse_output = np.linalg.inv(output_change)
    inverse_observation = np.linalg.inv(observation_change)
    inverse_parameter = np.linalg.inv(parameter_change)
    changed = affine_goal_risk(
        output_change @ case["E"] @ inverse_observation,
        output_change @ case["offset"],
        observation_change @ case["O"] @ inverse_parameter,
        output_change @ case["T"] @ inverse_parameter,
        CovarianceFactor(parameter_change @ case["Cxi"].factor),
        CovarianceFactor(observation_change @ case["Ce"].factor),
        case["L"] @ inverse_output,
        observation_change @ case["B"],
        case["Ceta"],
        case["rho"],
    )
    assert_affine_components_equal(result, changed)
    for name, original in originals.items():
        np.testing.assert_array_equal(case[name], original)
    np.testing.assert_allclose(
        result.value_estimate,
        result.parameter_variance + result.noise_variance + result.robust.attained_estimate,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.upper_estimate,
        result.parameter_variance + result.noise_variance + result.robust.dual_upper_estimate,
        atol=1e-13,
    )


@pytest.mark.parametrize("history_row,expected", [([0.0, 1.0], 0.0), ([1.0, 0.0], 0.5)])
def test_history_only_separates_source_when_it_observes_imitating_field(history_row, expected):
    # На основном интервале A_y=(1,0)^T имитируется первым компонентом поля.
    # История с A_h=0 должна наблюдать именно этот компонент.
    A = np.array([[0.0], [1.0], [0.0]])
    nuisance = np.vstack((history_row, np.eye(2)))
    original_A, original_nuisance = A.copy(), nuisance.copy()
    result = profile_information(A, nuisance, np.eye(3))
    np.testing.assert_allclose(result.information, [[expected]], atol=2e-13)
    np.testing.assert_allclose(result.projected.T @ result.projected, [[expected]], atol=2e-13)
    assert result.nuisance_rank == 2
    np.testing.assert_array_equal(A, original_A)
    np.testing.assert_array_equal(nuisance, original_nuisance)


def test_repeated_independent_history_increases_profile_information():
    first = profile_information(np.array([[0.0], [1.0]]), np.ones((2, 1)), np.eye(2))
    second = profile_information(np.array([[0.0], [0.0], [1.0]]), np.ones((3, 1)), np.eye(3))
    np.testing.assert_allclose(first.information, [[0.5]], atol=1e-13)
    np.testing.assert_allclose(second.information, [[2.0 / 3.0]], atol=1e-13)
    assert second.information[0, 0] > first.information[0, 0]


def test_profile_handles_exact_nuisance_rank_deficiency_without_ridge():
    A = np.array([[1.0], [0.0]])
    nuisance = np.array([[1.0, 2.0], [1.0, 2.0]])
    result = profile_information(A, nuisance, np.eye(2))
    assert result.nuisance_rank == 1
    np.testing.assert_allclose(result.projected, [[0.5], [-0.5]], atol=2e-13)
    np.testing.assert_allclose(result.information, [[0.5]], atol=2e-13)
    np.testing.assert_allclose(nuisance.T @ result.projected, np.zeros((2, 1)), atol=2e-13)


def test_profile_numerical_rank_uses_declared_relative_threshold():
    A, nuisance = np.array([[0.0], [1.0]]), np.diag([1.0, 1e-12])
    coarse = profile_information(A, nuisance, np.eye(2), relative_threshold=1e-10)
    fine = profile_information(A, nuisance, np.eye(2), relative_threshold=1e-14)
    assert coarse.nuisance_rank == 1 and fine.nuisance_rank == 2
    np.testing.assert_allclose(coarse.information, [[1.0]], atol=1e-13)
    np.testing.assert_allclose(fine.information, [[0.0]], atol=1e-13)
    np.testing.assert_allclose(coarse.nuisance_threshold, 1e-10, rtol=1e-13)
    np.testing.assert_allclose(fine.nuisance_threshold, 1e-14, rtol=1e-13)


def test_profile_correlated_metric_matches_independent_scalar_schur_complement():
    # Sigma^{-1}=[[4,-1],[-1,2]]/7; A=(1,0), C=(0,1).
    # S=4/7-((-1/7)^2)/(2/7)=1/2.
    result = profile_information(
        np.array([[1.0], [0.0]]),
        np.array([[0.0], [1.0]]),
        np.array([[2.0, 1.0], [1.0, 4.0]]),
    )
    np.testing.assert_allclose(result.information, [[0.5]], atol=2e-13)


def test_design_selection_supports_three_goals_nonunit_baselines_and_permutations():
    baselines = {"arrival": 5.0, "stock": 10.0, "flux": 20.0}
    ratios = {
        "alpha": (0.2, 0.8, 0.8),
        "beta": (0.8, 0.2, 0.8),
        "gamma": (0.8, 0.8, 0.2),
        "delta": (0.6, 0.6, 0.6),
        "dominated": (0.8, 0.8, 0.8),
    }
    candidates = [
        DesignCandidate(
            name,
            2.0,
            {goal: ratio * baselines[goal] for goal, ratio in zip(baselines, values)},
        )
        for name, values in ratios.items()
    ]
    selection = select_designs(candidates, baselines, budgets=(0.0, 2.0), mode="exact")
    assert selection[0].winner is None and selection[0].ranking == ()
    chosen = selection[1]
    assert chosen.budget == 2.0 and chosen.mode == "exact"
    assert chosen.winner == "delta"
    assert [rank.identifier for rank in chosen.ranking] == [
        "delta",
        "alpha",
        "beta",
        "gamma",
        "dominated",
    ]
    assert chosen.goal_winners == {"arrival": "alpha", "stock": "beta", "flux": "gamma"}
    assert set(chosen.pareto) == {"alpha", "beta", "gamma", "delta"}
    np.testing.assert_allclose(chosen.ranking[0].maximum, 0.6, atol=1e-14)
    np.testing.assert_allclose(chosen.ranking[0].total, 1.8, atol=1e-14)
    for rank in chosen.ranking:
        np.testing.assert_allclose(
            [rank.normalized[goal] for goal in baselines],
            ratios[rank.identifier],
            atol=1e-14,
        )
    reversed_result = select_designs(
        list(reversed(candidates)),
        dict(reversed(list(baselines.items()))),
        budgets=(2.0,),
        mode="exact",
    )[0]
    assert reversed_result.winner == chosen.winner
    assert [rank.identifier for rank in reversed_result.ranking] == [
        rank.identifier for rank in chosen.ranking
    ]
    assert reversed_result.goal_winners == chosen.goal_winners
    assert set(reversed_result.pareto) == set(chosen.pareto)


def test_budget_modes_and_owned_risk_mappings():
    risks = {"load": 5.0, "delay": 10.0, "energy": 15.0}
    candidate = DesignCandidate("small", 1.0, risks)
    risks["load"] = 500.0
    assert candidate.risks["load"] == 5.0
    baselines = {"load": 5.0, "delay": 5.0, "energy": 5.0}
    other = DesignCandidate("large", 2.0, {"load": 1.0, "delay": 1.0, "energy": 1.0})
    exact = select_designs([candidate, other], baselines, budgets=(1.5,), mode="exact")[0]
    at_most = select_designs([candidate, other], baselines, budgets=(1.5,), mode="at_most")[0]
    assert exact.winner is None and exact.ranking == ()
    assert at_most.winner == "small"
    np.testing.assert_allclose(at_most.ranking[0].maximum, 3.0, atol=1e-14)
    baselines["load"] = 5000.0
    assert at_most.ranking[0].normalized["load"] == 1.0


def test_selection_rejects_normalization_that_erases_strict_risk_order():
    candidates = [
        DesignCandidate("a", 1.0, {"state": 2e-200}),
        DesignCandidate("z", 1.0, {"state": 1e-200}),
    ]
    with pytest.raises(ValueError, match="positive normalized risk underflows"):
        select_designs(candidates, {"state": 1e200}, budgets=(1.0,))
    selection = select_designs(candidates, {"state": 1.0}, budgets=(1.0,))[0]
    assert selection.winner == "z"
    assert selection.goal_winners == {"state": "z"}
    assert selection.pareto == ("z",)
    assert [row.identifier for row in selection.ranking] == ["z", "a"]


def test_three_mask_table_has_exact_risks_and_a_greedy_counterexample():
    rows = {"A": [2.0, 0.0], "B": [1.0, np.sqrt(2.0)], "C": [1.0, -np.sqrt(2.0)]}
    expected = {
        "A": 6.0 / 5.0,
        "B": 5.0 / 4.0,
        "C": 5.0 / 4.0,
        "AB": 9.0 / 16.0,
        "AC": 9.0 / 16.0,
        "BC": 8.0 / 15.0,
        "ABC": 12.0 / 35.0,
    }
    candidates = []
    for size in (1, 2, 3):
        for names in combinations(rows, size):
            name = "".join(names)
            observation = np.array([rows[key] for key in names])
            estimator = np.linalg.solve(np.eye(2) + observation.T @ observation, observation.T)
            risk = affine_goal_risk(
                estimator,
                np.zeros(2),
                observation,
                np.eye(2),
                np.eye(2),
                np.eye(size),
                np.eye(2),
                np.zeros((size, 1)),
                CovarianceFactor(np.empty((1, 0))),
                0.0,
            )
            # Для этой линейной задачи риск равен tr((I+O^T O)^{-1});
            # семь значений выше получены независимыми дробными вычислениями.
            np.testing.assert_allclose(risk.value_estimate, expected[name], atol=2e-13)
            candidates.append(DesignCandidate(name, size, {"state": risk.value_estimate}))
    first, second = select_designs(candidates, {"state": 2.0}, budgets=(1, 2), mode="exact")
    assert first.winner == "A" and second.winner == "BC"
    greedy_extensions = [
        candidate
        for candidate in candidates
        if len(candidate.identifier) == 2 and first.winner in candidate.identifier
    ]
    greedy = select_designs(greedy_extensions, {"state": 2.0}, budgets=(2,), mode="exact")[0]
    assert greedy.winner in {"AB", "AC"}
    assert greedy.ranking[0].maximum > second.ranking[0].maximum


@pytest.mark.parametrize("bad", [True, -1.0, np.nan, np.inf, "1"])
def test_robust_maximum_rejects_invalid_radius(bad):
    with pytest.raises((TypeError, ValueError)):
        robust_ball_maximum(np.ones(1), np.ones((1, 1)), bad)


@pytest.mark.parametrize("options", [{"maxiter": 0}, {"maxiter": True}, {"rtol": -1.0}])
def test_robust_maximum_rejects_invalid_numerical_controls(options):
    with pytest.raises((TypeError, ValueError)):
        robust_ball_maximum(np.ones(1), np.ones((1, 1)), 1.0, **options)


@pytest.mark.parametrize(
    "a,matrix",
    [
        (np.ones((1, 1)), np.ones((1, 1))),
        (np.ones(2), np.ones((1, 1))),
        (np.array([np.nan]), np.ones((1, 1))),
        (np.ones(1), np.array([[np.inf]])),
        (np.array([1j]), np.ones((1, 1))),
        (np.ones(1), np.array([[True]])),
    ],
)
def test_robust_maximum_rejects_invalid_array_contracts(a, matrix):
    with pytest.raises((TypeError, ValueError)):
        robust_ball_maximum(a, matrix, 1.0)


@pytest.mark.parametrize("bad", [np.ones(2), np.array([[np.nan]]), np.array([[1j]]), [[True]]])
def test_covariance_factor_rejects_nonmatrix_or_nonreal_values(bad):
    with pytest.raises((TypeError, ValueError)):
        CovarianceFactor(bad)


@pytest.mark.parametrize(
    "name,bad",
    [
        ("Cxi", np.diag([1.0, -1e-4])),
        ("Ce", np.array([[1.0, 0.3], [0.0, 1.0]])),
        ("Ceta", np.diag([1.0, -1e-4])),
        ("offset", np.zeros((2, 1))),
        ("L", np.ones((1, 3))),
        ("E", np.array([[1.0, np.inf], [0.0, 1.0]])),
        ("O", np.ones((3, 2))),
        ("rho", np.nan),
    ],
)
def test_affine_risk_rejects_invalid_covariances_shapes_and_values(name, bad):
    case = affine_case()
    case[name] = bad
    with pytest.raises((TypeError, ValueError)):
        affine_goal_risk(**case)


def test_affine_risk_rejects_nonfinite_intermediate_arithmetic():
    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        affine_goal_risk(
            np.array([[1e308]]),
            np.zeros(1),
            np.array([[1e308]]),
            np.ones((1, 1)),
            np.ones(1),
            np.ones(1),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.ones(1),
            1.0,
        )


@pytest.mark.parametrize("covariance", [np.diag([1.0, 0.0]), np.diag([1.0, -1.0])])
def test_profile_requires_positive_definite_working_covariance(covariance):
    with pytest.raises((TypeError, ValueError)):
        profile_information(np.ones((2, 1)), np.ones((2, 1)), covariance)


@pytest.mark.parametrize("baseline", [0.0, -1.0, np.nan, np.inf])
def test_selection_rejects_invalid_baselines(baseline):
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"g": 1.0})], {"g": baseline}, budgets=(1,))


def test_selection_rejects_duplicate_ids_goal_mismatch_and_nonfinite_inputs():
    with pytest.raises((TypeError, ValueError)):
        select_designs(
            [
                DesignCandidate("same", 1, {"g": 1.0}),
                DesignCandidate("same", 2, {"g": 2.0}),
            ],
            {"g": 5.0},
            budgets=(2,),
        )
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"other": 1.0})], {"g": 5.0}, budgets=(1,))
    with pytest.raises((TypeError, ValueError)):
        select_designs(
            [
                DesignCandidate("one", 1, {"g": 1.0}),
                DesignCandidate("two", 2, {"other": 1.0}),
            ],
            {"g": 5.0},
            budgets=(2,),
        )
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", np.inf, {"g": 1.0})], {"g": 5.0}, budgets=(1,))
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"g": np.nan})], {"g": 5.0}, budgets=(1,))
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"g": 1.0})], {"g": 5.0}, budgets=(np.inf,))


def test_no_data_affine_baseline_is_five():
    result = affine_goal_risk(
        np.empty((1, 0)),
        np.array([1.0]),
        np.empty((0, 1)),
        np.array([[2.0]]),
        np.ones(1),
        np.empty((0, 0)),
        np.ones((1, 1)),
        np.empty((0, 0)),
        np.empty((0, 0)),
        1.0,
    )
    # Без наблюдений остаются Var(-2*xi)=4 и квадрат смещения 1.
    np.testing.assert_allclose(result.parameter_variance, 4.0, atol=1e-13)
    assert result.noise_variance == 0.0
    np.testing.assert_allclose(result.robust.attained_estimate, 1.0, atol=1e-13)
    np.testing.assert_allclose(result.value_estimate, 5.0, atol=1e-13)
    np.testing.assert_allclose(result.upper_estimate, 5.0, atol=1e-13)
    assert result.converged_estimate


def test_model_error_family_has_unbounded_greedy_ratio_as_penalty_vanishes():
    penalty = 1e-3
    full_observation = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    full_model_direction = np.array([[0.0], [1.0], [0.0]])
    candidates = []
    for size in (1, 2, 3):
        for indices in combinations(range(3), size):
            rows = list(indices)
            observation = full_observation[rows]
            estimator = np.linalg.solve(
                observation.T @ observation + penalty * np.eye(2), observation.T
            )
            risks = {}
            for coordinate, goal in enumerate(("first", "second")):
                risk = affine_goal_risk(
                    estimator,
                    np.zeros(2),
                    observation,
                    np.eye(2),
                    np.eye(2),
                    penalty**2 * np.eye(size),
                    np.eye(2)[coordinate : coordinate + 1],
                    full_model_direction[rows],
                    np.ones(1),
                    1.0,
                )
                assert risk.converged_estimate
                risks[goal] = risk.value_estimate
            identifier = "".join(str(index + 1) for index in indices)
            candidates.append(DesignCandidate(identifier, size, risks))

    assert {candidate.identifier for candidate in candidates} == {
        "1",
        "2",
        "3",
        "12",
        "13",
        "23",
        "123",
    }
    baseline = {"first": 1.0, "second": 1.0}
    singleton, pair = select_designs(candidates, baseline, budgets=(1, 2), mode="exact")
    assert singleton.winner == "2"
    assert pair.winner == "13"
    expected_optimal = 2.0 * penalty**2 / (1.0 + penalty) ** 2
    np.testing.assert_allclose(pair.ranking[0].maximum, expected_optimal, rtol=1e-10, atol=1e-15)
    optimum = next(candidate for candidate in candidates if candidate.identifier == "13")
    np.testing.assert_allclose(
        list(optimum.risks.values()),
        [expected_optimal, expected_optimal],
        rtol=1e-10,
        atol=1e-15,
    )
    # Начав с лучшего одиночного варианта 2, добавление без удаления
    # может дать только 12 или 23. Их модельное смещение не исчезает при lambda->0.
    extensions = [
        candidate
        for candidate in candidates
        if len(candidate.identifier) == 2 and "2" in candidate.identifier
    ]
    assert {candidate.identifier for candidate in extensions} == {"12", "23"}
    greedy = select_designs(extensions, baseline, budgets=(2,), mode="exact")[0]
    assert greedy.winner in {"12", "23"}
    assert greedy.ranking[0].maximum / pair.ranking[0].maximum > 1000.0


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_robust_maximum_rejects_underflow_or_overflow_of_nonzero_objective(magnitude):
    with pytest.raises(ValueError):
        robust_ball_maximum(np.array([magnitude]), np.zeros((1, 0)), 0.0)


def test_smallest_subnormal_noise_covariance_preserves_scaled_variance():
    tiny = np.nextafter(0.0, 1.0)
    covariances = (
        np.array([[tiny]]),
        np.array([tiny]),
        CovarianceFactor(np.array([[np.sqrt(tiny)]])),
    )
    results = [
        affine_goal_risk(
            np.array([[1e162]]),
            np.zeros(1),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros(1),
            covariance,
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.zeros(1),
            1.0,
        )
        for covariance in covariances
    ]
    for result in results:
        assert result.noise_variance > 1.0
        assert result.parameter_variance == 0.0
        assert result.robust.attained_estimate == 0.0
        np.testing.assert_allclose(result.value_estimate, result.noise_variance, rtol=1e-14)
        np.testing.assert_allclose(
            result.noise_variance, results[0].noise_variance, rtol=2e-14, atol=0.0
        )


def test_model_factor_chain_preserves_representable_risk_after_radius_scaling():
    result = affine_goal_risk(
        np.array([[1e-200]]),
        np.zeros(1),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros(1),
        np.zeros(1),
        np.ones((1, 1)),
        np.array([[1e-200]]),
        np.array([1e300]),
        1e250,
    )
    # B*sqrt(Ceta)=1e-50, E*(B*sqrt(Ceta))=1e-250;
    # умножение на радиус даёт 1. Раннее формирование E*B потеряло бы этот вклад.
    assert result.parameter_variance == result.noise_variance == 0.0
    np.testing.assert_allclose(result.value_estimate, 1.0, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(result.upper_estimate, 1.0, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(abs(result.robust.witness[0] / 1e250), 1.0, rtol=2e-12, atol=1e-13)
    assert result.converged_estimate


def test_hard_case_does_not_claim_unattainable_relative_tolerance():
    result = robust_ball_maximum(np.array([0.0, 1.0]), np.diag([2.0, 1.0]), 1.0, rtol=1e-30)
    assert not result.converged_estimate
    assert result.status in {"resolution_limit_estimate", "numerical_inconsistency"}
    assert result.certified is False
    np.testing.assert_allclose(result.attained_estimate, 16.0 / 3.0, atol=2e-12)


def test_finite_design_descent_uses_all_three_moves_and_certifies_only_the_final_frontier():
    optional = ("a", "b", "c", "d", "e")
    pool, mandatory = frozenset(("k", *optional)), frozenset({"k"})
    route = [
        optional,
        ("a", "b", "c", "d"),
        ("a", "b", "c"),
        ("a", "b"),
        ("a", "d"),
        ("a", "d", "e"),
    ]
    expected_path = tuple(frozenset(("k", *items)) for items in route)
    landscape = {candidate: float(10 - index) for index, candidate in enumerate(expected_path)}
    global_better = frozenset({"k", "c"})
    landscape[global_better] = 1.0
    evaluated, checkpoints = [], []

    def evaluate(candidate):
        evaluated.append(candidate)
        value = landscape.get(candidate, 100.0)
        return {"source:S": value, "coefficients:J": value / 2}

    result = finite_design_descent(
        reversed(tuple(sorted(pool))),
        mandatory,
        evaluate,
        admissible=lambda candidate: len(candidate) > 1,
        on_evaluation=lambda candidate, record: checkpoints.append((candidate, record)),
    )
    assert isinstance(result, FiniteDesignResult)
    assert result.path == expected_path
    assert result.selected == expected_path[-1]
    assert result.status == "local_minimum"
    assert len(evaluated) == len(set(evaluated)) == len(checkpoints)
    assert set(evaluated) == set(result.evaluations)
    feasible = [
        frozenset(("k", *items)) for size in range(1, 6) for items in combinations(optional, size)
    ]
    # Independent enumeration by symmetric-difference sizes, rather than by moves.
    frontier = {
        candidate
        for candidate in feasible
        if (len(result.selected - candidate), len(candidate - result.selected))
        in {(1, 0), (0, 1), (1, 1)}
    }
    assert set(result.frontier) == frontier
    assert not result.unresolved_frontier
    assert all(max(result.evaluations[candidate].values()) >= 5 for candidate in frontier)
    assert landscape[global_better] < max(result.evaluations[result.selected].values())


@pytest.mark.parametrize("tie", ["sum", "identifier"])
def test_finite_design_descent_breaks_improving_ties_deterministically(tie):
    pool, mandatory = frozenset({"k", "a", "b"}), frozenset({"k"})
    left, right = frozenset({"k", "a"}), frozenset({"k", "b"})
    risks = {
        pool: {"g1": 5.0, "g2": 4.0},
        left: {"g1": 4.0, "g2": 1.0},
        right: {"g1": 4.0, "g2": 0.0 if tie == "sum" else 1.0},
    }
    expected = right if tie == "sum" else left
    for order in (("a", "k", "b"), ("b", "k", "a")):
        result = finite_design_descent(
            order, mandatory, risks.__getitem__, admissible=lambda candidate: len(candidate) > 1
        )
        assert result.path == (pool, expected)
        assert result.status == "local_minimum"


def test_finite_design_descent_does_not_cross_a_primary_risk_plateau_for_a_smaller_sum():
    pool = frozenset({"k", "a", "b"})
    risks = {
        pool: {"q": 4.0, "theta": 4.0},
        frozenset({"k", "a"}): {"q": 4.0, "theta": 0.0},
        frozenset({"k", "b"}): {"q": 4.0, "theta": 1.0},
    }
    result = finite_design_descent(
        pool, {"k"}, risks.__getitem__, admissible=lambda candidate: len(candidate) > 1
    )
    assert result.selected == pool
    assert result.path == (pool,)
    assert result.status == "local_minimum"


def test_finite_design_descent_preserves_unresolved_risks_and_resumes_from_the_saved_current():
    pool, left, right = map(frozenset, ({"k", "a", "b"}, {"k", "a"}, {"k", "b"}))
    values = {
        pool: {"g": 3.0},
        left: {"g": 2.0},
        right: UnresolvedDesign("ordinary derivative is not established"),
    }
    result = finite_design_descent(
        pool, {"k"}, values.__getitem__, admissible=lambda candidate: len(candidate) > 1
    )
    assert result.selected == left and result.path == (pool, left)
    assert result.status == "incomplete" and result.unresolved_frontier == (right,)
    assert isinstance(result.evaluations[right], UnresolvedDesign)

    def unexpected(candidate):
        pytest.fail(f"A cached result was evaluated again: {candidate}")

    repeated = finite_design_descent(
        pool,
        {"k"},
        unexpected,
        admissible=lambda candidate: len(candidate) > 1,
        cache=result.evaluations,
        resume_from=left,
    )
    assert repeated.path == (left,) and repeated.status == "incomplete"
    completed = dict(result.evaluations)
    completed[right] = {"g": 1.0}
    resumed = finite_design_descent(
        pool,
        {"k"},
        unexpected,
        admissible=lambda candidate: len(candidate) > 1,
        cache=completed,
        resume_from=left,
    )
    assert resumed.path == (left, right)
    assert resumed.selected == right and resumed.status == "local_minimum"
    assert not resumed.unresolved_frontier
    assert isinstance(result.evaluations[right], UnresolvedDesign)


def test_finite_design_descent_does_not_step_from_an_unresolved_start():
    pool = frozenset({"k", "a", "b"})
    evaluated = []

    def evaluate(candidate):
        evaluated.append(candidate)
        return UnresolvedDesign("No stationary starting branch")

    result = finite_design_descent(pool, {"k"}, evaluate)
    assert evaluated == [pool]
    assert result.status == "unresolved_start" and result.selected is None
    assert result.path == (pool,) and not result.frontier


def test_finite_design_descent_checkpoints_validated_results_and_reuses_them_after_interruption():
    pool, left, right = map(frozenset, ({"k", "a", "b"}, {"k", "a"}, {"k", "b"}))
    risks = {pool: {"g": 3.0}, left: {"g": 2.0}, right: {"g": 1.0}}
    saved, evaluated = {}, []

    def evaluate(candidate):
        evaluated.append(candidate)
        return risks[candidate]

    def checkpoint(candidate, record):
        saved[candidate] = record
        with pytest.raises(TypeError):
            record["g"] = 100.0
        if len(saved) == 2:
            raise InterruptedError("caller interruption")

    with pytest.raises(InterruptedError, match="caller interruption"):
        finite_design_descent(
            pool,
            {"k"},
            evaluate,
            admissible=lambda candidate: len(candidate) > 1,
            on_evaluation=checkpoint,
        )
    assert set(saved) == {pool, left}
    resumed = finite_design_descent(
        pool, {"k"}, evaluate, admissible=lambda candidate: len(candidate) > 1, cache=saved
    )
    assert resumed.selected == right and resumed.status == "local_minimum"
    assert len(evaluated) == len(set(evaluated)) == 3
    risks[right]["g"] = 100.0
    assert resumed.evaluations[right]["g"] == 1.0
    with pytest.raises(TypeError):
        resumed.evaluations[right]["g"] = 2.0
    with pytest.raises(TypeError):
        resumed.evaluations[pool] = None
    with pytest.raises(FrozenInstanceError):
        resumed.status = "global_minimum"


def test_finite_design_descent_batches_only_uncached_neighbors_in_checkpoint_order():
    pool = frozenset({"k", "a", "b", "c"})
    left, middle, right = (pool - {name} for name in ("c", "b", "a"))
    small = frozenset({"k", "c"})
    risks = {
        pool: {"q": 6.0, "theta": 5.0},
        left: {"q": 3.0, "theta": 1.0},
        middle: UnresolvedDesign("no stationary derivative"),
        right: {"q": 3.0, "theta": 0.0},
        small: {"q": 2.0, "theta": 0.0},
    }
    cache = {left: risks[left]}
    checkpoints, batches = [], []
    offered = [pool]

    def admissible(candidate):
        return candidate in risks

    def serial(candidate):
        assert candidate == pool
        return risks[candidate]

    def batch(candidates):
        batches.append(candidates)
        for candidate in candidates:
            assert checkpoints == offered
            offered.append(candidate)
            yield risks[candidate]

    expected = finite_design_descent(
        pool, {"k"}, risks.__getitem__, admissible=admissible, cache=cache
    )
    result = finite_design_descent(
        pool,
        {"k"},
        serial,
        admissible=admissible,
        cache=cache,
        on_evaluation=lambda candidate, _: checkpoints.append(candidate),
        evaluate_many=batch,
    )
    assert result == expected
    assert result.path == (pool, right, small)
    assert result.status == "incomplete" and result.unresolved_frontier == (middle,)
    assert batches == [(middle, right), (small,)]
    assert checkpoints == [pool, middle, right, small]


@pytest.mark.parametrize("batch", [False, True])
def test_finite_design_descent_screens_equal_risk_and_reuses_the_bound(batch):
    pool, preferred, excluded, smaller = map(frozenset, ("kab", "ka", "kb", "k"))
    risks = {pool: {"g": 5.0}, preferred: {"g": 3.0}, excluded: {"g": 5.0}, smaller: {"g": 2.0}}
    evaluated, pruned, calls = [], [], []

    def evaluate(candidate):
        assert candidate == pool
        return risks[candidate]

    def get_neighbor(candidate, cutoff):
        calls.append((candidate, cutoff))
        if candidate == excluded:
            return CertifiedDesignBound(5.0)
        return risks[candidate]

    def evaluate_batch(candidates, cutoff):
        return (get_neighbor(candidate, cutoff) for candidate in candidates)

    mode = (
        {"evaluate_many_with_cutoff": evaluate_batch}
        if batch
        else {"evaluate_with_cutoff": get_neighbor}
    )
    result = finite_design_descent(
        pool,
        {"k"},
        evaluate,
        on_evaluation=lambda candidate, _: evaluated.append(candidate),
        on_pruning=lambda candidate, bound: pruned.append((candidate, bound)),
        **mode,
    )
    reference = finite_design_descent(pool, {"k"}, risks.__getitem__)
    assert result.path == reference.path == (pool, preferred, smaller)
    assert result.frontier == reference.frontier and result.status == "local_minimum"
    assert result.pruned_frontier == (excluded,) and not result.unresolved_frontier
    assert result.pruned == {excluded: 5.0} and pruned == [(excluded, 5.0)]
    assert evaluated == [pool, preferred, smaller] and excluded not in result.evaluations
    assert calls == [(preferred, 5.0), (excluded, 5.0), (smaller, 3.0)]


def test_finite_design_descent_persists_screening_separately_and_resumes():
    pool, improving, excluded = map(frozenset, ("kab", "ka", "kb"))
    evaluations = {pool: {"g": 5.0}}
    bounds = {}

    def cutoff(candidate, incumbent):
        assert incumbent == 5.0
        return {"g": 4.0} if candidate == improving else CertifiedDesignBound(6.0)

    def checkpoint(candidate, bound):
        bounds[candidate] = bound
        raise InterruptedError("save before interruption")

    with pytest.raises(InterruptedError):
        finite_design_descent(
            pool,
            {"k"},
            pytest.fail,
            admissible=lambda candidate: len(candidate) > 1,
            cache=evaluations,
            evaluate_with_cutoff=cutoff,
            on_evaluation=evaluations.__setitem__,
            on_pruning=checkpoint,
        )
    assert evaluations == {pool: {"g": 5.0}, improving: {"g": 4.0}}
    assert bounds == {excluded: 6.0}
    reused = {}
    resumed = finite_design_descent(
        pool,
        {"k"},
        pytest.fail,
        admissible=lambda candidate: len(candidate) > 1,
        cache=evaluations,
        resume_from=improving,
        bound_cache=bounds,
        evaluate_with_cutoff=pytest.fail,
        on_pruning=reused.__setitem__,
    )
    assert resumed.path == (improving,)
    assert resumed.status == "local_minimum"
    assert resumed.pruned_frontier == (excluded,) and reused == bounds


@pytest.mark.parametrize("tied_total", [1.0, 2.0])
def test_streamed_design_tightens_only_unsent_jobs_and_preserves_ties(tied_total):
    pool = frozenset("kabcd")
    first, second, excluded, tied = (pool - {name} for name in "dcba")
    values = {
        pool: {"g": 8.0, "h": 8.0},
        first: {"g": 3.0, "h": 2.0},
        second: {"g": 5.0, "h": 3.0},
        excluded: {"g": 4.0, "h": 1.0},
        tied: {"g": 3.0, "h": tied_total},
    }
    submitted, archived, bounds = [], {}, {}

    def stream(claim):
        initial = claim()
        if initial is None:
            return
        submitted.extend((initial, claim()))
        assert submitted == [(first, 8.0), (second, 8.0)]
        yield first, values[first]
        submitted.append(claim())
        yield excluded, CertifiedDesignBound(4.0)
        yield second, values[second]
        submitted.append(claim())
        yield tied, values[tied]
        assert claim() is None

    def admissible(candidate):
        return len(candidate) >= 4

    result = finite_design_descent(
        pool,
        {"k"},
        values.__getitem__,
        admissible=admissible,
        evaluate_stream_with_cutoff=stream,
        on_evaluation=archived.__setitem__,
        on_pruning=bounds.__setitem__,
    )
    cutoff = np.nextafter(3.0, np.inf)
    assert submitted[2:] == [(excluded, cutoff), (tied, cutoff)]
    expected = tied if tied_total == 1.0 else first
    assert result.path == (pool, expected) and result.pruned == {excluded: 4.0}
    replayed = finite_design_descent(
        pool, {"k"}, pytest.fail, admissible=admissible, cache=archived, bound_cache=bounds
    )
    assert replayed == result


def test_streamed_design_rejects_a_certificate_below_its_submission_cutoff():
    pool = frozenset("kab")

    def stream(claim):
        first, _ = claim()
        second, _ = claim()
        yield first, {"g": 3.0}
        yield second, CertifiedDesignBound(4.0)

    with pytest.raises(ValueError, match="bound below the incumbent"):
        finite_design_descent(
            pool, {"k"}, lambda _: {"g": 5.0}, evaluate_stream_with_cutoff=stream
        )


def test_finite_design_descent_keeps_an_unresolved_score_distinct_from_a_bound():
    pool, unknown, excluded = map(frozenset, ("kab", "ka", "kb"))
    cache = {pool: {"g": 5.0}, unknown: UnresolvedDesign("undefined derivative")}
    result = finite_design_descent(
        pool,
        {"k"},
        pytest.fail,
        cache=cache,
        bound_cache={unknown: 5.0, excluded: 5.0},
        evaluate_with_cutoff=pytest.fail,
    )
    assert result.status == "incomplete"
    assert result.pruned_frontier == (excluded,)
    assert result.unresolved_frontier == (unknown,)


@pytest.mark.parametrize("bound", [4.0, -1.0, np.nan, np.inf, True])
def test_finite_design_descent_rejects_an_invalid_screening_certificate(bound):
    pool = frozenset({"k", "a"})
    with pytest.raises(ValueError, match="design.*bound"):
        finite_design_descent(
            pool,
            {"k"},
            lambda _: {"g": 5.0},
            evaluate_with_cutoff=lambda *args: CertifiedDesignBound(bound),
            on_pruning=pytest.fail,
        )


def test_finite_design_descent_checks_below_cutoff_bound_against_full_score():
    pool, neighbor = map(frozenset, ("ka", "k"))
    risks = {pool: {"g": 5.0}, neighbor: {"g": 4.0}}
    result = finite_design_descent(pool, {"k"}, risks.__getitem__, bound_cache={neighbor: 3.0})
    assert result.path == (pool, neighbor) and not result.pruned
    for cache in ({pool: risks[pool]}, risks):
        with pytest.raises(ValueError, match="contradicts its lower bound"):
            finite_design_descent(
                pool, {"k"}, risks.__getitem__, cache=cache, bound_cache={neighbor: 4.5}
            )


@pytest.mark.parametrize("length", [0, 2])
def test_finite_design_descent_rejects_batch_result_with_wrong_length(length):
    pool, neighbor = frozenset({"k", "a"}), frozenset({"k"})
    checkpoints = []
    with pytest.raises(ValueError, match="zip"):
        finite_design_descent(
            pool,
            {"k"},
            lambda candidate: {"q": 2.0},
            evaluate_many=lambda candidates: [{"q": 1.0}] * length,
            on_evaluation=lambda candidate, _: checkpoints.append(candidate),
        )
    assert checkpoints == ([pool] if length == 0 else [pool, neighbor])


def test_finite_design_descent_uses_exact_set_cache_keys_without_display_identifier_collisions():
    pool = frozenset({"k", "a", "b", "a|b"})
    high, low = frozenset({"k", "a", "b"}), frozenset({"k", "a|b"})
    assert "|".join(sorted(high)) == "|".join(sorted(low))
    cache = {high: {"g": 9.9}, low: {"g": 1.0}}
    evaluated = []

    def evaluate(candidate):
        evaluated.append(candidate)
        return {"g": 9.0 if candidate == frozenset({"k", "b", "a|b"}) else 10.0}

    result = finite_design_descent(
        pool, {"k"}, evaluate, admissible=lambda candidate: len(candidate) > 1, cache=cache
    )
    assert result.selected == low and result.status == "local_minimum"
    assert result.evaluations[high]["g"] == 9.9 and result.evaluations[low]["g"] == 1.0
    assert high not in evaluated and low not in evaluated
    assert cache == {high: {"g": 9.9}, low: {"g": 1.0}}


def test_finite_design_descent_rejects_a_changing_admissibility_rule():
    pool, left, right = map(frozenset, ({"k", "a", "b"}, {"k", "a"}, {"k", "b"}))
    invalidated = set()

    def admissible(candidate):
        return len(candidate) > 1 and candidate not in invalidated

    def evaluate(candidate):
        if candidate == left:
            invalidated.add(right)
        return {"g": 3.0 if candidate == pool else 2.0}

    with pytest.raises(ValueError, match="admissibility changed"):
        finite_design_descent(pool, {"k"}, evaluate, admissible=admissible)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"g": -1.0},
        {"g": np.inf},
        {"g": np.ma.array(1.0, mask=True)},
        {"g": 1e308, "h": 1e308},
    ],
)
def test_finite_design_descent_does_not_checkpoint_invalid_risks_as_unresolved(bad):
    saved = []
    with pytest.raises((TypeError, ValueError)):
        finite_design_descent(
            {"k", "a"},
            {"k"},
            lambda candidate: bad,
            on_evaluation=lambda *args: saved.append(args),
        )
    assert not saved


def test_finite_design_descent_keeps_goal_identity_and_propagates_evaluator_failures():
    pool = frozenset({"k", "a", "b"})
    cache = {pool: {"source:S": 3.0}}
    with pytest.raises(ValueError, match="same normalized risk goals"):
        finite_design_descent(
            pool,
            {"k"},
            lambda candidate: {"coefficients:J": 1.0},
            admissible=lambda candidate: len(candidate) > 1,
            cache=cache,
        )
    assert cache == {pool: {"source:S": 3.0}}

    def failed(candidate):
        raise RuntimeError("external evaluation failed")

    with pytest.raises(RuntimeError, match="external evaluation failed"):
        finite_design_descent(pool, {"k"}, failed)


@pytest.mark.parametrize(
    "defect", ["mandatory", "initial", "cache_key", "cache_inadmissible", "resume_unresolved"]
)
def test_finite_design_descent_requires_a_valid_frozen_domain_and_resolved_resume(defect):
    pool, mandatory, kwargs = frozenset({"k", "a"}), {"k"}, {}
    if defect == "mandatory":
        mandatory = {"outside"}
    elif defect == "initial":
        kwargs["admissible"] = lambda candidate: False
    elif defect == "cache_key":
        kwargs["cache"] = {("k", "a"): {"g": 1.0}}
    elif defect == "cache_inadmissible":
        kwargs.update(
            cache={frozenset({"k"}): {"g": 1.0}}, admissible=lambda candidate: len(candidate) > 1
        )
    else:
        kwargs.update(cache={pool: UnresolvedDesign("not assessed")}, resume_from=pool)
    with pytest.raises(ValueError):
        finite_design_descent(
            pool,
            mandatory,
            lambda candidate: pytest.fail("Invalid domain reached evaluator"),
            **kwargs,
        )
