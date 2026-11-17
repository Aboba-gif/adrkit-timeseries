"""MAD и AR(1): точные ветви, вероятностные моменты и временные интервалы."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from timeseries.statistics import (
    fit_ar1_reference,
    mad_sensitivity,
    median_absolute_deviation,
    mad_of_differences_sensitivity,
    median_absolute_deviation_of_differences,
)


@pytest.mark.parametrize(
    "values, expected, gradient",
    [
        ([-7.0, -2.0, 1.0, 5.0, 11.0], 4.0, [0.0, 0.0, -1.0, 1.0, 0.0]),
        ([0.0, 1.0, 5.0, 10.0], 2.5, [-0.5, 0.0, 0.5, 0.0]),
    ],
)
def test_ordinary_branch_matches_analytic_gradient_and_both_difference_signs(
    values, expected, gradient
):
    values, gradient = np.array(values), np.array(gradient)
    normalization = 1.5
    result = mad_sensitivity(values, normalization)
    assert result.value == median_absolute_deviation(values, normalization)
    assert result.value == normalization * expected
    assert result.status == "ordinary_certified"
    np.testing.assert_array_equal(result.ordinary_gradient, normalization * gradient)
    direction = np.arange(1, len(values) + 1, dtype=float) ** 2
    slope = normalization * float(gradient @ direction)
    assert result.directional(direction) == slope
    np.testing.assert_array_equal(result.vjp(-2.0), -2.0 * normalization * gradient)
    for step in (-(2.0**-18), 2.0**-18):
        finite_difference = (
            median_absolute_deviation(values + step * direction, normalization) - result.value
        ) / step
        np.testing.assert_allclose(finite_difference, slope, rtol=1e-9, atol=1e-9)


def test_two_distinct_values_certify_structural_absolute_deviation_tie():
    result = mad_sensitivity([-3.0, 5.0])
    assert result.value == 4.0 and result.status == "ordinary_certified"
    np.testing.assert_array_equal(result.ordinary_gradient, [-0.5, 0.5])
    for direction in ([2.0, -4.0], [-2.0, 4.0], [1.0, 1.0]):
        assert result.directional(direction) == 0.5 * (direction[1] - direction[0])


def test_two_equal_values_require_absolute_direction_difference():
    result = mad_sensitivity([2.0, 2.0])
    assert result.value == 0.0 and result.status == "ordinary_uncertified"
    assert result.ordinary_gradient is None
    assert result.directional([1.0, -3.0]) == 2.0
    assert result.directional([-1.0, 3.0]) == 2.0
    with pytest.raises(ValueError, match="not certified"):
        result.vjp(1.0)


def test_three_value_cusp_has_nonadditive_hadamard_derivative():
    # MAD(-1, epsilon, 1) = 1 - abs(epsilon) near zero.
    result = mad_sensitivity([-1.0, 0.0, 1.0])
    positive, negative = np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
    assert result.status == "ordinary_uncertified" and result.ordinary_gradient is None
    assert result.directional(positive) == result.directional(negative) == -1.0
    assert result.directional(positive + negative) == 0.0
    for step in (2.0**-5, 2.0**-15):
        for direction in (positive, negative):
            difference = (
                median_absolute_deviation(np.array([-1.0, 0.0, 1.0]) + step * direction) - 1
            )
            assert difference / step == -1.0


def test_nonlinear_center_cancels_in_the_certified_outer_expression():
    values = np.array([-3.0, -1.0, 0.0, 0.0, 0.0, 2.0, 4.0, 5.0])
    result = mad_sensitivity(values)
    gradient = np.zeros(8)
    gradient[1], gradient[5] = -0.5, 0.5
    assert result.status == "ordinary_certified" and result.value == 1.5
    np.testing.assert_array_equal(result.ordinary_gradient, gradient)
    direction = np.array([2.0, -3.0, 7.0, -1.0, 5.0, 1.0, -2.0, 4.0])
    assert result.directional(direction) == float(gradient @ direction)
    for step in (-(2.0**-16), 2.0**-16):
        difference = median_absolute_deviation(values + step * direction) - result.value
        assert difference / step == float(gradient @ direction)


def test_even_linear_center_does_not_automatically_certify_mad():
    result = mad_sensitivity([-3.0, 0.0, 0.0, 5.0])
    # The raw median is the average of the middle pair; MAD still has a cusp.
    direction = np.array([0.0, 1.0, -1.0, 0.0])
    assert result.value == 1.5 and result.status == "ordinary_uncertified"
    assert result.directional(direction) == result.directional(-direction) == 0.5


@pytest.mark.parametrize("value", [-4.0, 0.0, 7.0])
def test_single_value_has_identically_zero_mad_and_gradient(value):
    result = mad_sensitivity([value], normalization=3.0)
    assert result.value == 0.0 and result.status == "ordinary_certified"
    np.testing.assert_array_equal(result.ordinary_gradient, [0.0])
    assert result.directional([-1e308]) == 0.0
    np.testing.assert_array_equal(result.vjp(1e308), [0.0])


def test_duplicate_values_use_directional_absolute_values_not_sign_zero():
    result = mad_sensitivity([4.0, 4.0, 4.0])
    direction = np.array([-2.0, 0.0, 3.0])
    assert result.status == "ordinary_uncertified"
    assert result.directional(direction) == result.directional(-direction) == 2.0


@pytest.mark.parametrize("values", [[-1.0, 0.0, 1.0], [-3.0, -1.0, 0.0, 0.0, 0.0, 2.0, 4.0, 5.0]])
def test_translation_permutation_and_positive_homogeneity(values):
    values = np.array(values)
    direction = np.arange(len(values), dtype=float) ** 2 - 3
    permutation = np.arange(len(values))[::-1]
    base = mad_sensitivity(values)
    moved = mad_sensitivity(values + 16)
    permuted = mad_sensitivity(values[permutation])
    scaled = mad_sensitivity(3 * values)
    assert moved.value == permuted.value == base.value
    assert scaled.value == 3 * base.value
    assert moved.directional(direction) == base.directional(direction)
    assert permuted.directional(direction[permutation]) == base.directional(direction)
    assert scaled.directional(3 * direction) == 3 * base.directional(direction)
    assert base.directional(4 * direction) == 4 * base.directional(direction)
    if base.ordinary_gradient is not None:
        np.testing.assert_array_equal(moved.ordinary_gradient, base.ordinary_gradient)
        np.testing.assert_array_equal(
            permuted.ordinary_gradient, base.ordinary_gradient[permutation]
        )


def test_rounded_midpoint_must_not_create_a_false_ordinary_certificate():
    values = 1e16 + np.array([0.0, 2.0, 4.0, 6.0])
    result = mad_sensitivity(values)
    reference = mad_sensitivity([0.0, 2.0, 4.0, 6.0])
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    assert result.value == reference.value == 2.0
    assert result.status == reference.status == "ordinary_uncertified"
    assert result.directional(direction) == reference.directional(direction) == -0.5
    assert result.directional(-direction) == 0.0


def test_finite_extremes_do_not_overflow_unselected_absolute_deviations():
    largest = np.finfo(float).max
    result = mad_sensitivity([-largest, -largest, largest])
    assert result.value == 0.0
    assert result.directional([-1.0, 1.0, 0.0]) == 2.0


def test_unrepresentable_output_is_not_clipped():
    largest = np.finfo(float).max
    for function in (median_absolute_deviation, mad_sensitivity):
        with pytest.raises(FloatingPointError, match="not representable"):
            function([-largest, largest], normalization=largest)


def test_nonzero_mad_gradient_direction_and_vjp_must_not_underflow_to_zero():
    smallest = np.nextafter(0.0, 1.0)
    for function in (median_absolute_deviation, mad_sensitivity):
        with pytest.raises(FloatingPointError, match="not representable"):
            function([0.0, smallest])
    # The MAD value is representable, but its ordinary gradient is not.
    assert median_absolute_deviation([0.0, 2.0], normalization=smallest) == smallest
    with pytest.raises(FloatingPointError, match="not representable"):
        mad_sensitivity([0.0, 2.0], normalization=smallest)
    ordinary = mad_sensitivity([0.0, 2.0])
    with pytest.raises(FloatingPointError, match="not representable"):
        ordinary.vjp(smallest)
    directional = mad_sensitivity([-1.0, 0.0, 1.0], normalization=smallest)
    with pytest.raises(FloatingPointError, match="not representable"):
        directional.directional([0.0, 0.5, 0.0])


def test_masked_scalars_cannot_supply_normalization_or_cotangent():
    for mask in (False, True):
        scalar = np.ma.array(1.0, mask=mask)
        for function in (median_absolute_deviation, mad_sensitivity):
            with pytest.raises(ValueError, match="unmasked"):
                function([0.0, 2.0], normalization=scalar)
        with pytest.raises(ValueError, match="unmasked"):
            mad_sensitivity([0.0, 2.0]).vjp(scalar)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[1.0]],
        [1.0, np.nan],
        [1.0, np.inf],
        [True, False],
        [1.0 + 1j],
        np.ma.array([1.0, 2.0], mask=[False, True]),
    ],
)
def test_missing_or_invalid_input_is_not_compressed(values):
    for function in (median_absolute_deviation, mad_sensitivity):
        with pytest.raises(ValueError):
            function(values)


@pytest.mark.parametrize("normalization", [0.0, -1.0, np.nan, np.inf, True, 1j, [1.0]])
def test_normalization_is_explicit_positive_and_finite(normalization):
    for function in (median_absolute_deviation, mad_sensitivity):
        with pytest.raises(ValueError):
            function([1.0, 2.0], normalization)


def test_result_owns_immutable_data_and_vjp_requires_a_scalar():
    values = np.array([-3.0, 5.0])
    result = mad_sensitivity(values)
    values[:] = 100.0
    assert result.value == 4.0 and result.directional([1.0, 0.0]) == -0.5
    np.testing.assert_array_equal(result.ordinary_gradient, [-0.5, 0.5])
    with pytest.raises(ValueError):
        result.ordinary_gradient.setflags(write=True)
    with pytest.raises(ValueError):
        result.ordinary_gradient[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        result.value = 0.0
    for cotangent in ([1.0], np.nan, np.inf, True, 1j):
        with pytest.raises(ValueError):
            result.vjp(cotangent)
    for direction in ([1.0], [np.nan, 0.0], [True, False]):
        with pytest.raises(ValueError):
            result.directional(direction)


def test_paired_mad_ordinary_gradient_and_vjp_use_original_values():
    left = np.array([1.0, 2.0, 4.0])
    right = np.array([2.0, 4.0, 8.0])
    result = mad_of_differences_sensitivity(left, right, normalization=2.0)
    assert result.value == median_absolute_deviation_of_differences(left, right, 2.0) == 2.0
    assert result.status == "ordinary_certified"
    expected = np.array([[2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]])
    np.testing.assert_array_equal(result.ordinary_gradient, expected)
    d_left, d_right = np.array([1.0, -1.0, 2.0]), np.array([0.0, 3.0, -2.0])
    assert result.directional(d_left, d_right) == 10.0
    np.testing.assert_array_equal(result.vjp(-3.0), -3.0 * expected)
    left[:], right[:] = 100.0, -100.0
    assert result.value == 2.0 and result.directional(d_left, d_right) == 10.0
    with pytest.raises(ValueError):
        result.ordinary_gradient.setflags(write=True)
    for function in (median_absolute_deviation_of_differences, mad_of_differences_sensitivity):
        with pytest.raises(ValueError):
            function([0.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            function([np.nan], [1.0])


def test_exact_paired_differences_do_not_certify_the_wrong_rounded_branch():
    h, b, normalization = float(2**53), 0.25, 1.482602218505602
    values = np.array([3 * b, b, h, 3 * h])
    direction = np.array([0.0, 1.0, 0.0, 0.0])
    result = mad_of_differences_sensitivity(values[:-1], values[1:], normalization)
    expected_value = float(Fraction(normalization) * (Fraction(h) + Fraction(b)))
    assert result.value == expected_value
    assert result.status == "ordinary_uncertified" and result.ordinary_gradient is None
    assert result.directional(direction[:-1], direction[1:]) == -2 * normalization
    assert result.directional(-direction[:-1], -direction[1:]) == -normalization
    with pytest.raises(ValueError, match="not certified"):
        result.vjp(1.0)
    # Rounding right-left first selects a different, smooth MAD branch.
    rounded = mad_sensitivity(np.diff(values), normalization)
    assert rounded.status == "ordinary_certified"
    assert rounded.directional(np.diff(direction)) == normalization


def test_paired_values_and_directions_keep_unrepresentable_intermediate_differences():
    largest = np.finfo(float).max
    left = np.array([-largest, -largest, largest])
    right = np.array([0.0, largest, largest])
    assert median_absolute_deviation_of_differences(left, right) == largest
    zero = mad_of_differences_sensitivity(np.zeros(3), np.zeros(3))
    assert zero.value == 0.0 and zero.directional(left, right) == largest


def test_ar1_independent_reference_posterior_uses_student_variance():
    result = fit_ar1_reference(
        [0.0, 1.0, 2.0, 3.0], [0, 1, 3, 5], rho_grid=[0.0], prior_weights=[1.0]
    )
    # GLS has mean 3/2, residual sum 5, precision 4 and n-1=3 degrees of freedom.
    np.testing.assert_allclose(result.mean_location, [1.5], rtol=1e-14)
    np.testing.assert_allclose(result.mean_precision, [4.0], rtol=1e-14)
    np.testing.assert_allclose(result.residual_sum_squares, [5.0], rtol=1e-14)
    np.testing.assert_allclose(result.variance_mean, [5.0], rtol=1e-14)
    np.testing.assert_array_equal(result.posterior_weights, [1.0])
    assert result.degrees_of_freedom == 3
    mean, covariance = result.predictive_moments([6, 8], relation="replicate")
    np.testing.assert_allclose(mean, [1.5, 1.5], rtol=1e-14)
    np.testing.assert_allclose(covariance, 5.0 * (np.eye(2) + np.ones((2, 2)) / 4), rtol=1e-14)


@pytest.mark.parametrize("rho", [-0.5, 0.5])
def test_ar1_reference_uses_actual_missing_hour_lags_and_signed_correlations(rho):
    positions = np.array([0, 2, 5, 7])
    values = np.array([0.25, -0.5, 1.75, 0.5])
    result = fit_ar1_reference(values, positions, rho_grid=[rho], prior_weights=[1.0])
    # The stationary Markov likelihood gives an independent tridiagonal precision
    # calculation, with phi=rho**gap rather than compressing gaps to one hour.
    phi = rho ** np.diff(positions)
    innovation_variance = 1.0 - phi**2
    precision = 1.0 + np.sum((1.0 - phi) ** 2 / innovation_variance)
    score = values[0] + np.sum(
        (1.0 - phi) * (values[1:] - phi * values[:-1]) / innovation_variance
    )
    location = score / precision
    centered = values - location
    residual_sum = centered[0] ** 2 + np.sum(
        (centered[1:] - phi * centered[:-1]) ** 2 / innovation_variance
    )
    np.testing.assert_allclose(result.mean_precision, [precision], rtol=2e-13)
    np.testing.assert_allclose(result.mean_location, [location], rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(result.residual_sum_squares, [residual_sum], rtol=2e-13)
    np.testing.assert_allclose(result.variance_mean, [residual_sum], rtol=2e-13)
    mean, covariance = result.predictive_moments([10, 13], relation="replicate")
    correlation = np.array([[1.0, rho**3], [rho**3, 1.0]])
    np.testing.assert_allclose(mean, np.full(2, location), rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(
        covariance,
        residual_sum * (correlation + np.ones((2, 2)) / precision),
        rtol=3e-13,
        atol=2e-14,
    )


def test_ar1_continuation_and_independent_replicate_have_distinct_known_moments():
    result = fit_ar1_reference(
        [-1.0, 0.0, 1.0, 2.0],
        [0, 1, 2, 3],
        rho_grid=[0.5],
        prior_weights=[1.0],
    )
    np.testing.assert_allclose(result.mean_location, [0.5], rtol=1e-14)
    np.testing.assert_allclose(result.mean_precision, [2.0], rtol=1e-14)
    np.testing.assert_allclose(result.residual_sum_squares, [31 / 6], rtol=2e-14)
    continued_mean, continued_covariance = result.predictive_moments(
        [4, 6], relation="continuation"
    )
    replicate_mean, replicate_covariance = result.predictive_moments([4, 6], relation="replicate")
    # Future dependence on the observations is through the last observation:
    # A*y = (1/2,1/8)*2 and A*1 = (1/2,1/8).
    np.testing.assert_allclose(continued_mean, [5 / 4, 11 / 16], rtol=2e-14)
    np.testing.assert_allclose(
        continued_covariance,
        (31 / 6) * np.array([[7 / 8, 13 / 32], [13 / 32, 175 / 128]]),
        rtol=3e-14,
    )
    np.testing.assert_allclose(replicate_mean, [0.5, 0.5], rtol=2e-14)
    np.testing.assert_allclose(
        replicate_covariance,
        (31 / 6) * np.array([[1.5, 0.75], [0.75, 1.5]]),
        rtol=3e-14,
    )
    # Unknown-mean predictive covariances have no universal Loewner ordering.
    assert np.linalg.det(replicate_covariance - continued_covariance) < 0.0


def test_ar1_grid_evidence_and_predictive_covariance_include_between_mean_variance():
    result = fit_ar1_reference(
        [0.0, 0.0, 0.0, 4.0],
        [0, 1, 2, 3],
        rho_grid=[0.0, 0.5],
        prior_weights=[1.0, 3.0],
    )
    # For rho=0 and 1/2, respectively: (T,m,Q)=(4,1,12),(2,4/3,160/9).
    evidence = np.array(
        [
            1.0 / (2 * 12.0**1.5),
            3.0 * (3 / 4) ** (-1.5) / (np.sqrt(2) * (160 / 9) ** 1.5),
        ]
    )
    weights = evidence / evidence.sum()
    np.testing.assert_allclose(result.prior_weights, [0.25, 0.75], rtol=1e-14)
    np.testing.assert_allclose(result.posterior_weights, weights, rtol=3e-13)
    np.testing.assert_allclose(result.log_posterior_weights, np.log(weights), rtol=3e-13)
    mean, covariance = result.predictive_moments([4, 6], relation="replicate")
    location = weights @ np.array([1.0, 4 / 3])
    first_covariance = 12 * (np.eye(2) + np.ones((2, 2)) / 4)
    second_covariance = (160 / 9) * (np.array([[1.0, 0.25], [0.25, 1.0]]) + np.ones((2, 2)) / 2)
    within = weights[0] * first_covariance + weights[1] * second_covariance
    between = weights[0] * weights[1] * (1 / 3) ** 2
    np.testing.assert_allclose(mean, np.full(2, location), rtol=3e-13)
    np.testing.assert_allclose(covariance, within + between * np.ones((2, 2)), rtol=3e-13)
    assert between > 0.0


@pytest.mark.parametrize("relation", ["replicate", "continuation"])
def test_ar1_reference_is_equivariant_under_units_and_calendar_origin_changes(relation):
    values = np.array([-1.0, 0.0, 1.0, 4.0, 2.0])
    positions = np.array([0, 1, 3, 4, 7])
    future = np.array([8, 11])
    arguments = {"rho_grid": [-0.4, 0.0, 0.6], "prior_weights": [1.0, 2.0, 3.0]}
    original = fit_ar1_reference(values, positions, **arguments)
    scale, shift = 3.5, -9.0
    converted = fit_ar1_reference(scale * values + shift, positions + 100, **arguments)
    np.testing.assert_allclose(converted.posterior_weights, original.posterior_weights, rtol=5e-13)
    np.testing.assert_allclose(
        converted.mean_location,
        scale * original.mean_location + shift,
        rtol=5e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        converted.variance_mean, scale**2 * original.variance_mean, rtol=5e-13
    )
    mean, covariance = original.predictive_moments(future, relation=relation)
    moved_mean, moved_covariance = converted.predictive_moments(future + 100, relation=relation)
    np.testing.assert_allclose(moved_mean, scale * mean + shift, rtol=5e-13, atol=2e-13)
    np.testing.assert_allclose(moved_covariance, scale**2 * covariance, rtol=8e-13)


def test_ar1_calibration_and_predictive_moments_own_immutable_inputs():
    values, positions = np.array([-1.0, 0.0, 1.0, 3.0]), np.array([0, 1, 3, 5])
    rho, prior = np.array([-0.5, 0.0, 0.5]), np.array([1.0, 2.0, 1.0])
    result = fit_ar1_reference(values, positions, rho_grid=rho, prior_weights=prior)
    names = (
        "values",
        "positions",
        "rho_grid",
        "prior_weights",
        "posterior_weights",
        "log_posterior_weights",
        "mean_location",
        "mean_precision",
        "residual_sum_squares",
        "variance_mean",
    )
    saved = {name: getattr(result, name).copy() for name in names}
    future = np.array([6, 8])
    mean, covariance = result.predictive_moments(future, relation="replicate")
    saved_mean, saved_covariance = mean.copy(), covariance.copy()
    values[:], positions[:], rho[:], prior[:], future[:] = 99, 0, 0, 1, 0
    for name in names:
        array = getattr(result, name)
        np.testing.assert_array_equal(array, saved[name])
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.degrees_of_freedom = 99
    for array in (mean, covariance):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(mean, saved_mean)
    np.testing.assert_array_equal(covariance, saved_covariance)
    repeated = result.predictive_moments([6, 8], relation="replicate")
    np.testing.assert_array_equal(repeated[0], saved_mean)
    np.testing.assert_array_equal(repeated[1], saved_covariance)


@pytest.mark.parametrize("values", [[0.0], [0.0, 1.0], [0.0, 1.0, 2.0]])
def test_ar1_reference_requires_enough_samples_for_finite_predictive_covariance(values):
    with pytest.raises(ValueError):
        fit_ar1_reference(values, np.arange(len(values)), rho_grid=[0.0], prior_weights=[1.0])


@pytest.mark.parametrize("constant", [0.0, 2.5])
def test_ar1_reference_does_not_invent_a_positive_scale_for_constant_data(constant):
    with pytest.raises(ValueError):
        fit_ar1_reference(
            np.full(4, constant), [0, 1, 2, 3], rho_grid=[-0.5, 0.5], prior_weights=[1.0, 1.0]
        )


@pytest.mark.parametrize("positions", [[0, 1, 1, 3], [0, 1, 2.5, 3]])
def test_ar1_reference_rejects_duplicate_or_fractional_hour_coordinates(positions):
    with pytest.raises(ValueError):
        fit_ar1_reference([0.0, 1.0, 2.0, 3.0], positions, rho_grid=[-0.5], prior_weights=[1.0])


@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_ar1_reference_excludes_singular_correlation_endpoints(rho):
    with pytest.raises(ValueError):
        fit_ar1_reference([0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3], rho_grid=[rho], prior_weights=[1.0])


def test_ar1_predictive_relation_is_explicit_and_continuation_cannot_reobserve_training():
    result = fit_ar1_reference(
        [0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3], rho_grid=[0.5], prior_weights=[1.0]
    )
    with pytest.raises(TypeError):
        result.predictive_moments([4, 5])
    with pytest.raises(ValueError):
        result.predictive_moments([4, 5], relation="unknown")
    with pytest.raises(ValueError):
        result.predictive_moments([3, 4], relation="continuation")
    # A new independent trajectory may use the same relative hour coordinates.
    mean, covariance = result.predictive_moments([0, 1], relation="replicate")
    assert mean.shape == (2,) and covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


@pytest.mark.parametrize(
    "rho, prior",
    [([], []), ([0.0, 0.5, 0.0], [1.0, 1.0, 1.0])],
)
def test_ar1_reference_requires_a_nonempty_grid_of_distinct_models(rho, prior):
    with pytest.raises(ValueError):
        fit_ar1_reference([0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3], rho_grid=rho, prior_weights=prior)


@pytest.mark.parametrize("prior", [[0.0, 1.0], [-0.1, 1.0]])
def test_ar1_reference_requires_strictly_positive_model_prior_weights(prior):
    with pytest.raises(ValueError):
        fit_ar1_reference(
            [0.0, 1.0, 2.0, 3.0],
            [0, 1, 2, 3],
            rho_grid=[-0.5, 0.5],
            prior_weights=prior,
        )


@pytest.mark.parametrize(
    "field, value",
    [("values", [0.0, 1.0, 2.0, 3.0, 4.0]), ("prior_weights", [1.0])],
)
def test_ar1_reference_requires_observation_and_model_axes_to_match(field, value):
    arguments = {
        "values": [0.0, 1.0, 2.0, 3.0],
        "positions": [0, 1, 2, 3],
        "rho_grid": [-0.5, 0.5],
        "prior_weights": [1.0, 1.0],
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        fit_ar1_reference(**arguments)


@pytest.mark.parametrize(
    "field, value",
    [
        ("values", [0.0, np.nan, 2.0, 3.0]),
        ("values", np.ma.array([0.0, 1.0, 2.0, 3.0], mask=[False, True, False, False])),
        ("positions", [0, 1, 2, np.inf]),
        ("positions", np.ma.array([0, 1, 2, 3], mask=[False, True, False, False])),
        ("rho_grid", [0.0, np.nan]),
        ("rho_grid", np.ma.array([0.0, 0.5], mask=[False, True])),
        ("prior_weights", [1.0, np.inf]),
        ("prior_weights", np.ma.array([1.0, 1.0], mask=[False, True])),
    ],
)
def test_ar1_reference_does_not_silently_impute_or_drop_unavailable_inputs(field, value):
    arguments = {
        "values": [0.0, 1.0, 2.0, 3.0],
        "positions": [0, 1, 2, 3],
        "rho_grid": [0.0, 0.5],
        "prior_weights": [1.0, 1.0],
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        fit_ar1_reference(**arguments)


@pytest.mark.parametrize("relation", ["replicate", "continuation"])
def test_ar1_large_integer_calendar_origin_preserves_precision_and_forecast_lags(relation):
    values = np.array([-1.0, 0.0, 1.0, 3.0])
    positions = np.array([0, 2, 5, 7], dtype=np.int64)
    future = np.array([9, 12], dtype=np.int64)
    origin = np.int64(2**60 + 1)
    arguments = {"rho_grid": [-0.4, 0.0, 0.6], "prior_weights": [1.0, 2.0, 3.0]}
    original = fit_ar1_reference(values, positions, **arguments)
    shifted = fit_ar1_reference(values, positions + origin, **arguments)
    np.testing.assert_array_equal(shifted.positions - origin, positions)
    for name in (
        "mean_precision",
        "mean_location",
        "residual_sum_squares",
        "variance_mean",
        "posterior_weights",
    ):
        np.testing.assert_allclose(
            getattr(shifted, name), getattr(original, name), rtol=2e-14, atol=1e-14
        )
    mean, covariance = original.predictive_moments(future, relation=relation)
    shifted_mean, shifted_covariance = shifted.predictive_moments(
        future + origin, relation=relation
    )
    np.testing.assert_allclose(shifted_mean, mean, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(shifted_covariance, covariance, rtol=2e-14, atol=1e-14)
