"""MAD: точные ветви, неаддитивные направления и достаточные сертификаты."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from timeseries.statistics import (
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
