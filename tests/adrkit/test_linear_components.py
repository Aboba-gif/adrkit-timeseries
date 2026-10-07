"""Малые аналитические примеры отображений, базисов источника, матриц Грама и наблюдений."""
from dataclasses import FrozenInstanceError
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse.linalg import LinearOperator

from adrkit.backends.whole_space import Transport, WholeSpacePredictor
from adrkit.sources import P1Basis, hat_matrix, h1_gram, matrix_source_map, sampled_source_map
from adrkit.spaces import ArraySpace, matrix_map
from adrkit.observations.grid import GridObservation, observation_map
from adrkit.selection import Selection


TOL = 5e-13  # Малые вещественные матрицы float64 с независимо полученными аналитическими ответами.


def vector_space(coordinates, *, units="unknown", scale=1., binding=None):
    return ArraySpace((len(coordinates),), axes=("coordinate",),
                      coordinates=(coordinates,), units=units, scale=scale, binding=binding)


def test_space_exact_metadata_and_caller_ownership():
    coordinates = np.array([0., .5, 2.])
    space = vector_space(coordinates, units="q", binding="P1-grid-1")
    coordinates[:] = 99
    assert space.coordinates == ((0., .5, 2.),)
    space.require_compatible(vector_space([0., .5, 2.], units="q", binding="P1-grid-1"))
    for other in (vector_space([0., 2., .5], units="q", binding="P1-grid-1"),
                  vector_space([0., .5, 2.], units="other", binding="P1-grid-1"),
                  vector_space([0., .5, 2.], units="q", binding="P1-grid-2"),
                  vector_space([0., .5, 2.], units="q", scale=2, binding="P1-grid-1")):
        with pytest.raises(ValueError, match="Incompatible"):
            space.require_compatible(other)
    with pytest.raises(FrozenInstanceError):
        space.units = "changed"


def test_space_rejects_invalid_layout_and_arrays():
    with pytest.raises(ValueError):
        vector_space([1, 1])
    with pytest.raises(ValueError):
        vector_space([np.nan])
    with pytest.raises(ValueError):
        ArraySpace((2,), axes=("x",), coordinates=((1, 2),), units="q", dtype=complex)
    space = vector_space([0, 1])
    for value in ([1.], [1., np.nan], [1j, 2j]):
        with pytest.raises(ValueError):
            space.array(value)


def test_p1_nonuniform_values_and_free_endpoints():
    basis = P1Basis([0, Fraction(1, 3), 1, 2], time_unit="h")
    times = [0, Fraction(1, 6), Fraction(1, 3), Fraction(2, 3), 1, Fraction(3, 2), 2]
    expected = [[1, 0, 0, 0], [.5, .5, 0, 0], [0, 1, 0, 0],
                [0, .5, .5, 0], [0, 0, 1, 0], [0, 0, .5, .5], [0, 0, 0, 1]]
    assert_allclose(basis.values(times), expected, rtol=TOL, atol=TOL)
    assert_allclose(basis.values(times).sum(axis=1), 1, rtol=TOL, atol=TOL)
    assert_allclose(basis.values(times) @ basis.knots, np.asarray(times, float),
                    rtol=TOL, atol=TOL)


def test_p1_derivative_convention_and_affine_oracle():
    basis = P1Basis([0, 1/3, 1, 2], time_unit="h")
    times = [0, 1/6, 1/3, 2/3, 1, 1.5, 2]
    expected = [[-3, 3, 0, 0], [-3, 3, 0, 0], [0, -1.5, 1.5, 0],
                [0, -1.5, 1.5, 0], [0, 0, -1, 1], [0, 0, -1, 1], [0, 0, -1, 1]]
    derivative = basis.derivatives(times)
    assert_allclose(derivative, expected, rtol=TOL, atol=TOL)
    assert_allclose(derivative @ (2+3*basis.knots), 3, rtol=TOL, atol=TOL)
    assert_allclose(derivative @ np.ones(4), 0, rtol=TOL, atol=TOL)


def test_p1_snapshots_and_domain_guards_preserve_endpoint_extrapolation():
    knots = np.array([0., 1., 2.])
    basis = P1Basis(knots, time_unit="h")
    knots[:] = 99
    exported = basis.knots
    exported.setflags(write=True)
    exported[:] = 88
    assert_array_equal(basis.knots, [0, 1, 2])
    for times in ([-1e-15], [np.nextafter(2., np.inf)], [np.nan], [[1.]]):
        with pytest.raises(ValueError):
            basis.values(times)
    for knots in ([0, 0], [1, 0], [0, np.inf], [0]):
        with pytest.raises(ValueError):
            P1Basis(knots, time_unit="h")
    with pytest.raises(ValueError):
        P1Basis([0, 1], time_unit="h", endpoint="zero")
    assert_array_equal(hat_matrix([-1, 3], [0, 1, 2]), [[1, 0, 0], [0, 0, 1]])


def test_full_h1_mass_and_nonuniform_affine_integrals():
    r"""Проверить интегралы значений и производной аффинного профиля.

    Notes
    -----
    Для :math:`q(t)=2+3t` независимые интегралы равны

    .. math::

        \int_0^2 q(t)^2\,dt=56,\qquad
        \int_0^2 q'(t)^2\,dt=18.
    """
    knots = np.array([0., 1/3, 1., 2.])
    ones = np.ones(4)
    affine = 2+3*knots
    for tau in (0., .5, 2., -.5):
        gram = h1_gram(knots, tau)
        assert_allclose(ones @ gram @ ones, 2, rtol=TOL, atol=TOL)
        assert_allclose(affine @ gram @ affine, 56+18*tau*tau, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("knots", [[], [0.], [1., 0.], [0., 0.],
                                    [0., np.inf], [[0., 1.]], [0., 1.j]])
def test_h1_gram_rejects_invalid_p1_interval_before_assembly(knots):
    with np.errstate(divide="raise", invalid="raise"), pytest.raises(ValueError, match="P1"):
        h1_gram(knots, 0.)


@pytest.mark.parametrize("scale", [np.nan, np.inf, [1., 2.], True, .5+0j, "0.5"])
def test_h1_gram_rejects_invalid_derivative_scale_before_assembly(scale):
    with pytest.raises(ValueError, match="time_scale"):
        h1_gram([0., 1.], scale)


def source_fixture():
    basis = P1Basis([0, 1, 3], time_unit="h")
    times = np.array([0., .5, 1., 2., 3.])
    domain = vector_space(["a0", "a1", "a2"], units="1")
    codomain = vector_space(times, units="ug/(m*s)")
    mapping = sampled_source_map(basis, times, scale=2.5, domain=domain, codomain=codomain)
    return times, mapping


def test_source_actions_manual_scale_covector_and_scipy_bridge():
    _, mapping = source_fixture()
    at = np.array([2., -1., 4.])
    h = np.array([1., 3., -2.])
    z = np.array([1., -2., 3., 4., -1.])
    expected = 2.5*np.array([2., .5, -1., 1.5, 4.])
    jh = 2.5*np.array([1., 2., 3., .5, -2.])
    jtz = 2.5*np.array([z[0]+z[1]/2, z[1]/2+z[2]+z[3]/2, z[3]/2+z[4]])
    assert_allclose(mapping.predict.predict(at), expected, rtol=TOL, atol=TOL)
    assert_allclose(mapping.jvp.jvp(at, h), jh, rtol=TOL, atol=TOL)
    assert_allclose(mapping.vjp.vjp(at, z), jtz, rtol=TOL, atol=TOL)
    assert_allclose(jh @ z, h @ jtz, rtol=TOL, atol=TOL)
    # Одновременное удвоение обеих производных сохраняет тождество сопряжённости, но нарушает
    # независимо полученный ответ.
    assert_allclose((2*jh) @ z, h @ (2*jtz), rtol=TOL, atol=TOL)
    assert np.linalg.norm(2*jh-jh) > 1
    assert np.linalg.norm(jtz/2.5-jtz) > 1
    op = mapping.predict.linear_operator()
    assert isinstance(op, LinearOperator) and op.dtype == np.dtype("float64")
    assert_allclose(op @ h, jh, rtol=TOL, atol=TOL)
    assert_allclose(op.rmatvec(z), jtz, rtol=TOL, atol=TOL)
    assert_allclose(op.matmat(np.column_stack([h, -h])), np.column_stack([jh, -jh]),
                    rtol=TOL, atol=TOL)


def test_source_snapshots_external_basis_and_no_output_alias():
    class Basis:
        matrix = np.array([[1., -2.], [3., 4.]])

        def values(self, times):
            return self.matrix

    basis = Basis()
    times = np.array([0., 1.])
    mapping = sampled_source_map(basis, times, scale=3,
                                domain=vector_space(["a", "b"]), codomain=vector_space(times))
    basis.matrix[:] = 0
    times[:] = 50
    mapping.predict.matrix[:] = 90
    a = np.array([2., 1.])
    returned = mapping.predict.predict(a)
    returned[:] = 99
    assert_array_equal(mapping.predict.predict(a), [0, 30])
    assert_array_equal(mapping.vjp.vjp(a, [1, 2]), [21, 18])
    operator = mapping.predict.linear_operator()
    operator._CustomLinearOperator__matvec_impl = lambda x: np.zeros(2)
    assert_array_equal(mapping.predict.predict(a), [0, 30])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_copied_matrix_operations_share_one_independent_tensor_map(dtype):
    domain = ArraySpace((2, 2), axes=("row", "column"),
        coordinates=((0, 1), (0, 1)), units="1", dtype=dtype)
    codomain = ArraySpace((2,), axes=("value",),
        coordinates=((0, 1),), units="1", dtype=dtype)
    original = matrix_map([[1., 2., -1., 0.], [0., 3., 2., -2.]],
                          domain=domain, codomain=codomain)
    owned = deepcopy(original)
    assert owned.predict is owned.jvp is owned.vjp
    assert owned.predict is not original.predict
    original.predict._matrix[:] = 0
    at = np.array([[1., 2.], [-3., 4.]], dtype=dtype)
    direction = np.array([[2., -1.], [1., 3.]], dtype=dtype)
    assert_array_equal(original.predict.predict(at), [0., 0.])
    assert_array_equal(owned.predict.predict(at), [8., -8.])
    assert_array_equal(owned.jvp.jvp(at, direction), [-1., -7.])
    assert_array_equal(owned.vjp.vjp(at, [2., -1.]), [[2., 1.], [-4., 2.]])
    assert owned.predict.predict(at).dtype == np.dtype(dtype)


def test_copied_whole_space_predictor_retains_type_and_extra_state():
    predictor = WholeSpacePredictor([1., 2.], [[.5, 0.], [.5, 0.]], [0., 0.],
        [0., 1., 2.], Transport(.2, (.1, 0.), .01, .3),
        domain=vector_space([0, 1]), codomain=vector_space([0, 1]))
    predictor.intervals = np.array([0., 1., 2.])
    expected = predictor.predict([2., 3.])
    owned = deepcopy(predictor.capabilities())
    assert isinstance(owned.predict, WholeSpacePredictor)
    assert owned.predict is owned.jvp is owned.vjp
    predictor.intervals[:] = 99
    predictor._matrix[:] = 0
    assert_array_equal(owned.predict.intervals, [0., 1., 2.])
    assert_array_equal(owned.predict.predict([2., 3.]), expected)


def test_source_changed_direction_and_nextafter_are_not_cached():
    domain = vector_space([0, 1])
    mapping = matrix_source_map(np.eye(2), scale=1, domain=domain, codomain=domain)
    a = np.array([1., 2.])
    changed = np.nextafter(a, np.inf)
    assert_array_equal(mapping.predict.predict(changed), changed)
    assert np.any(mapping.predict.predict(changed) != mapping.predict.predict(a))
    h = np.array([3., -1.])
    old = mapping.jvp.jvp(a, h)
    h[:] = [-4, 7]
    assert_array_equal(old, [3, -1])
    assert_array_equal(mapping.jvp.jvp(changed, h), [-4, 7])
    assert_array_equal(mapping.vjp.vjp(a, [6, 9]), [6, 9])


def test_source_rejects_wrong_sampling_binding_and_shapes():
    basis = P1Basis([0, 1], time_unit="h")
    domain = vector_space([0, 1])
    with pytest.raises(ValueError, match="sampling times"):
        sampled_source_map(basis, [0, 1], scale=1, domain=domain, codomain=vector_space([1, 0]))
    with pytest.raises(ValueError):
        matrix_source_map([[1, 2]], scale=1, domain=domain, codomain=domain)
    with pytest.raises(ValueError):
        matrix_source_map(np.eye(2), scale=0, domain=domain, codomain=domain)


def grid_fixture():
    s = np.array([[2., -.5], [.25, 3.]])
    # Неупорядоченный снимок, внешний функционал со знаком и строка усреднения.
    w = np.array([[0., 0., 1.], [1., -2., 0.], [0., .25, .75]])
    domain = ArraySpace((3, 2), axes=("time", "cell"),
                        coordinates=([0, 1, 2], ["x0", "x1"]), units="C")
    rows = [(r, t) for r in ("r1", "r0") for t in ("t2", "signed", "window")]
    codomain = vector_space(rows, units="C")
    return s, w, domain, codomain


def test_grid_observation_manual_tensor_and_transpose_order():
    s, w, domain, codomain = grid_fixture()
    mapping = observation_map(s, w, domain=domain, codomain=codomain)
    c = np.array([[1., -2.], [3., 4.], [-5., 6.]])
    h = np.array([[2., 3.], [-1., .5], [4., -2.]])
    z = np.array([1., -2., .5, 3., 4., -1.])
    def manual(values):
        return np.array([sum(w[j, t]*s[r, k]*values[t, k]
                             for t in range(3) for k in range(2))
                         for r in range(2) for j in range(3)])
    pullback = np.array([[sum(z[3*r+j]*w[j, t]*s[r, k]
                              for r in range(2) for j in range(3))
                          for k in range(2)] for t in range(3)])
    assert_allclose(mapping.predict.predict(c), manual(c), rtol=TOL, atol=TOL)
    assert_allclose(mapping.jvp.jvp(c, h), manual(h), rtol=TOL, atol=TOL)
    assert_allclose(mapping.vjp.vjp(c, z), pullback, rtol=TOL, atol=TOL)
    assert_allclose(manual(h) @ z, np.sum(h*pullback), rtol=TOL, atol=TOL)
    wrong_order = manual(c).reshape(2, 3).T.ravel()
    assert np.linalg.norm(wrong_order-manual(c)) > 1


def test_grid_snapshot_and_defensive_weights_outputs():
    s, w, domain, codomain = grid_fixture()
    observation = GridObservation(s, w, domain=domain, codomain=codomain)
    c = np.arange(6.).reshape(3, 2)
    expected = observation.predict(c)
    s[:] = 0
    w[:] = 0
    observation.space_weights[:] = 0
    observation.time_weights[:] = 0
    observation.predict(c)[:] = 0
    assert_array_equal(observation.predict(c), expected)
    assert np.linalg.norm(expected) > 0
    with pytest.raises(ValueError):
        observation.vjp(c, [1, 2])


def test_selection_hidden_nan_order_scatter_and_covariance_before_inverse():
    layout = vector_space(["r0", "hidden", "r2"], units="C")
    indices = np.array([2, 0])
    selection = Selection(indices, layout)
    indices[:] = 1
    selection.indices[:] = 1
    assert_array_equal(selection.take([5, np.nan, 7]), [7, 5])
    assert_array_equal(selection.scatter([11, -3]), [-3, 0, 11])
    assert selection.selected_layout.coordinates == (("r2", "r0"),)
    covariance = np.array([[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]])
    selected = selection.take_covariance(covariance)
    assert_array_equal(selected, [[2, 0], [0, 2]])
    assert_allclose(np.linalg.inv(selected), np.diag([.5, .5]), rtol=TOL, atol=TOL)
    assert np.linalg.norm(selection.take_covariance(np.linalg.inv(covariance))
                          -np.diag([.5, .5])) > .1


def test_selection_rejects_duplicates_fractional_unknown_and_bad_shapes():
    layout = vector_space(["a", "b", "c"])
    for indices in ([0, 0], [1.5], [-1], [3], [], [True]):
        with pytest.raises(ValueError):
            Selection(indices, layout)
    selection = Selection([1, 0], layout)
    with pytest.raises(ValueError):
        selection.take([1, 2])
    with pytest.raises(ValueError):
        selection.scatter([1, 2, 3])
