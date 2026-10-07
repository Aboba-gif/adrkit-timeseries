"""Независимые тождества производных и условия построения функционалов."""
from types import SimpleNamespace

import numpy as np
import pytest

from adrkit.contracts import MapCapabilities, ScalarCapabilities
from adrkit.errors import CapabilityError
from adrkit.inverse.functionals import pullback_functional, scalar_from_functions, scale_functional, sum_functionals
from adrkit.inverse.optimizers import solve_lbfgsb
from adrkit.inverse.regularization import QuadraticPenalty
from adrkit.maps import map_from_functions
from adrkit.spaces import ArraySpace, matrix_map


@pytest.mark.parametrize("operation", ["predict", "jvp", "vjp"])
def test_matrix_map_rejects_overflowing_finite_inputs(operation):
    layout = ArraySpace((1,), axes=("entry",), coordinates=((0,),), units="1")
    mapping = matrix_map([[1e308]], domain=layout, codomain=layout)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="valid values"):
        if operation == "predict":
            mapping.predict.predict(np.array([2.]))
        elif operation == "jvp":
            mapping.jvp.jvp(np.array([0.]), np.array([2.]))
        else:
            mapping.vjp.vjp(np.array([0.]), np.array([2.]))


def space(shape=(2, 2), **changes):
    options = dict(axes=tuple(f"axis-{j}" for j in range(len(shape))),
                   coordinates=tuple(tuple(range(n)) for n in shape),
                   units="1", binding="fixed")
    options.update(changes)
    return ArraySpace(shape, **options)


def forbidden(*args):
    raise AssertionError("no function may execute during construction/preflight")


def exponential(domain):
    """Создать покомпонентную экспоненту с прямой и сопряжённой производными."""
    return map_from_functions(domain=domain, codomain=domain, predict=np.exp,
                              jvp=lambda at, h: np.exp(at)*h,
                              vjp=lambda at, z: np.exp(at)*z)


def quartic(domain):
    r"""Создать четвертичный штраф с центром 0.3 в объявленном пространстве.

    Parameters
    ----------
    domain : ArraySpace
        Пространство аргумента штрафа.

    Returns
    -------
    ScalarCapabilities
        Скалярный функционал и его координатный ковектор.

    Notes
    -----
    В данном тестовом функционале

    .. math::

        R(x)=\frac14\sum_i(x_i-0.3)^4,\qquad
        (dR(x))_i=(x_i-0.3)^3.
    """
    return scalar_from_functions(domain=domain,
                                 value=lambda x: np.sum((x-.3)**4)/4,
                                 cotangent=lambda x: (x-.3)**3)


def test_nonlinear_function_map_has_independent_jvp_and_coordinate_pullback():
    domain = space()
    mapping = exponential(domain)
    x = np.array([[.2, -.1], [.3, .7]])
    h = np.array([[.5, .4], [-.3, .1]])
    z = np.array([[.8, -.2], [-.1, .6]])
    exact_tangent = np.exp(x)*h
    np.testing.assert_allclose(mapping.jvp.jvp(x, h), exact_tangent, rtol=1e-14)
    np.testing.assert_allclose(np.sum(z*exact_tangent),
                               np.sum(h*mapping.vjp.vjp(x, z)), rtol=1e-14)
    eps = 1e-5
    numerical = (mapping.predict.predict(x+eps*h)-mapping.predict.predict(x-eps*h))/(2*eps)
    np.testing.assert_allclose(exact_tangent, numerical, rtol=1e-9, atol=1e-10)


def test_pullback_of_nonlinear_scalar_matches_chain_rule_and_directional_difference():
    domain = space()
    pulled = pullback_functional(quartic(domain), exponential(domain))
    x = np.array([[.2, -.1], [.3, .7]])
    h = np.array([[.5, .4], [-.3, .1]])
    expected_value = np.sum((np.exp(x)-.3)**4)/4
    expected_covector = np.exp(x)*(np.exp(x)-.3)**3
    np.testing.assert_allclose(pulled.value.value(x), expected_value, rtol=1e-14)
    np.testing.assert_allclose(pulled.cotangent.cotangent(x), expected_covector, rtol=1e-14)
    eps = 1e-5
    numerical = (pulled.value.value(x+eps*h)-pulled.value.value(x-eps*h))/(2*eps)
    np.testing.assert_allclose(numerical, np.sum(expected_covector*h), rtol=1e-9, atol=1e-10)


def test_sum_and_signed_scale_keep_tensor_layout_and_exact_weights():
    domain = space()
    center = np.array([[.4, -.5], [.1, .2]])
    quadratic = scalar_from_functions(domain=domain,
        value=lambda x: np.sum((x-center)**2)/2, cotangent=lambda x: x-center)
    functional = sum_functionals(scale_functional(quadratic, 2.5),
                                 scale_functional(quartic(domain), -.7))
    x = np.array([[.1, .8], [-.4, .7]])
    np.testing.assert_allclose(functional.value.value(x),
        1.25*np.sum((x-center)**2)-.175*np.sum((x-.3)**4), rtol=1e-14)
    np.testing.assert_allclose(functional.cotangent.cotangent(x),
        2.5*(x-center)-.7*(x-.3)**3, rtol=1e-14)
    assert functional.domain == domain


@pytest.mark.parametrize("missing", ["vjp", "cotangent", "both"])
def test_pullback_missing_derivative_rejects_before_any_function(missing):
    domain = space()
    mapping = map_from_functions(domain=domain, codomain=domain, predict=forbidden,
                                vjp=None if missing in ("vjp", "both") else forbidden)
    scalar = scalar_from_functions(domain=domain, value=forbidden,
                                  cotangent=None if missing in ("cotangent", "both") else forbidden)
    functional = pullback_functional(scalar, mapping)
    assert functional.cotangent is None
    with pytest.raises(CapabilityError, match="cotangent"):
        solve_lbfgsb(functional, np.zeros(domain.shape))
    with pytest.raises(CapabilityError, match="cotangent"):
        functional.value.cotangent(np.zeros(domain.shape))


def test_value_only_sum_and_zero_weight_do_not_manufacture_derivatives():
    domain = space()
    value_only = scalar_from_functions(domain=domain, value=lambda x: np.sum(np.abs(x)))
    for functional in (scale_functional(value_only, 0),
                       sum_functionals(quartic(domain), value_only)):
        assert functional.cotangent is None
        with pytest.raises(CapabilityError):
            solve_lbfgsb(functional, np.zeros(domain.shape))
    assert scale_functional(value_only, 0).value.value(np.ones(domain.shape)) == 0


def test_partial_map_capabilities_are_explicit_and_independent():
    domain = space()
    mapping = map_from_functions(domain=domain, codomain=domain, predict=forbidden,
                                jvp=lambda at, h: 2*h)
    assert mapping.jvp is not None and mapping.vjp is None
    np.testing.assert_array_equal(mapping.jvp.jvp(np.zeros(domain.shape), np.ones(domain.shape)), 2)
    with pytest.raises(CapabilityError, match="VJP"):
        mapping.predict.vjp(np.zeros(domain.shape), np.ones(domain.shape))


@pytest.mark.parametrize("change", [dict(coordinates=((1, 0), (0, 1))),
                                    dict(units="kg"), dict(binding="other")])
def test_incompatible_spaces_fail_during_inert_assembly(change):
    domain, other = space(), space(**change)
    left = scalar_from_functions(domain=domain, value=forbidden, cotangent=forbidden)
    right = scalar_from_functions(domain=other, value=forbidden, cotangent=forbidden)
    with pytest.raises(ValueError, match="Incompatible"):
        sum_functionals(left, right)
    mapping = map_from_functions(domain=domain, codomain=domain, predict=forbidden, vjp=forbidden)
    with pytest.raises(ValueError, match="Incompatible"):
        pullback_functional(right, mapping)


def test_sum_scale_and_pullback_never_inherit_a_prox():
    domain = space()
    scalar = scalar_from_functions(domain=domain, value=lambda x: np.sum(x*x)/2,
        cotangent=lambda x: x, prox=lambda x, step: x/(1+step), prox_pairing="euclidean")
    np.testing.assert_array_equal(scalar.prox.prox(np.ones(domain.shape), 1), .5)
    assert scalar.prox_pairing == "euclidean"
    for functional in (sum_functionals(scalar), scale_functional(scalar, 2),
                       pullback_functional(scalar, exponential(domain))):
        assert functional.prox is None and functional.prox_pairing is None


def test_public_matrix_map_snapshots_matrix_and_preserves_c_order_transpose():
    domain, codomain = space(), space((3,), binding="data")
    matrix = np.array([[1., 2., 0., -1.], [0., 1., 3., 0.], [2., 0., 1., 1.]])
    original = matrix.copy()
    mapping = matrix_map(matrix, domain=domain, codomain=codomain)
    matrix[:] = -900
    x = np.array([[.2, -.1], [.3, .7]])
    z = np.array([.3, -.2, .8])
    np.testing.assert_array_equal(mapping.predict.predict(x), original@x.ravel(order="C"))
    np.testing.assert_array_equal(mapping.vjp.vjp(x, z), (original.T@z).reshape(domain.shape))
    np.testing.assert_allclose(z@mapping.predict.predict(x), np.sum(x*mapping.vjp.vjp(x, z)), rtol=1e-14)


def test_matrix_map_supports_scalar_layout_and_does_not_infer_positivity():
    domain = space(())
    mapping = matrix_map([[-2]], domain=domain, codomain=domain)
    assert mapping.predict.predict(np.asarray(3.)).shape == ()
    assert mapping.predict.predict(np.asarray(3.)) == -6


def test_functions_receive_owned_input_copies_and_results_are_independent():
    domain = space()
    x = np.ones(domain.shape)
    h = np.full(domain.shape, 2.)
    output = np.full(domain.shape, 7.)

    def prediction(at):
        at[:] = -100
        return output

    def derivative(at, direction):
        at[:] = -200
        direction[:] = -300
        return output

    mapping = map_from_functions(domain=domain, codomain=domain, predict=prediction,
                                jvp=derivative, vjp=derivative)
    for result in (mapping.predict.predict(x), mapping.jvp.jvp(x, h), mapping.vjp.vjp(x, h)):
        result[:] = 500
    np.testing.assert_array_equal(x, 1)
    np.testing.assert_array_equal(h, 2)
    np.testing.assert_array_equal(output, 7)


def test_scalar_functions_cannot_mutate_caller_or_neighbouring_sum_inputs():
    domain = space()
    x = np.ones(domain.shape)

    def mutating_value(at):
        at[:] = 100
        return 3.

    def mutating_derivative(at):
        at[:] = 200
        return at

    first = scalar_from_functions(domain=domain, value=mutating_value,
                                  cotangent=mutating_derivative)
    second = scalar_from_functions(domain=domain, value=lambda at: np.sum(at), cotangent=lambda at: at)
    combined = sum_functionals(first, second)
    assert combined.value.value(x) == 7.
    np.testing.assert_array_equal(combined.cotangent.cotangent(x), 201)
    np.testing.assert_array_equal(x, 1)


def test_function_adapter_does_not_claim_to_freeze_external_closure():
    domain = space()
    settings = {"weight": 1.}
    scalar = scalar_from_functions(domain=domain, value=lambda x: settings["weight"]*np.sum(x))
    assert scalar.value.value(np.ones(domain.shape)) == 4
    settings["weight"] = 2
    assert scalar.value.value(np.ones(domain.shape)) == 8


@pytest.mark.parametrize("operation", ["predict", "jvp", "vjp", "cotangent", "prox"])
@pytest.mark.parametrize("bad", [np.ones(4), np.full((2, 2), np.nan), np.ones((2, 2), dtype=complex)])
def test_array_outputs_require_exact_finite_real_layout(operation, bad):
    domain = space()
    mapping = map_from_functions(domain=domain, codomain=domain,
                                predict=lambda x: bad, jvp=lambda at, h: bad, vjp=lambda at, z: bad)
    scalar = scalar_from_functions(domain=domain, value=lambda x: 1.,
        cotangent=lambda x: bad, prox=lambda x, step: bad, prox_pairing="euclidean")
    with pytest.raises(ValueError):
        if operation == "predict":
            mapping.predict.predict(np.zeros(domain.shape))
        elif operation in ("jvp", "vjp"):
            getattr(getattr(mapping, operation), operation)(np.zeros(domain.shape), np.ones(domain.shape))
        elif operation == "cotangent":
            scalar.cotangent.cotangent(np.zeros(domain.shape))
        else:
            scalar.prox.prox(np.zeros(domain.shape), 1.)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1j, [1.], "1", True])
def test_scalar_function_outputs_are_finite_real_scalars(bad):
    domain = space()
    scalar = scalar_from_functions(domain=domain, value=lambda x: bad)
    with pytest.raises(ValueError, match="finite real scalar"):
        scalar.value.value(np.zeros(domain.shape))


@pytest.mark.parametrize("weight", [True, np.nan, np.inf, [2.], 1j])
def test_invalid_weights_are_rejected_without_value(weight):
    scalar = scalar_from_functions(domain=space(), value=forbidden)
    with pytest.raises(ValueError):
        scale_functional(scalar, weight)


@pytest.mark.parametrize("step", [0, -1, True, np.nan, np.inf, 1j])
def test_invalid_prox_step_is_rejected_without_function(step):
    scalar = scalar_from_functions(domain=space(), value=forbidden,
                                   prox=forbidden, prox_pairing="euclidean")
    with pytest.raises(ValueError):
        scalar.prox.prox(np.zeros((2, 2)), step)


@pytest.mark.parametrize("factory,keyword", [("map", "predict"), ("map", "jvp"),
    ("map", "vjp"), ("scalar", "value"), ("scalar", "cotangent"), ("scalar", "prox")])
def test_noncallable_functions_are_rejected_during_inert_construction(factory, keyword):
    if factory == "map":
        options = dict(domain=space(), codomain=space(), predict=forbidden)
        constructor = map_from_functions
    else:
        options = dict(domain=space(), value=forbidden)
        constructor = scalar_from_functions
    options[keyword] = object()
    with pytest.raises(TypeError, match="callable"):
        constructor(**options)


def test_invalid_space_and_empty_sum_are_rejected_without_functions():
    with pytest.raises(TypeError, match="ArraySpace"):
        map_from_functions(domain=object(), codomain=space(), predict=forbidden)
    with pytest.raises(TypeError, match="ArraySpace"):
        scalar_from_functions(domain=object(), value=forbidden)
    with pytest.raises(TypeError, match="ArraySpace"):
        matrix_map(np.eye(4), domain=object(), codomain=space())
    with pytest.raises(ValueError, match="at least one"):
        sum_functionals()


def test_malformed_existing_capability_records_fail_before_numerical_work():
    domain = space()
    malformed = ScalarCapabilities(SimpleNamespace(value=3), domain)
    with pytest.raises(TypeError, match="callable value"):
        sum_functionals(malformed)
    with pytest.raises(TypeError, match="callable value"):
        scale_functional(malformed, 1.)
    scalar = scalar_from_functions(domain=domain, value=forbidden, cotangent=forbidden)
    mapping = MapCapabilities(SimpleNamespace(predict=forbidden), domain, domain, vjp=object())
    with pytest.raises(TypeError, match="callable vjp"):
        pullback_functional(scalar, mapping)


def test_previous_quadratic_capabilities_remain_usable_in_functional_algebra():
    domain = space((2,))
    penalty = QuadraticPenalty([.2, -.1], [[2., .3], [.3, 1.]], .7).capabilities(domain)
    functional = sum_functionals(scale_functional(penalty, 2.), penalty)
    x = np.array([.8, -.5])
    d = x-np.array([.2, -.1])
    gram = np.array([[2., .3], [.3, 1.]])
    np.testing.assert_allclose(functional.value.value(x), 1.05*d@gram@d, rtol=1e-14)
    np.testing.assert_allclose(functional.cotangent.cotangent(x), 2.1*gram@d, rtol=1e-14)


def test_wrong_inputs_fail_before_functions_and_missing_prox_is_explicit():
    domain = space()
    scalar = scalar_from_functions(domain=domain, value=forbidden, cotangent=forbidden)
    mapping = map_from_functions(domain=domain, codomain=domain, predict=forbidden,
                                jvp=forbidden, vjp=forbidden)
    with pytest.raises(ValueError, match="shape"):
        scalar.value.value(np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        scalar.cotangent.cotangent(np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        mapping.predict.predict(np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        mapping.jvp.jvp(np.zeros(domain.shape), np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        mapping.vjp.vjp(np.zeros(domain.shape), np.zeros(4))
    with pytest.raises(CapabilityError, match="prox"):
        scalar.value.prox(np.zeros(domain.shape), 1.)
