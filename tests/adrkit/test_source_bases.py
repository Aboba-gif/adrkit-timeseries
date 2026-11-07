"""Независимые проверки внешнего базиса и переноса производной источника."""

import numpy as np
import pytest

import adrkit.sources as source_module
from adrkit.sources import (
    GaussianBasis,
    P1Basis,
    register_source_basis,
    sampled_source_map,
    source_basis_from_record,
)
from adrkit.spaces import ArraySpace


class PolynomialBasis:
    def values(self, times):
        return np.column_stack((np.ones_like(times), times, times**2))


def test_external_basis_scale_duality_and_owned_samples():
    times = np.array([0.0, 0.25, 1.5, 2.0])
    domain = ArraySpace((3,), axes=("coefficient",), coordinates=((0, 1, 2),), units="1")
    output = ArraySpace((4,), axes=("time",), coordinates=(times,), units="kg/s")
    basis = PolynomialBasis()
    mapping = sampled_source_map(basis, times, scale=2.5, domain=domain, codomain=output)
    x, h, z = (
        np.array([0.4, -0.2, 0.3]),
        np.array([0.1, 0.7, -0.4]),
        np.array([2.0, -1.0, 0.5, 3.0]),
    )
    expected = 2.5 * np.array([[1, 0, 0], [1, 0.25, 0.0625], [1, 1.5, 2.25], [1, 2, 4]])
    times[:] = -99
    basis.values = lambda t: np.zeros((len(t), 3))
    np.testing.assert_allclose(mapping.predict.predict(x), expected @ x, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(mapping.jvp.jvp(x, h), expected @ h, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(mapping.vjp.vjp(x, z), expected.T @ z, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z @ mapping.jvp.jvp(x, h), h @ mapping.vjp.vjp(x, z), rtol=1e-14)
    returned = mapping.predict.predict(x)
    returned[:] = 123
    np.testing.assert_allclose(mapping.predict.predict(x), expected @ x, rtol=1e-14, atol=1e-14)


def test_nonuniform_p1_endpoints_and_derivative_convention():
    basis = P1Basis([0, 0.5, 2], time_unit="s")
    np.testing.assert_array_equal(
        basis.values([0, 0.25, 0.5, 1.25, 2]),
        [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5], [0, 0, 1]],
    )
    np.testing.assert_allclose(
        basis.derivatives([0, 0.5, 2]),
        [[-2, 2, 0], [0, -2 / 3, 2 / 3], [0, -2 / 3, 2 / 3]],
        rtol=1e-15,
    )
    with pytest.raises(ValueError, match="closed P1 interval"):
        basis.values([2.000001])


@pytest.mark.parametrize("knots", [[0, 0, 1], [1, 0], [0, float("nan")]])
def test_invalid_basis_rejected(knots):
    with pytest.raises(ValueError):
        P1Basis(knots, time_unit="s")


@pytest.mark.parametrize("include_constant", [False, True])
def test_gaussian_values_integrals_and_gram_match_quadrature(include_constant):
    from scipy.integrate import quad

    basis = GaussianBasis([-0.2, 0.5, 1.3], 0.27, time_unit="s", include_constant=include_constant)
    left, right = 0.0, 1.0
    areas = basis.integrals(left, right)
    gram = basis.gram(left, right)
    for i in range(basis.size):
        expected_area = quad(lambda t: basis.values([t])[0, i], left, right, epsabs=1e-12)[0]
        np.testing.assert_allclose(areas[i], expected_area, rtol=2e-13, atol=1e-14)
        for j in range(basis.size):
            expected = quad(
                lambda t: basis.values([t])[0, i] * basis.values([t])[0, j],
                left,
                right,
                epsabs=1e-12,
            )[0]
            np.testing.assert_allclose(gram[i, j], expected, rtol=3e-13, atol=1e-14)
    np.testing.assert_array_equal(gram, gram.T)
    assert np.linalg.eigvalsh(gram).min() > 0
    np.testing.assert_allclose(
        basis.integrals(np.array([[0.0], [0.2]]), np.array([0.5, 1.0])),
        np.asarray([[basis.integrals(a, b) for b in [0.5, 1.0]] for a in [0.0, 0.2]]),
    )


def test_gaussian_basis_owns_centers_and_supports_constant_only():
    centers = np.array([0.0, 1.0])
    basis = GaussianBasis(centers, 0.2, time_unit="s")
    centers[:] = 99
    returned = basis.centers
    returned[:] = 88
    np.testing.assert_array_equal(basis.centers, [0.0, 1.0])
    constant = GaussianBasis([], 1.0, time_unit="s")
    assert constant.size == 1
    np.testing.assert_array_equal(constant.values([0.0, 2.0]), [[1.0], [1.0]])
    np.testing.assert_array_equal(constant.integrals([0.0, 0.5], [1.0, 2.0]), [[1.0], [1.5]])
    np.testing.assert_array_equal(constant.gram(0.0, 2.0), [[2.0]])


def test_p1_integrals_and_gram_match_piecewise_quadrature_and_interval_loads():
    from scipy.integrate import quad
    from adrkit.loads import interval_hat_integrals
    from adrkit.sources import h1_gram

    basis = P1Basis([0.0, 0.5, 2.0], time_unit="s")
    edges = np.array([-1.0, 0.0, 0.3, 0.5, 1.2, 2.0, 3.0])
    intervals = basis.integrals(edges[:-1], edges[1:])
    np.testing.assert_array_equal(interval_hat_integrals(basis, edges), intervals)
    clipped = np.clip(edges, 0.0, 2.0)
    np.testing.assert_allclose(intervals.sum(axis=1), np.diff(clipped), atol=1e-15)
    np.testing.assert_allclose(intervals @ basis.knots, np.diff(clipped**2) / 2, atol=1e-15)
    for left, right in [(0.2, 1.7), (-1.0, 3.0)]:
        gram = basis.gram(left, right)
        a, b = max(left, 0.0), min(right, 2.0)
        points = [0.5]
        for i in range(basis.size):
            area = quad(lambda t: basis.values([t])[0, i], a, b, points=points)[0]
            np.testing.assert_allclose(basis.integrals(left, right)[i], area, atol=1e-14)
            for j in range(basis.size):
                expected = quad(
                    lambda t: basis.values([t])[0, i] * basis.values([t])[0, j],
                    a,
                    b,
                    points=points,
                )[0]
                np.testing.assert_allclose(gram[i, j], expected, atol=1e-14)
    np.testing.assert_allclose(basis.gram(0.0, 2.0), h1_gram(basis.knots, 0.0), atol=1e-15)
    np.testing.assert_array_equal(basis.integrals(-1e300, -1e200), np.zeros(basis.size))


@pytest.mark.parametrize(
    "centers,width",
    [([0.0, np.nan], 0.2), ([0.0], 0.0), ([0.0], np.inf), ([[0.0]], 1.0)],
)
def test_invalid_gaussian_settings_are_rejected(centers, width):
    with pytest.raises(ValueError):
        GaussianBasis(centers, width, time_unit="s")


@pytest.fixture
def source_registry(monkeypatch):
    monkeypatch.setattr(
        source_module, "_SOURCE_BASIS_FACTORIES", dict(source_module._SOURCE_BASIS_FACTORIES)
    )
    monkeypatch.setattr(source_module, "entry_points", lambda **filters: ())


@pytest.mark.parametrize(
    "basis",
    [
        GaussianBasis([-0.2, 0.5, 1.3], 0.27, time_unit="s"),
        GaussianBasis([0.3, 0.8], 0.2, time_unit="min", include_constant=False),
        P1Basis([0.0, 0.5, 2.0], time_unit="s"),
    ],
)
def test_builtin_source_record_preserves_basis(basis, source_registry):
    record = basis.to_record()
    restored = source_basis_from_record(record)
    assert restored.to_record() == record
    assert type(restored) is type(basis)
    times = np.linspace(0.0, 1.0, 11)
    np.testing.assert_array_equal(restored.values(times), basis.values(times))
    np.testing.assert_array_equal(restored.integrals(0.1, 0.9), basis.integrals(0.1, 0.9))
    np.testing.assert_array_equal(restored.gram(0.1, 0.9), basis.gram(0.1, 0.9))


def test_custom_source_factory_receives_parameters_without_mutating_record(source_registry):
    received = {}

    def factory(*, nodes, time_unit, label):
        received.update(nodes=nodes, time_unit=time_unit, label=label)
        return P1Basis(nodes, time_unit=time_unit)

    register_source_basis("custom", factory)
    register_source_basis("custom", factory)
    record = {"type": "custom", "nodes": [0.0, 0.5, 2.0], "time_unit": "min", "label": "test"}
    basis = source_basis_from_record(record)
    assert received == {key: value for key, value in record.items() if key != "type"}
    assert record["type"] == "custom"
    assert basis.time_unit == "min"
    np.testing.assert_array_equal(basis.values([0.0, 0.5, 2.0]), np.eye(3))


@pytest.mark.parametrize("name", ["", " ", " custom", "custom ", None, 3])
def test_source_factory_name_is_explicit(name, source_registry):
    with pytest.raises(ValueError, match="name"):
        register_source_basis(name, P1Basis)
    with pytest.raises(ValueError, match="name"):
        source_basis_from_record({"type": name})


def test_invalid_source_factory_and_record_are_rejected(source_registry):
    with pytest.raises(TypeError, match="callable"):
        register_source_basis("custom", 3)
    with pytest.raises(TypeError, match="mapping"):
        source_basis_from_record([])
    with pytest.raises(KeyError, match="type"):
        source_basis_from_record({"knots": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Unknown source basis"):
        source_basis_from_record({"type": "missing"})


def test_local_and_builtin_source_names_cannot_be_replaced(source_registry):
    with pytest.raises(ValueError, match="already registered"):
        register_source_basis("gaussian", P1Basis)
    register_source_basis("custom", P1Basis)
    with pytest.raises(ValueError, match="already registered"):
        register_source_basis("custom", GaussianBasis)


def _installed_factories(monkeypatch, providers):
    def selected_entry_points(*, group, name):
        assert group == "adrkit_timeseries.source_bases"
        return tuple(provider for provider in providers if provider.name == name)

    monkeypatch.setattr(source_module, "entry_points", selected_entry_points)


class _FactoryEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self.factory = factory
        self.loaded = False

    def load(self):
        self.loaded = True
        return self.factory


def test_only_requested_installed_source_factory_is_loaded(monkeypatch, source_registry):
    selected = _FactoryEntryPoint("installed", P1Basis)
    unrelated = _FactoryEntryPoint("unrelated", object())
    _installed_factories(monkeypatch, [selected, unrelated])
    basis = source_basis_from_record(
        {"type": "installed", "knots": [0.0, 0.5, 2.0], "time_unit": "h"}
    )
    assert selected.loaded
    assert not unrelated.loaded
    assert basis.size == 3
    assert basis.time_unit == "h"


@pytest.mark.parametrize("name", ["gaussian", "custom", "installed"])
def test_source_provider_collisions_are_reported_before_loading(
    name, monkeypatch, source_registry
):
    if name == "custom":
        register_source_basis(name, P1Basis)
    providers = [_FactoryEntryPoint(name, P1Basis)]
    if name == "installed":
        providers.append(_FactoryEntryPoint(name, GaussianBasis))
    _installed_factories(monkeypatch, providers)
    with pytest.raises(ValueError, match="Multiple source basis factories"):
        source_basis_from_record({"type": name})
    assert not any(provider.loaded for provider in providers)
    with pytest.raises(ValueError, match="already registered"):
        register_source_basis(name, P1Basis)


def test_installed_source_factory_must_be_callable(monkeypatch, source_registry):
    _installed_factories(monkeypatch, [_FactoryEntryPoint("invalid", object())])
    with pytest.raises(TypeError, match="callable"):
        source_basis_from_record({"type": "invalid"})


def test_source_factory_constructor_errors_are_preserved(source_registry):
    failure = RuntimeError("the custom constructor failed")

    def factory(**parameters):
        raise failure

    register_source_basis("broken", factory)
    with pytest.raises(RuntimeError) as caught:
        source_basis_from_record({"type": "broken", "custom_argument": [1, 2]})
    assert caught.value is failure


def test_source_entry_point_load_errors_are_preserved(monkeypatch, source_registry):
    failure = ImportError("the installed module failed")

    class BrokenEntryPoint:
        name = "broken"

        def load(self):
            raise failure

    _installed_factories(monkeypatch, [BrokenEntryPoint()])
    with pytest.raises(ImportError) as caught:
        source_basis_from_record({"type": "broken"})
    assert caught.value is failure
