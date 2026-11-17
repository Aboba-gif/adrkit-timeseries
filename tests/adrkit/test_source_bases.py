"""Независимые проверки внешнего базиса и переноса производной источника."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import adrkit.sources as source_module
from adrkit.sources import (
    GaussianBasis,
    P1Basis,
    ShiftedBasis,
    register_source_basis,
    sampled_source_map,
    source_basis_from_record,
    source_basis_providers,
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
    for resolve in (source_basis_from_record, source_basis_providers):
        with pytest.raises(ValueError, match="Multiple source basis factories"):
            resolve({"type": name})
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


@pytest.mark.parametrize("origin", [-2.5, 3.0])
@pytest.mark.parametrize("include_constant", [False, True])
def test_shifted_native_gaussian_is_the_same_all_time_function_in_new_coordinates(
    origin, include_constant
):
    child = GaussianBasis([0.0, 1.0], 0.3, time_unit="h", include_constant=include_constant)
    shifted = ShiftedBasis(child, origin, time_unit="h")
    translated = GaussianBasis(
        [origin, 1.0 + origin], 0.3, time_unit="h", include_constant=include_constant
    )
    times = np.array([-8.0, -1.0, 0.0, 1.0, 9.0])
    np.testing.assert_allclose(shifted.values(times), translated.values(times), atol=1e-15)
    left = np.array([[-4.0], [0.25]])
    right = np.array([2.0, 6.0])
    np.testing.assert_allclose(
        shifted.integrals(left, right), translated.integrals(left, right), atol=2e-15
    )
    np.testing.assert_allclose(shifted.gram(-4.0, 6.0), translated.gram(-4.0, 6.0), atol=2e-15)
    assert shifted.size == child.size
    assert shifted.time_unit == "h"


def test_shifted_native_preserves_p1_values_domain_and_its_native_zero_integrals():
    shifted = ShiftedBasis(P1Basis([0.0, 1.0], time_unit="h"), 2.0, time_unit="h")
    np.testing.assert_array_equal(shifted.values([2.0, 2.5, 3.0]), [[1, 0], [0.5, 0.5], [0, 1]])
    with pytest.raises(ValueError, match="closed P1 interval"):
        shifted.values([1.0])
    np.testing.assert_allclose(shifted.integrals(-1.0, 4.0), [0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(shifted.gram(-1.0, 4.0), [[1 / 3, 1 / 6], [1 / 6, 1 / 3]])


@pytest.mark.parametrize("extension", ["zero", "constant"])
def test_shifted_p1_values_integrals_gram_and_profile_mass_share_one_extension(extension):
    shifted = ShiftedBasis(
        P1Basis([0.0, 1.0], time_unit="h"),
        2.0,
        time_unit="h",
        extension=extension,
        interval=(0.0, 1.0),
    )
    values = [[0, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0]]
    areas = [[0, 0], [3 / 8, 1 / 8], [1 / 8, 3 / 8], [0, 0]]
    gram = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
    mass = 3.0
    if extension == "constant":
        values[0], values[-1] = [1, 0], [0, 1]
        areas[0], areas[-1] = [3, 0], [0, 1]
        gram += np.diag([3.0, 1.0])
        mass = 13.0
    edges = np.array([-1.0, 2.0, 2.5, 3.0, 4.0])
    np.testing.assert_array_equal(shifted.values([1.0, 2.0, 2.5, 3.0, 4.0]), values)
    np.testing.assert_allclose(shifted.integrals(edges[:-1], edges[1:]), areas, atol=1e-15)
    np.testing.assert_allclose(shifted.gram(-1.0, 4.0), gram, atol=1e-15)
    coefficients = np.array([2.0, 4.0])
    np.testing.assert_allclose(coefficients @ shifted.integrals(-1.0, 4.0), mass, atol=1e-15)
    np.testing.assert_allclose(
        coefficients @ shifted.gram(-1.0, 4.0) @ coefficients,
        28 / 3 + (28.0 if extension == "constant" else 0.0),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "extension,expected_gram",
    [
        ("zero", [[13 / 96, 11 / 96], [11 / 96, 13 / 96]]),
        ("constant", [[7 / 24, 5 / 24], [5 / 24, 7 / 24]]),
    ],
)
def test_shifted_extension_uses_declared_local_interval_not_child_endpoints(
    extension, expected_gram
):
    shifted = ShiftedBasis(
        P1Basis([0.0, 1.0], time_unit="h"),
        -2.0,
        time_unit="h",
        extension=extension,
        interval=(0.25, 0.75),
    )
    outer = [[0, 0], [0, 0]] if extension == "zero" else [[0.75, 0.25], [0.25, 0.75]]
    np.testing.assert_array_equal(shifted.values([-2.0, -1.0]), outer)
    np.testing.assert_array_equal(shifted.values([-1.75, -1.25]), [[0.75, 0.25], [0.25, 0.75]])
    np.testing.assert_allclose(shifted.gram(-2.0, -1.0), expected_gram, atol=1e-15)
    np.testing.assert_allclose(
        shifted.integrals(-2.0, -1.0),
        [0.25, 0.25] if extension == "zero" else [0.5, 0.5],
        atol=1e-15,
    )


class _ConfigurablePowerBasis:
    """Registered arbitrary-size polynomial basis with mutable constructor input."""

    def __init__(self, count, *, time_unit, weights=None):
        self.size = count
        self.time_unit = time_unit
        self.weights = (
            np.ones(count) if weights is None else np.array(weights, dtype=float, copy=True)
        )

    def values(self, times):
        return np.asarray(times)[:, None] ** np.arange(self.size) * self.weights

    def integrals(self, left, right):
        powers = np.arange(1, self.size + 1)
        left, right = np.broadcast_arrays(left, right)
        return (right[..., None] ** powers - left[..., None] ** powers) / powers * self.weights

    def gram(self, left, right):
        powers = np.add.outer(np.arange(self.size), np.arange(self.size)) + 1
        return (right**powers - left**powers) / powers * np.outer(self.weights, self.weights)

    def to_record(self):
        return {
            "type": "test-powers",
            "count": self.size,
            "weights": self.weights.tolist(),
            "time_unit": self.time_unit,
        }


def test_shifted_origin_composition_and_polynomial_change_of_variables():
    child = _ConfigurablePowerBasis(3, time_unit="h")
    shifted = ShiftedBasis(child, 3.0, time_unit="h")
    nested = ShiftedBasis(shifted, -5.0, time_unit="h")
    np.testing.assert_array_equal(
        shifted.values([2.0, 3.0, 5.0]), [[1, -1, 1], [1, 0, 0], [1, 2, 4]]
    )
    np.testing.assert_allclose(shifted.integrals(2.0, 5.0), [3.0, 1.5, 3.0], atol=1e-15)
    np.testing.assert_allclose(
        shifted.gram(2.0, 5.0),
        [[3, 1.5, 3], [1.5, 3, 15 / 4], [3, 15 / 4, 33 / 5]],
        atol=1e-15,
    )
    # (s-(-5))-3 = s+2: a positive origin delays the child, it does not advance it.
    np.testing.assert_array_equal(
        nested.values([-3.0, -2.0, 0.0]), shifted.values([2.0, 3.0, 5.0])
    )
    np.testing.assert_array_equal(nested.integrals(-3.0, 0.0), shifted.integrals(2.0, 5.0))
    np.testing.assert_array_equal(nested.gram(-3.0, 0.0), shifted.gram(2.0, 5.0))


@pytest.mark.parametrize("count", [2, 5])
def test_shifted_nested_custom_registry_record_keeps_configurable_size_and_owns_inputs(
    count, source_registry
):
    register_source_basis("test-powers", _ConfigurablePowerBasis)
    child = _ConfigurablePowerBasis(count, time_unit="h")
    shifted = ShiftedBasis(child, 2.0, time_unit="h", extension="constant", interval=(0.0, 1.0))
    child.weights[:] = 99
    shifted.basis.weights[:] = 88
    record = shifted.to_record()
    restored = source_basis_from_record(record)
    record["basis"]["weights"][:] = [77] * count
    record["interval"][:] = [4.0, 5.0]
    assert restored.size == count
    np.testing.assert_array_equal(
        restored.values([1.0, 2.0, 3.0, 4.0]), shifted.values([1.0, 2.0, 3.0, 4.0])
    )
    np.testing.assert_array_equal(restored.integrals(1.0, 4.0), shifted.integrals(1.0, 4.0))
    np.testing.assert_array_equal(restored.gram(1.0, 4.0), shifted.gram(1.0, 4.0))
    assert restored.to_record() == shifted.to_record()
    np.testing.assert_array_equal(restored.basis.weights, np.ones(count))


def test_shifted_record_inherits_only_absent_child_unit_without_mutating_record(source_registry):
    record = {
        "type": "shifted",
        "origin": 2.0,
        "time_unit": "h",
        "basis": {"type": "p1", "knots": [0.0, 1.0]},
    }
    shifted = source_basis_from_record(record)
    record["basis"]["knots"][0] = -9.0
    assert "time_unit" not in record["basis"]
    np.testing.assert_array_equal(shifted.values([2.0, 3.0]), np.eye(2))
    assert shifted.to_record()["basis"]["time_unit"] == "h"
    with pytest.raises(ValueError, match="same time unit"):
        ShiftedBasis({"type": "p1", "knots": [0.0, 1.0], "time_unit": "s"}, 2.0, time_unit="h")


@pytest.mark.parametrize(
    "parameters",
    [
        {"time_unit": ""},
        {"time_unit": "s"},
        {"origin": np.inf},
        {"origin": 1j},
        {"origin": [1.0]},
        {"extension": "periodic"},
        {"interval": (0.0, 1.0)},
        {"extension": "zero"},
        {"extension": "constant", "interval": (1.0, 1.0)},
        {"extension": "constant", "interval": (2.0, 1.0)},
        {"extension": "zero", "interval": (0.0, np.inf)},
        {"extension": "zero", "interval": (0.0, 1.0, 2.0)},
    ],
)
def test_shifted_basis_requires_explicit_compatible_clock_and_extension(parameters):
    settings = {"origin": 0.0, "time_unit": "h", **parameters}
    with pytest.raises(ValueError):
        ShiftedBasis(P1Basis([0.0, 1.0], time_unit="h"), **settings)


def test_shifted_basis_does_not_invent_optional_derivatives():
    shifted = ShiftedBasis(P1Basis([0.0, 1.0], time_unit="h"), 1.0, time_unit="h")
    assert not hasattr(shifted, "derivatives")


def _provider_entry(name, factory, version):
    provider = _FactoryEntryPoint(name, factory)
    provider.value = f"source_{name}:factory"
    provider.dist = SimpleNamespace(metadata={"Name": f"source-{name}"}, version=version)
    return provider


def _provider_description(provider):
    return {
        "name": provider.name,
        "factory": provider.value,
        "distribution": provider.dist.metadata["Name"],
        "version": provider.dist.version,
    }


def test_nested_source_provider_versions_do_not_construct_or_load_unrequested_factories(
    monkeypatch, source_registry
):
    def forbidden_constructor(**parameters):
        raise AssertionError("Provenance must not construct the basis")

    child = _provider_entry("installed-child", forbidden_constructor, "2.7")
    unrelated = _provider_entry("unrequested", forbidden_constructor, "9.0")
    _installed_factories(monkeypatch, [child, unrelated])
    record = {
        "type": "shifted",
        "origin": 72.0,
        "time_unit": "h",
        "basis": {
            "type": "shifted",
            "origin": -3.0,
            "basis": {"type": "installed-child", "notes": {"type": "unrequested"}},
        },
    }
    original = deepcopy(record)
    expected = [_provider_description(child)]
    assert source_basis_providers(record) == expected
    assert record == original
    assert child.loaded and not unrelated.loaded
    returned = source_basis_providers(record)
    returned[0]["version"] = "changed"
    assert source_basis_providers(record) == expected


def test_provider_deduplication_preserves_distinct_subtrees_and_shared_children(
    monkeypatch, source_registry
):
    class Branches:
        def __new__(cls, **parameters):
            raise AssertionError("Dependency traversal must not construct a composite")

        @staticmethod
        def source_basis_dependencies(parameters):
            return parameters["children"]

    def leaf(**parameters):
        raise AssertionError("Dependency traversal must not construct a leaf")

    branch = _provider_entry("branches", Branches, "1.0")
    first = _provider_entry("first", leaf, "2.0")
    second = _provider_entry("second", leaf, "3.0")
    _installed_factories(monkeypatch, [branch, first, second])
    left = {"type": "branches", "children": [{"type": "first"}]}
    right = {"type": "branches", "children": [{"type": "second"}]}
    record = {"type": "branches", "children": [left, right, left]}
    assert source_basis_providers(record) == [
        _provider_description(branch),
        _provider_description(first),
        _provider_description(second),
    ]
    assert all(provider.loaded for provider in (branch, first, second))


def test_explicit_requested_and_effective_roots_retain_an_alias_provider_without_construction(
    monkeypatch, source_registry
):
    constructions = []

    def alias(*, knots, time_unit):
        constructions.append("alias")
        return P1Basis(knots, time_unit=time_unit)

    provider = _provider_entry("alias", alias, "4.2")
    _installed_factories(monkeypatch, [provider])
    requested = {"type": "alias", "knots": [0.0, 1.0], "time_unit": "h"}
    effective = {"type": "p1", "knots": [0.0, 1.0], "time_unit": "h", "endpoint": "free"}
    assert source_basis_providers(requested, effective, requested) == [
        _provider_description(provider)
    ]
    assert constructions == []


@pytest.mark.parametrize(
    "dependencies,error,match",
    [
        (None, TypeError, "finite sequence"),
        ({"type": "gaussian"}, TypeError, "finite sequence"),
        ("child", TypeError, "finite sequence"),
        ([3], TypeError, "mappings"),
        ([{"type": "unknown-child"}], ValueError, "Unknown source basis"),
        ([{"parameters": []}], KeyError, "type"),
    ],
)
def test_explicit_source_dependencies_are_validated_without_pruning(
    dependencies, error, match, source_registry
):
    class Composite:
        @staticmethod
        def source_basis_dependencies(parameters):
            return parameters["children"]

    register_source_basis("declared-children", Composite)
    with pytest.raises(error, match=match):
        source_basis_providers({"type": "declared-children", "children": dependencies})


def test_source_dependency_hook_must_be_callable(source_registry):
    class BrokenComposite:
        source_basis_dependencies = None

    register_source_basis("broken-dependencies", BrokenComposite)
    with pytest.raises(TypeError, match="must be callable"):
        source_basis_providers({"type": "broken-dependencies"})


def test_source_provider_cycles_fail_without_rejecting_local_leaves(source_registry):
    class Cycle:
        @staticmethod
        def source_basis_dependencies(parameters):
            return [{"type": "cycle", "label": parameters["label"]}]

    register_source_basis("cycle", Cycle)
    with pytest.raises(ValueError, match="cycle"):
        source_basis_providers({"type": "cycle", "label": "same-source"})
    record = {"type": "gaussian"}
    record["self"] = record
    with pytest.raises(ValueError, match="acyclic serialized JSON"):
        source_basis_providers(record)
    register_source_basis("local-leaf", P1Basis)
    assert source_basis_providers({"type": "local-leaf"}) == []
    assert source_basis_providers({"type": "p1"}) == []


def test_source_provenance_uses_finite_json_without_restricting_basis_loader(source_registry):
    record = {"type": "gaussian", "centers": np.array([0.0, 1.0]), "width": 0.2, "time_unit": "h"}
    assert source_basis_from_record(record).size == 3
    with pytest.raises(ValueError, match="serialized JSON"):
        source_basis_providers(record)
    with pytest.raises(ValueError, match="serialized JSON"):
        source_basis_providers({"type": "gaussian", "width": float("inf")})
