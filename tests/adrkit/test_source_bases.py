"""Независимые проверки внешнего базиса и переноса производной источника."""
import numpy as np
import pytest

from adrkit.sources import P1Basis, sampled_source_map
from adrkit.spaces import ArraySpace


class PolynomialBasis:
    def values(self, times):
        return np.column_stack((np.ones_like(times), times, times**2))


def test_external_basis_scale_duality_and_owned_samples():
    times = np.array([0., .25, 1.5, 2.])
    domain = ArraySpace((3,), axes=('coefficient',), coordinates=((0, 1, 2),), units='1')
    output = ArraySpace((4,), axes=('time',), coordinates=(times,), units='kg/s')
    basis = PolynomialBasis()
    mapping = sampled_source_map(basis, times, scale=2.5, domain=domain, codomain=output)
    x, h, z = np.array([.4, -.2, .3]), np.array([.1, .7, -.4]), np.array([2., -1., .5, 3.])
    expected = 2.5*np.array([[1, 0, 0], [1, .25, .0625], [1, 1.5, 2.25], [1, 2, 4]])
    times[:] = -99
    basis.values = lambda t: np.zeros((len(t), 3))
    np.testing.assert_allclose(mapping.predict.predict(x), expected@x, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(mapping.jvp.jvp(x, h), expected@h, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(mapping.vjp.vjp(x, z), expected.T@z, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z@mapping.jvp.jvp(x, h), h@mapping.vjp.vjp(x, z), rtol=1e-14)
    returned = mapping.predict.predict(x)
    returned[:] = 123
    np.testing.assert_allclose(mapping.predict.predict(x), expected@x, rtol=1e-14, atol=1e-14)


def test_nonuniform_p1_endpoints_and_derivative_convention():
    basis = P1Basis([0, .5, 2], time_unit='s')
    np.testing.assert_array_equal(basis.values([0, .25, .5, 1.25, 2]),
                                  [[1,0,0], [.5,.5,0], [0,1,0], [0,.5,.5], [0,0,1]])
    np.testing.assert_allclose(basis.derivatives([0, .5, 2]),
                               [[-2,2,0], [0,-2/3,2/3], [0,-2/3,2/3]], rtol=1e-15)
    with pytest.raises(ValueError, match='closed P1 interval'):
        basis.values([2.000001])


@pytest.mark.parametrize('knots', [[0,0,1], [1,0], [0,float('nan')]])
def test_invalid_basis_rejected(knots):
    with pytest.raises(ValueError):
        P1Basis(knots, time_unit='s')
