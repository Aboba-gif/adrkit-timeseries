"""Независимые адаптивные интегралы для гауссова отклика P1-источника."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.integrate import quad

from adrkit.backends.whole_space import Transport, WholeSpacePredictor, observation_matrix
from adrkit.observations.analytic import SensorSpec, SpatialSource, Quadrature, p1_response
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace


def spaces(basis, count):
    return dict(domain=ArraySpace((basis.size,), axes=('coefficient',),
                                 coordinates=(basis.knots,), units='q'),
                codomain=ArraySpace((count,), axes=('observation',),
                                   coordinates=(tuple(range(count)),), units='c'))


@pytest.mark.parametrize('windows', [None, [.6, 1.5, .8, .4]])
def test_kernel_hat_quadrature_against_independent_integrals(windows):
    basis = P1Basis([0., .3, 1.2, 2.], time_unit='s')
    times = [1.4, .4, 2.3, 0.]
    receivers = [[.2, -.1], [0., 0.], [-.3, .2], [0., 0.]]
    transport = Transport(.15, (.2, -.1), .12, .7)
    sensor = SensorSpec(times, receivers, windows=windows)
    scale = 2.3
    caps = p1_response(basis, sensor, transport, SpatialSource([0., 0.]),
                       quadrature=Quadrature(24), coefficient_scale=scale, **spaces(basis, 4))
    theta = np.array([.4, -.2, .6, .3])
    def reference_snapshot(time, receiver, coefficients):
        total = 0.
        for j, (left, right) in enumerate(zip(basis.knots[:-1], basis.knots[1:])):
            if time <= left:
                continue
            def integrand(source_time):
                lag = time-source_time
                variance = .7**2+2*.15*lag
                radius = (receiver[0]-.2*lag)**2+(receiver[1]+.1*lag)**2
                kernel = np.exp(-.12*lag-radius/(2*variance))/(2*np.pi*variance)
                q = coefficients[j]+(coefficients[j+1]-coefficients[j])*(source_time-left)/(right-left)
                return kernel*q
            total += quad(integrand, left, min(time, right), epsabs=1e-13, epsrel=1e-13)[0]
        return total
    reference = []
    for i, (time, receiver) in enumerate(zip(times, receivers)):
        if windows is None:
            value = reference_snapshot(time, receiver, theta)
        else:
            lower = max(0., time-windows[i])
            breaks = [k for k in basis.knots if lower < k < time]
            value = quad(lambda t: reference_snapshot(t, receiver, theta), lower, time,
                         points=breaks, epsabs=1e-13, epsrel=1e-13)[0]/windows[i]
        reference.append(scale*value)
    prediction = caps.predict.predict(theta)
    assert_allclose(prediction, reference, rtol=5e-13, atol=5e-13)
    assert prediction[-1] == 0
    h, z = np.arange(4.)-.4, np.arange(4.)+.3
    matrix = caps.predict.matrix
    assert_allclose(caps.jvp.jvp(theta, h), matrix @ h, rtol=5e-13, atol=5e-13)
    assert_allclose(caps.vjp.vjp(theta, z), matrix.T @ z, rtol=5e-13, atol=5e-13)
    assert_allclose(caps.jvp.jvp(theta, h) @ z, h @ caps.vjp.vjp(theta, z), rtol=5e-13, atol=5e-13)
    matrix[:] = 0
    receivers[0][:] = [999, 999]
    assert_array_equal(caps.predict.predict(theta), prediction)


def test_p1_end_hats_constant_matches_p0_response_and_scale():
    basis = P1Basis([0., .3, 1.2, 2.], time_unit='s')
    times = [2., .2, 1.2, 0.]
    receivers = [[0., 0.]]*4
    transport = Transport(.15, (.2, -.1), .12, .7)
    caps = p1_response(basis, SensorSpec(times, receivers), transport, SpatialSource((0., 0.)),
                       quadrature=Quadrature(), coefficient_scale=3., **spaces(basis, 4))
    expected = observation_matrix(times, receivers, [0., 0.], basis.knots, transport) @ np.ones(3)
    assert_allclose(caps.predict.predict(np.ones(4)), 3*expected, rtol=5e-13, atol=5e-13)
    assert caps.predict.matrix[-2, -1] == 0  # Последняя базисная функция равна нулю до узла 1.2.
    assert caps.predict.matrix[0, -1] > 0   # Свободный последний конец сохраняется.
    domain = ArraySpace((3,), axes=('bin',), coordinates=((0, 1, 2),), units='q')
    predictor = WholeSpacePredictor(times, receivers, [0., 0.], basis.knots, transport,
                                   domain=domain, codomain=caps.codomain)
    assert_array_equal(predictor.predict(np.ones(3)), expected)
    assert not hasattr(predictor, 'states')
    with pytest.raises(ValueError):
        Transport(.15, (.2, -.1), .12, 0.)
