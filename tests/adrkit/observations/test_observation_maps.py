"""Порядок квадратур по приёмникам и внешний нелинейный оператор наблюдений."""
from copy import deepcopy

import numpy as np

from adrkit.contracts import MapCapabilities
from adrkit.inverse.objectives import compose_maps
from adrkit.predictions import grid_observation
from adrkit.observations.grid import observation_map
from adrkit.selection import Selection
from adrkit.spaces import ArraySpace


def test_grid_weights_row_order_and_covector_have_no_extra_metric():
    domain = ArraySpace((3,2), axes=('time','node'), coordinates=((0,1,2),('L','R')), units='kg/m3')
    output = ArraySpace((4,), axes=('observation',), coordinates=((('B','mean'),('B','snapshot'),('A','mean'),('A','snapshot')),), units='kg/m3')
    S, W = np.array([[2.,.5],[-1.,3.]]), np.array([[.2,.3,.5],[0.,1.,0.]])
    mapping = observation_map(S, W, domain=domain, codomain=output)
    x, z = np.arange(6.).reshape(3,2), np.array([.2,-1.,2.,.5])
    expected = np.array([sum(W[k,t]*S[r,j]*x[t,j] for t in range(3) for j in range(2)) for r in range(2) for k in range(2)])
    pullback = np.array([[sum(z[2*r+k]*S[r,j]*W[k,t] for r in range(2) for k in range(2)) for j in range(2)] for t in range(3)])
    S[:] = W[:] = 999
    np.testing.assert_allclose(mapping.predict.predict(x), expected, rtol=1e-14)
    np.testing.assert_allclose(mapping.vjp.vjp(x,z), pullback, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z@expected, np.sum(x*pullback), rtol=1e-14)
    selection = Selection([3,0], output)
    assert selection.selected_layout.coordinates == ((('A','snapshot'),('B','mean')),)
    np.testing.assert_array_equal(selection.scatter([7.,8.]), [8.,0.,0.,7.])
    np.testing.assert_array_equal(selection.take([8.,np.nan,np.nan,7.]), [7.,8.])


def test_copied_grid_observation_owns_both_weights_and_all_operations():
    domain = ArraySpace((3, 2), axes=("time", "node"),
        coordinates=((0, 1, 2), (0, 1)), units="C")
    codomain = ArraySpace((1,), axes=("observation",), coordinates=((0,),), units="C")
    original = observation_map([[1., 2.]], [[0., 1., 0.]],
                               domain=domain, codomain=codomain)
    owned = deepcopy(original)
    assert owned.predict is owned.jvp is owned.vjp
    assert owned.predict is not original.predict
    original.predict._space_weights[:] = 0
    original.predict._time_weights[:] = 0
    at = np.arange(6.).reshape(3, 2)
    np.testing.assert_array_equal(owned.predict.predict(at), [8.])
    np.testing.assert_array_equal(owned.jvp.jvp(at, np.ones((3, 2))), [3.])
    np.testing.assert_array_equal(owned.vjp.vjp(at, [3.]), [[0., 0.], [3., 6.], [0., 0.]])


def test_grid_factory_preserves_integral_units_and_coordinate_binding():
    domain = ArraySpace((3, 2), axes=("time", "node"),
        coordinates=((0, 1, 2), (0, 1)), units="C")
    codomain = ArraySpace((2,), axes=("functional",),
        coordinates=(("integral", "contrast"),), units="C*km^2*h", scale=2., binding="quadrature")
    observation = grid_observation([[2., -1.], [.5, 3.]], [[1., -2., 1.]],
                                   state_space=domain, codomain=codomain)
    at = np.array([[1., -2.], [3., 4.], [-5., 6.]])
    direction = np.array([[2., 3.], [-1., .5], [4., -2.]])
    np.testing.assert_array_equal(observation.predict(at), [-16., -17.])
    np.testing.assert_array_equal(observation.jvp(at, direction), [16., 4.])
    np.testing.assert_array_equal(observation.vjp(at, [2., -1.]), [[3.5, -5.], [-7., 10.], [3.5, -5.]])
    assert observation.codomain is codomain


class SquaredSensor:
    """Покомпонентный квадрат с прямой и сопряжённой производными."""
    def predict(self, x):
        return x**2

    def jvp(self, at, h):
        return 2*at*h

    def vjp(self, at, z):
        return 2*at*z


def test_external_nonlinear_observation_composes_at_each_point():
    layout = ArraySpace((2,), axes=('sensor',), coordinates=(('B','A'),), units='1')
    sensor = SquaredSensor()
    caps = MapCapabilities(sensor, layout, layout, sensor, sensor)
    mapping = compose_maps(caps, caps)
    x, h, z = np.array([.7,1.2]), np.array([-.4,.2]), np.array([2.,-.3])
    np.testing.assert_allclose(mapping.predict.predict(x), x**4, rtol=1e-14)
    np.testing.assert_allclose(mapping.jvp.jvp(x,h), 4*x**3*h, rtol=1e-14)
    np.testing.assert_allclose(mapping.vjp.vjp(x,z), 4*x**3*z, rtol=1e-14)
