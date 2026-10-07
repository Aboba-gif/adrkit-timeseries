"""Привязка кеша к точке, снимки и защищённые линейные операторы в фиксированной точке."""
import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.state_maps import state_map
from adrkit.backends.cached import CachedSolver
from adrkit.spaces import ArraySpace


@pytest.mark.parametrize('solver',[BoundedSolver,CachedSolver])
def test_exact_at_change_and_linearization_survive_mutation(small_protocol,solver,monkeypatch):
    backend = solver(small_protocol['model'],(3,3),4)
    domain = ArraySpace((5,),axes=('time',),coordinates=(backend.times,),units='kg/s')
    output = ArraySpace((5,9),axes=('time','node'),coordinates=(backend.times,tuple(range(9))),units='kg/m3')
    mapping = state_map(backend,domain=domain,codomain=output)
    calls = []
    original = solver.solve_controls
    def counted(self,controls):
        calls.append(np.array(controls,copy=True))
        return original(self,controls)
    monkeypatch.setattr(solver,'solve_controls',counted)
    x = np.linspace(.1,.5,5)
    h, z = np.cos(np.arange(5)), np.sin(np.arange(45)).reshape(5,9)
    states = mapping.predict.predict(x)
    tangent, pullback = mapping.jvp.jvp(x,h), mapping.vjp.vjp(x,z)
    linear = mapping.predict.linear_operator(x)
    assert len(calls)==1
    np.testing.assert_allclose(np.sum(z*tangent),h@pullback,rtol=1e-10,atol=1e-12)
    states[:] = 999
    backend.load[:] = 0
    small_protocol['model']['diffusion'] = 99
    np.testing.assert_array_equal(mapping.jvp.jvp(x,h),tangent)
    shifted = np.nextafter(x,np.inf)
    mapping.predict.predict(shifted)
    assert len(calls)==2
    np.testing.assert_array_equal(calls[-1],shifted)
    x[:] = -4
    mapping.predict.predict(x)
    np.testing.assert_array_equal(linear.matvec(h).reshape(output.shape),tangent)
    np.testing.assert_array_equal(linear.rmatvec(z.ravel()),pullback)
    np.testing.assert_allclose(mapping.jvp.jvp(x,2*h),2*mapping.jvp.jvp(x,h),rtol=1e-10,atol=1e-12)


def test_foreign_time_coordinates_reject_before_solve(small_protocol,monkeypatch):
    backend = BoundedSolver(small_protocol['model'],3,4)
    def forbidden(*args):
        raise AssertionError('unexpected solve')
    monkeypatch.setattr(BoundedSolver,'solve_controls',forbidden)
    domain = ArraySpace((5,),axes=('time',),coordinates=((0,.1,.2,.3,1),),units='q')
    output = ArraySpace((5,9),axes=('time','node'),coordinates=(backend.times,tuple(range(9))),units='c')
    with pytest.raises(ValueError,match='time nodes'):
        state_map(backend,domain=domain,codomain=output)
