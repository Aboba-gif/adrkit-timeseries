"""Четыре независимо реализованных семейства подключаются через программный интерфейс."""
from pathlib import Path

import numpy as np
import pytest

from adrkit.contracts import MapCapabilities, ScalarCapabilities
from adrkit.config.composition import Family, load
from tests.configuration.fixtures.composition.adr_actions import assemble, run
from adrkit.config.schema import fields
from adrkit.sources import sampled_source_map
from adrkit.spaces import ArraySpace


class Polynomial:
    """Базис из константы, линейного и квадратичного полиномов."""
    def values(self,t):
        return np.column_stack((np.ones_like(t),t,t*t))


class Sine:
    """Покомпонентный синус с прямой и сопряжённой производными."""
    def predict(self,x):
        return np.sin(x)

    def jvp(self,at,h):
        return np.cos(at)*h

    def vjp(self,at,z):
        return np.cos(at)*z


class Reverse:
    """Перестановка координат в обратном порядке."""
    def predict(self,x):
        return x[::-1].copy()

    def jvp(self,at,h):
        return h[::-1].copy()

    def vjp(self,at,z):
        return z[::-1].copy()


class Quartic:
    r"""Четвертичный штраф с центром 0.2.

    Notes
    -----
    Значение и координатный ковектор равны

    .. math::

        R(x)=\frac14\sum_i(x_i-0.2)^4,\qquad
        (dR(x))_i=(x_i-0.2)^3.
    """
    def value(self,x):
        return float(np.sum((x-.2)**4))/4

    def cotangent(self,x):
        return (x-.2)**3


def empty(p,*,partial):
    return dict(fields(p,set(),partial=partial))


def source(p):
    times = np.linspace(0,2,5)
    d = ArraySpace((3,),axes=('power',),coordinates=((0,1,2),),units='1')
    y = ArraySpace((5,),axes=('time',),coordinates=(times,),units='1')
    b = Polynomial()
    return b,sampled_source_map(b,times,scale=1.5,domain=d,codomain=y)


def surrogate(p,d):
    op = Sine()
    return MapCapabilities(op,d,d,op,op)


def observation(p,d):
    y = ArraySpace(d.shape,axes=d.axes,coordinates=(d.coordinates[0][::-1],),units=d.units)
    op = Reverse()
    return MapCapabilities(op,d,y,op,op)


def penalty(p,b,d):
    op = Quartic()
    return ScalarCapabilities(op,d,op)


def test_all_external_families_share_public_config_and_covector_chain():
    inputs = {role:{'external':Family(empty,builder)} for role,builder in
              [('source',source),('surrogate',surrogate),('observation',observation),('regularizer',penalty)]}
    overrides = ['/science/'+role+'={"kind":"external","parameters":{}}' for role in inputs]
    overrides += ['/science/action="score"']
    path = Path(__file__).resolve().parent/'fixtures/composition/configs/full.json'
    config = load([str(path)],inputs=inputs,overrides=overrides)
    c = assemble(config,inputs=inputs)
    x,h = np.array([.4,-.2,.3]),np.array([.2,.1,-.1])
    t = np.linspace(0,2,5)
    B = 1.5*np.column_stack((np.ones(5),t,t*t))
    prediction = np.sin(B@x)[::-1]
    residual = prediction-np.array([1,1.5,2,1.5,1])
    expected_g = B.T@(np.cos(B@x)*residual[::-1])+(x-.2)**3
    np.testing.assert_allclose(c.prediction.predict.predict(x),prediction,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(c.objective.cotangent.cotangent(x),expected_g,rtol=1e-13,atol=1e-14)
    errors = [abs(c.objective.value.value(x+e*h)-c.objective.value.value(x)-e*expected_g@h) for e in [.002,.001,.0005]]
    assert all(3.95<a/b<4.05 for a,b in zip(errors,errors[1:]))
    assert run(config,mode='smoke',inputs=inputs)['result']['capabilities']['vjp']
    # Дополнения передаются одному вызову; неявная регистрация и поиск при импорте отсутствуют.
    with pytest.raises(ValueError,match='unsupported'):
        load([str(path)],overrides=overrides)
