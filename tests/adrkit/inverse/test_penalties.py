"""Независимые интегралы полной H¹-нормы, ненулевые центры и различие ковектора с представителем Рисса."""
import numpy as np
import pytest

from adrkit.contracts import ScalarCapabilities
from adrkit.inverse.misfit import working_quadratic
from adrkit.inverse.objectives import make_objective
from adrkit.inverse.regularization import CoefficientMetric, build_penalty
from adrkit.sources import P1Basis, matrix_source_map
from adrkit.spaces import ArraySpace


def space():
    return ArraySpace((3,), axes=('coefficient',), coordinates=((0., .5, 2.),), units='kg/s')


@pytest.mark.parametrize('kind,tau', [('l2',0.), ('h1',.7)])
def test_nonzero_prior_exact_affine_integrals(kind, tau):
    r"""Проверить штраф относительно ненулевого центра по независимым интегралам.

    Notes
    -----
    Для разности с центром :math:`d(t)=1+2t` на интервале от 0 до 2
    интегралы значений и производной равны

    .. math::

        \int_0^2 d(t)^2\,dt=\frac{62}{3},\qquad
        \int_0^2 d'(t)^2\,dt=8.
    """
    domain = space()
    prior = np.array([.3, -.2, .8])
    p = build_penalty(kind, P1Basis([0,.5,2], time_unit='s'), prior, .4, tau=tau, domain=domain)
    x = prior + np.array([1.,2.,5.])
    np.testing.assert_allclose(p.value.value(x), .2*(62/3+tau*tau*8), rtol=1e-14)
    np.testing.assert_array_equal(p.cotangent.cotangent(prior), np.zeros(3))
    # Постоянная разность отличает полную норму от одного штрафа на производную.
    np.testing.assert_allclose(p.value.value(prior+1), .4, rtol=1e-14)


class QuarticPenalty:
    r"""Четвертичный штраф относительно заданного центра.

    Parameters
    ----------
    prior : ndarray of shape (n,)
        Центр штрафа в форме вектора коэффициентов; значения копируются.

    Notes
    -----
    Для центра :math:`c` вычисляются

    .. math::

        R(x)=0.075\sum_i(x_i-c_i)^4,\qquad
        (dR(x))_i=0.3(x_i-c_i)^3.
    """
    def __init__(self, prior):
        self.prior = prior.copy()

    def value(self, x):
        return .075*float(np.sum((x-self.prior)**4))

    def cotangent(self, x):
        return .3*(x-self.prior)**3


def test_external_penalty_weighted_objective_taylor_and_riesz():
    domain = space()
    A = np.array([[1.,2.,0.], [0.,1.,-1.], [2.,0.,1.]])
    mapping = matrix_source_map(A, scale=1.7, domain=domain, codomain=domain)
    target, weights = np.array([.2,-.5,.7]), np.array([.4,2.,3.])
    misfit = working_quadratic(target, weights, layout=domain, normalization=5.)
    prior = np.array([.1,.2,-.3])
    external = QuarticPenalty(prior)
    objective = make_objective(mapping, None, misfit, ScalarCapabilities(external, domain, external))
    x, h = np.array([.6,-.4,.9]), np.array([.2,.3,-.1])
    residual = 1.7*A@x-target
    expected = 1.7*A.T@(weights*residual)/5 + .3*(x-prior)**3
    g = objective.cotangent.cotangent(x)
    np.testing.assert_allclose(g, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(objective.value.value(x), .5*(weights*residual)@residual/5 + .075*np.sum((x-prior)**4), rtol=1e-14)
    errors = [abs(objective.value.value(x+e*h)-objective.value.value(x)-e*(g@h)) for e in (.02,.01,.005)]
    assert all(3.9 < a/b < 4.1 for a,b in zip(errors,errors[1:]))
    G = np.array([[2.,.3,0.], [.3,3.,.2], [0.,.2,4.]])
    metric = CoefficientMetric(G, domain=domain)
    riesz = metric.solve(g)
    np.testing.assert_allclose(h@G@riesz, h@g, rtol=1e-14)
    assert not np.allclose(riesz, g)
