"""Функционал обратной задачи, производные и взвешенный спектр чувствительности."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from adrkit.contracts import MapCapabilities, ScalarCapabilities
from adrkit.errors import CapabilityError
from adrkit.spaces import ArraySpace, MatrixMap
from adrkit.sources import P1Basis
from adrkit.selection import Selection
from adrkit.inverse.objectives import compose_maps, make_objective, evaluate_score
from adrkit.inverse.regularization import p1_matrices, QuadraticPenalty, CoefficientMetric, build_penalty
from adrkit.inverse.misfit import covariance_metric, QuadraticMisfit, GaussianLikelihood, working_precision, working_quadratic
from adrkit.inverse.optimizers import solve_lbfgsb
from adrkit.inverse.sensitivity import sensitivity_spectrum, matrix_spectrum, SensitivityLimits


def space(n, label='q', **kw):
    return ArraySpace((n,), axes=(label,), coordinates=(tuple(range(n)),), units=label, **kw)


def problem():
    xspace, yspace = space(2), space(3, 'y')
    A = np.array([[1., .3], [-.2, 2.], [.5, -.7]])
    Sigma = np.array([[2., .2, .3], [.2, 1., .1], [.3, .1, .7]])
    G = np.array([[1.5, .2], [.2, .8]])
    center, y = np.array([.4, -.2]), np.array([1., .1, -.3])
    prediction = MatrixMap(A, xspace, yspace).capabilities()
    return prediction, A, Sigma, G, center, y


@pytest.mark.parametrize('tau', [0., .7, 2.])
def test_p1_exact_constant_affine_and_prior(tau):
    knots = np.array([0., .13, .7, 2.])
    basis = P1Basis(knots, time_unit='s')
    M, K = p1_matrices(basis)
    affine = 1.+2*knots
    assert_allclose(np.ones(4)@M@np.ones(4), 2.)
    assert_allclose(np.ones(4)@K@np.ones(4), 0., atol=1e-14)
    assert_allclose(affine@M@affine, 62/3)
    assert_allclose(affine@K@affine, 8.)
    penalty = QuadraticPenalty(np.full(4, .3), M+tau*tau*K, .8)
    assert_allclose(penalty.value(affine+.3), .4*(62/3+tau*tau*8))
    assert_allclose(penalty.cotangent(affine+.3), .8*(M+tau*tau*K)@affine)
    built = build_penalty('h1', basis, np.full(4,.3), .8, tau=tau, domain=space(4))
    assert_allclose(evaluate_score(built, affine+.3), penalty.value(affine+.3))


def test_objective_nonzero_prior_manual_taylor_riesz_optimizer():
    p, A, S, G, center, y = problem()
    metric = covariance_metric(S, layout=p.codomain)
    objective = make_objective(p, None, QuadraticMisfit(y, metric, normalization=3.),
                               QuadraticPenalty(center, G, .7))
    x, h = np.array([.8,.5]), np.array([.2,-.3])
    r, d = A@x-y, x-center
    expected = .5*r@np.linalg.solve(S,r)/3+.35*d@G@d
    g = A.T@np.linalg.solve(S,r)/3+.7*G@d
    assert_allclose(evaluate_score(objective,x), expected, rtol=1e-14)
    assert_allclose(objective.cotangent.cotangent(x), g, rtol=1e-14)
    remainders = [abs(evaluate_score(objective,x+e*h)-expected-e*g@h) for e in (.01,.005,.0025)]
    assert_allclose(np.array(remainders[:-1])/remainders[1:], 4., rtol=1e-7)
    riesz = CoefficientMetric(G, domain=p.domain)
    assert_allclose(riesz.apply(riesz.solve(g)), g)
    assert not np.allclose(riesz.solve(g), g)
    H, rhs = A.T@np.linalg.solve(S,A)/3+.7*G, A.T@np.linalg.solve(S,y)/3+.7*G@center
    answer = solve_lbfgsb(objective,x,options={'gtol':1e-12,'ftol':1e-15})
    assert answer.success
    assert_allclose(answer.x, np.linalg.solve(H,rhs), atol=1e-8)


def test_selection_covariance_before_inverse_hidden_nan():
    p, A, S, G, center, y = problem()
    selection = Selection([2,0],p.codomain)
    y[1] = np.nan
    selected_cov = selection.take_covariance(S)
    loss = QuadraticMisfit(selection.take(y), covariance_metric(selected_cov,layout=selection.selected_layout))
    objective = make_objective(p,selection,loss)
    x = np.array([.2,.5]); r = (A@x-y)[[2,0]]
    assert_allclose(evaluate_score(objective,x), .5*r@np.linalg.solve(selected_cov,r))
    assert_allclose(objective.cotangent.cotangent(x), A[[2,0]].T@np.linalg.solve(selected_cov,r))
    assert not np.allclose(np.linalg.inv(selected_cov), np.linalg.inv(S)[np.ix_([2,0],[2,0])])


class Nonlinear:
    """Покомпонентный квадрат со счётчиком прямых вычислений."""
    def __init__(self): self.calls = 0
    def predict(self,x): self.calls += 1; return x*x
    def jvp(self,at,h): return 2*at*h
    def vjp(self,at,g): return 2*at*g


def test_nonlinear_composition_jvp_vjp_fresh_binding():
    s = space(2); operation=Nonlinear()
    nonlinear=MapCapabilities(operation,s,s,operation,operation)
    A=np.array([[1.,2.],[-.3,.5]])
    combined=compose_maps(nonlinear,MatrixMap(A,s,s).capabilities())
    x=np.array([.2,.8]); h=np.array([.1,-.4]); z=np.array([.7,-.2])
    assert operation.calls == 0
    assert_allclose(combined.predict.predict(x), A@(x*x))
    assert_allclose(combined.jvp.jvp(x,h), A@(2*x*h))
    assert_allclose(combined.vjp.vjp(x,z), 2*x*(A.T@z))
    assert_allclose(z@combined.jvp.jvp(x,h), h@combined.vjp.vjp(x,z))
    assert_allclose(combined.jvp.jvp(3*x,h), 3*A@(2*x*h))


def test_capabilities_value_only_nonsmooth_no_prox_or_forward_on_rejection():
    s=space(2); op=Nonlinear()
    p=MapCapabilities(op,s,s,op,None)
    loss=working_quadratic([0.,0.],[1.,2.],layout=s)
    objective=make_objective(p,None,loss)
    assert objective.cotangent is None and objective.prox is None
    with pytest.raises(CapabilityError): solve_lbfgsb(objective,[1.,2.])
    assert op.calls == 0
    assert evaluate_score(objective,[1.,2.]) == 16.5
    class L1:
        def value(self,x): return np.abs(x).sum()
    smooth=MapCapabilities(op,s,s,op,op)
    nonsmooth=make_objective(smooth,None,loss,ScalarCapabilities(L1(),s))
    with pytest.raises(CapabilityError): solve_lbfgsb(nonsmooth,[1.,2.])
    assert nonsmooth.prox is None
    class Support:
        def value(self,x): return np.inf
    assert evaluate_score(ScalarCapabilities(Support(),s),[1.,2.]) == np.inf


def test_layout_rejection_and_ownership():
    p,A,S,G,center,y=problem()
    loss=QuadraticMisfit(y,covariance_metric(S,layout=p.codomain))
    penalty=QuadraticPenalty(center,G,.5)
    original=make_objective(p,None,loss,penalty)
    score=evaluate_score(original,[.1,.2])
    A[:]=S[:]=G[:]=100; center[:]=y[:]=100
    assert evaluate_score(original,[.1,.2]) == score
    wrong=space(2,binding='different')
    with pytest.raises(ValueError):
        make_objective(p,None,loss,ScalarCapabilities(penalty,wrong,penalty))
    with pytest.raises(ValueError): compose_maps(p,p)


def test_gaussian_density_working_weights_and_invalid_metrics():
    p,A,S,G,center,y=problem()
    metric=covariance_metric(S,layout=p.codomain)
    gaussian=GaussianLikelihood(y,metric)
    assert_allclose(gaussian.value(y), .5*(3*np.log(2*np.pi)+np.linalg.slogdet(S)[1]))
    for invalid in (np.ones((3,3)), np.diag([1.,-1.,1.]), np.array([[1.,1.,0.],[0.,1.,0.],[0.,0.,1.]])):
        with pytest.raises(ValueError): covariance_metric(invalid,layout=p.codomain)
    with pytest.raises(ValueError): QuadraticPenalty(center,-np.eye(2),1.)
    with pytest.raises(ValueError): QuadraticMisfit(y,metric,normalization=0.)


@pytest.mark.parametrize('covariance, expected_factor', [
    (np.array([4., 9., 16.]), np.array([2., 3., 4.])),
    (np.array([[4., 2., 0.], [2., 10., 3.], [0., 3., 5.]]),
     np.array([[2., 0., 0.], [1., 3., 0.], [0., 1., 2.]])),
])
def test_covariance_factor_is_owned_and_preserves_precision(covariance, expected_factor):
    supplied = covariance.copy()
    metric = covariance_metric(supplied, layout=space(3, 'observation'))
    residual = np.array([.7, -1.2, .3])
    dense_covariance = np.diag(covariance) if covariance.ndim == 1 else covariance
    lower_factor = np.diag(expected_factor) if covariance.ndim == 1 else expected_factor
    expected_precision = np.linalg.solve(dense_covariance, residual)
    expected_white = np.linalg.solve(lower_factor, residual)
    factor = metric.covariance_factor
    assert factor.flags.owndata
    assert factor.dtype == expected_factor.dtype
    assert factor.shape == expected_factor.shape
    assert factor.tobytes(order='C') == expected_factor.tobytes(order='C')
    assert not np.shares_memory(factor, supplied)
    factor[:] = supplied[:] = 99.
    fresh_factor = metric.covariance_factor
    assert fresh_factor.tobytes(order='C') == expected_factor.tobytes(order='C')
    assert not np.shares_memory(fresh_factor, factor)
    assert_allclose(metric.apply_precision(residual), expected_precision, rtol=1e-14, atol=1e-14)
    assert_allclose(metric.whiten(residual), expected_white, rtol=1e-14, atol=1e-14)


def test_working_precision_owns_callback_arguments_and_result():
    supplied = np.array([1., 2.])
    shared = np.zeros(2)
    def action(residual):
        assert_array_equal(residual, [1., 2.])
        shared[:] = [2*residual[0], 3*residual[1]]
        residual[:] = 99.
        return shared
    metric = working_precision(action, layout=space(2))
    precision = metric.apply_precision(supplied)
    shared[:] = 99.
    assert_array_equal(supplied, [1., 2.])
    assert_array_equal(precision, [2., 6.])
    assert not np.shares_memory(precision, shared)


def test_matrix_spectrum_matches_independent_correlated_dense_svd():
    _, matrix, covariance, gram, _, _ = problem()
    lower = np.linalg.cholesky(covariance)
    root = np.linalg.cholesky(gram).T
    expected = np.linalg.svd(np.linalg.solve(lower, matrix) @ np.linalg.inv(root), compute_uv=False)
    result = matrix_spectrum(matrix, covariance, gram)
    assert_allclose(result.singular_values, expected, rtol=1e-14, atol=1e-14)
    assert result.rank == 2 and result.nullity == 0
    assert_allclose(result.threshold, 1e-10*expected[0], rtol=1e-14)
    assert_allclose(result.condition, expected[0]/expected[-1], rtol=1e-14)
    assert result.condition_reason is None


def test_sensitivity_metric_svd_limits_rank_units():
    p,A,S,G,center,y=problem()
    metric=covariance_metric(S,layout=p.codomain)
    def spectrum(p,A,S,G):
        return sensitivity_spectrum(p,[.2,.3],coefficient_metric=CoefficientMetric(G,domain=p.domain),
            observation_metric=covariance_metric(S,layout=p.codomain),limits=SensitivityLimits(2,6,2))
    result=spectrum(p,A,S,G)
    L=np.linalg.cholesky(S); R=np.linalg.cholesky(G).T
    K=np.linalg.solve(L,A)@np.linalg.inv(R)
    assert_allclose(result.singular_values,np.linalg.svd(K,compute_uv=False),rtol=1e-14)
    assert result.rank == 2 and result.nullity == 0
    converted=MatrixMap(A*5/3,p.domain,p.codomain).capabilities()
    assert_allclose(spectrum(converted,A*5/3,S*25,G/9).singular_values,result.singular_values)
    zero=MatrixMap(A*0,p.domain,p.codomain).capabilities()
    empty=spectrum(zero,A*0,S,G)
    assert empty.condition is None and empty.rank == 0 and empty.nullity == 2
    for limits in (SensitivityLimits(1,6,2),SensitivityLimits(2,5,2),SensitivityLimits(2,6,1)):
        with pytest.raises(ValueError): sensitivity_spectrum(p,[.2,.3],coefficient_metric=CoefficientMetric(G,domain=p.domain),observation_metric=metric,limits=limits)
    with pytest.raises(CapabilityError):
        sensitivity_spectrum(p,[.2,.3],coefficient_metric=CoefficientMetric(G,domain=p.domain),
            observation_metric=working_precision(lambda r:r,layout=p.codomain),limits=SensitivityLimits(2,6,2))


def test_linked_support_whitening_preserves_observed_gls():
    from adrkit.inverse.misfit import SupportWhitening
    lift=np.array([[1.,0.],[.3,.7],[0.,1.]])
    variance=np.array([.4,.7]); A=np.array([[1.,.2],[-.3,2.]])
    y=np.array([.4,-.7]); x=np.array([.2,.8])
    white=SupportWhitening(lift,variance)
    B,z=white.whiten_matrix(lift@A),white.whiten(lift@y)
    r=A@x-y
    assert_allclose(np.sum((B@x-z)**2), np.sum(r*r/variance),rtol=1e-14)
    with pytest.raises(ValueError): white.whiten(np.array([0.,1.,0.]))
    with pytest.raises(ValueError): SupportWhitening(np.ones((3,2)),variance)


@pytest.mark.parametrize('has_jvp,has_vjp', [(True,False),(False,True),(False,False),(True,True)])
def test_map_capabilities_stay_independent(has_jvp,has_vjp):
    s=space(2); op=Nonlinear()
    p=MapCapabilities(op,s,s,op if has_jvp else None,op if has_vjp else None)
    combined=compose_maps(p,p)
    assert (combined.jvp is not None)==has_jvp
    assert (combined.vjp is not None)==has_vjp
    assert op.calls==0


def test_invalid_sensitivity_rejected_before_jvp():
    s=space(2); op=Nonlinear(); p=MapCapabilities(op,s,s,None,op)
    with pytest.raises(CapabilityError):
        sensitivity_spectrum(p,[1.,2.],coefficient_metric=CoefficientMetric(np.eye(2),domain=s),
            observation_metric=covariance_metric(np.ones(2),layout=s),limits=SensitivityLimits(2,4,2))
    assert op.calls==0


def test_zero_alpha_psd_penalty_and_no_optional_imports():
    import sys
    s=space(2); G=np.array([[1.,-1.],[-1.,1.]])
    penalty=QuadraticPenalty([.2,.3],G,0.)
    assert penalty.value([1.,2.])==0.
    assert_array_equal(penalty.cotangent([1.,2.]),[0.,0.])
    assert not any(name in sys.modules for name in ('odl','cvxpy','sympy','torch'))
    loss=QuadraticMisfit([0.,0.],covariance_metric([1.,2.],layout=s))
    with pytest.raises(AttributeError): loss.normalization=2.
    with pytest.raises(AttributeError): loss.metric.layout=space(2,binding='different')


def test_optimizer_rejects_bad_bounds_before_prediction():
    s=space(2); op=Nonlinear()
    p=MapCapabilities(op,s,s,op,op)
    obj=make_objective(p,None,QuadraticMisfit([0.,0.],covariance_metric([1.,1.],layout=s)))
    with pytest.raises(ValueError): solve_lbfgsb(obj,[1.,2.],bounds=[(2.,1.),(0.,None)])
    assert op.calls==0
