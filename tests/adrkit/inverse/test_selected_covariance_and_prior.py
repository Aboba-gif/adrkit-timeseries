"""Тождества ковариации и априорного штрафа линейной обратной задачи с независимым матричным решением."""
import numpy as np

from adrkit.inverse.misfit import QuadraticMisfit, covariance_metric
from adrkit.inverse.objectives import make_objective
from adrkit.inverse.optimizers import solve_lbfgsb
from adrkit.inverse.regularization import QuadraticPenalty
from adrkit.selection import Selection
from adrkit.sources import matrix_source_map
from adrkit.spaces import ArraySpace


def test_selection_covariance_before_inverse_and_nonzero_prior_solution():
    d = ArraySpace((2,),axes=('coefficient',),coordinates=(('q0','q1'),),units='kg/s')
    y = ArraySpace((3,),axes=('observation',),coordinates=(('B','C','A'),),units='ug/m3')
    A = np.array([[1.,.2],[.3,1.],[1.2,-.4]])
    sigma = np.array([[2.,.6,.4],[.6,3.,.5],[.4,.5,1.5]])
    selection = Selection([2,0],y)
    target = selection.take([.7,np.nan,1.2])
    selected_sigma = sigma[np.ix_([2,0],[2,0])]
    metric = covariance_metric(selection.take_covariance(sigma),layout=selection.selected_layout)
    G, prior, alpha = np.array([[2.,.4],[.4,1.]]), np.array([.3,-.1]), .6
    objective = make_objective(matrix_source_map(A,scale=2,domain=d,codomain=y),selection,
        QuadraticMisfit(target,metric),QuadraticPenalty(prior,G,alpha))
    B = 2*A[[2,0]]
    P = np.linalg.inv(selected_sigma)
    assert not np.allclose(P,np.linalg.inv(sigma)[np.ix_([2,0],[2,0])])
    expected = np.linalg.solve(B.T@P@B+alpha*G,B.T@P@target+alpha*G@prior)
    solved = solve_lbfgsb(objective,[0.,0.],options={'maxiter':30,'ftol':1e-14,'gtol':1e-10})
    assert solved.success
    np.testing.assert_allclose(solved.x,expected,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(objective.cotangent.cotangent(expected),[0.,0.],rtol=0,atol=1e-14)
