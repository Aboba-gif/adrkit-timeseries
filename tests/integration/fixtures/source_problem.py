"""Составная задача восстановления источника и добавочный штраф по коэффициентам."""
import numpy as np

from adrkit import ADRProblem, ADRUnits, ArraySpace, matrix_map

from adrkit.backends.cached import CachedSolver

from adrkit.inverse import (
    InverseProblem, QuadraticMisfit, QuadraticPenalty, covariance_metric,
    p1_matrices, pullback_functional, scalar_from_functions,
    scale_functional, sum_functionals,
)

from adrkit.loads import IntervalAverageLoad

from adrkit.observations.grid import GridObservation

from adrkit.predictions import GridPrediction, solver_state_space

from adrkit.sources import P1Basis

def build_problem(observed):
    """Собрать задачу для 73 неотрицательных коэффициентов P1-источника.

    Parameters
    ----------
    observed : array_like, shape (36,)
        Концентрации в мкг/м³: по девять моментов для четырёх приёмников,
        сначала все моменты первого приёмника, затем остальных.

    Returns
    -------
    problem : InverseProblem
        Обратная задача на интервале от 0 до 3 ч с неотрицательными
        безразмерными коэффициентами источника.
    penalty : ScalarCapabilities
        Квадратичный штраф по значениям и первой производной источника.
    """
    units = ADRUnits("km", "h", "ug/m^3")
    equation = ADRProblem(
        diffusion=.018, velocity=(-.72, -.144), linear_loss=1/72,
        reaction_gamma=1/72, reaction_c_star=1., horizon=3.5,
        time_origin=-.5, units=units,
    )
    solver = equation.make_solver(CachedSolver, interior_points=3, time_steps=84)
    basis = P1Basis(np.linspace(0., 3., 73), time_unit="h")
    load = IntervalAverageLoad(
        basis, solver.times, origin=-.5, q_reference=100.,
        history_integral=lambda left, right: 0., history_initial=0.,
        source_unit=units.source,
    )
    spatial = np.ones((4, solver.size))*solver.cell_area/solver.size
    temporal = np.zeros((9, solver.nt+1))
    indices = np.rint((np.arange(1, 10)/3+.5)/solver.dt).astype(int)
    temporal[np.arange(9), indices] = 1.
    output = ArraySpace((36,), axes=("observation",),
                        coordinates=(np.arange(36),), units=units.concentration)
    observation = GridObservation(spatial, temporal,
                                  domain=solver_state_space(solver), codomain=output)
    prediction = GridPrediction(solver, load, observation)
    loss = QuadraticMisfit(observed, covariance_metric(np.eye(36), layout=output))
    mass, stiffness = p1_matrices(basis)
    penalty = QuadraticPenalty(np.zeros(73), 100.**2*(mass+.25**2*stiffness),
                               alpha=1.).capabilities(load.domain)
    return InverseProblem(prediction.capabilities(), loss.capabilities(), penalty=penalty,
                          bounds=[(0., None)]*basis.size), penalty

def regularize_source_summaries(parameter_space, base_penalty):
    r"""Добавить четвертичный штраф среднего коэффициента и разности краёв.

    Parameters
    ----------
    parameter_space : ArraySpace
        Векторное пространство коэффициентов с не менее чем двумя элементами.
    base_penalty : ScalarCapabilities
        Исходный штраф в том же пространстве коэффициентов.

    Returns
    -------
    ScalarCapabilities
        Сумма исходного и добавочного штрафов с координатным ковектором.

    Notes
    -----
    Для :math:`N` коэффициентов добавочный штраф зависит от их среднего
    и разности последнего и первого значений:

    .. math::

        \Phi(a) = R(a) + \frac{0.1}{4}
        \left[\left(\frac{1}{N}\sum_{i=0}^{N-1}a_i\right)^4
        +(a_{N-1}-a_0)^4\right].

    Здесь :math:`R` — ``base_penalty``; в интеграционном тесте :math:`N=73`.
    """
    summaries = ArraySpace((2,), axes=("summary",),
        coordinates=(("coefficient_mean", "endpoint_difference"),), units="dimensionless")
    weights = np.zeros((2, parameter_space.size))
    weights[0] = 1/parameter_space.size
    weights[1, 0], weights[1, -1] = -1., 1.
    summary_map = matrix_map(weights, domain=parameter_space, codomain=summaries)
    energy = scalar_from_functions(domain=summaries,
        value=lambda x: .25*np.sum(x**4), cotangent=lambda x: x**3)
    return sum_functionals(base_penalty,
        scale_functional(pullback_functional(energy, summary_map), .1))
