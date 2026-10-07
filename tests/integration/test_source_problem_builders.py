"""Согласованность сборки q(t) и производной составного штрафа."""
import numpy as np
from tests.integration.fixtures.source_problem import build_problem, regularize_source_summaries

def test_source_problem_and_composed_penalty():
    problem, penalty = build_problem(np.zeros(36))
    point = np.linspace(.01, .02, problem.parameter_space.size)
    custom = regularize_source_summaries(problem.parameter_space, penalty)
    direction, step = np.cos(np.arange(point.size)), 1e-5
    finite_difference = (custom.value.value(point+step*direction)
                         - custom.value.value(point-step*direction))/(2*step)
    np.testing.assert_allclose(custom.cotangent.cotangent(point) @ direction,
                               finite_difference, rtol=2e-8, atol=1e-8)
    assert problem.predict(point).shape == (36,)
    assert np.isfinite(problem.value(point))
    assert np.isfinite(problem.cotangent(point)).all()
