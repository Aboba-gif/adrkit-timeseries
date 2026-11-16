"""Целевой риск, профильная информация и конечный выбор: независимые матричные случаи."""

from itertools import combinations

import numpy as np
import pytest

from adrkit.inverse.design import (
    CovarianceFactor,
    DesignCandidate,
    affine_goal_risk,
    profile_information,
    robust_ball_maximum,
    select_designs,
)


def assert_ball_result(result, a, matrix, radius, expected, *, atol=2e-10):
    """Проверить допустимость и значение возвращённого свидетеля."""
    assert result.witness.shape == (matrix.shape[1],)
    assert np.linalg.norm(result.witness) <= radius + atol
    residual = a + matrix @ result.witness
    witnessed = float(residual @ residual)
    np.testing.assert_allclose(result.attained_estimate, witnessed, rtol=2e-12, atol=atol)
    np.testing.assert_allclose(result.attained_estimate, expected, rtol=2e-11, atol=atol)
    assert result.dual_upper_estimate + atol >= result.attained_estimate
    assert result.dual_upper_estimate + atol >= expected
    assert result.dual_upper_estimate - expected <= atol
    np.testing.assert_allclose(
        result.gap_estimate,
        result.dual_upper_estimate - result.attained_estimate,
        rtol=1e-6,
        atol=atol,
    )
    assert result.converged_estimate
    assert result.certified is False
    assert isinstance(result.status, str) and result.status
    assert result.iterations >= 0


def test_scalar_robust_maximum_is_nine_and_preserves_inputs():
    a, matrix = np.array([1.0]), np.array([[1.0]])
    original_a, original_matrix = a.copy(), matrix.copy()
    result = robust_ball_maximum(a, matrix, 2.0)
    # max_{|t|<=2} (1+t)^2 = 9; смещение входит внутрь максимума.
    assert_ball_result(result, a, matrix, 2.0, 9.0)
    np.testing.assert_allclose(result.witness, [2.0], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(matrix, original_matrix)
    witness = result.witness.copy()
    a[:] = 100
    matrix[:] = 100
    np.testing.assert_array_equal(result.witness, witness)


def test_hard_case_has_value_sixteen_thirds_and_nonzero_top_component():
    a, matrix = np.array([0.0, 1.0]), np.diag([2.0, 1.0])
    result = robust_ball_maximum(a, matrix, 1.0)
    # На сфере: 4-3*t_1^2+2*t_1+1; максимум при t_1=1/3.
    assert_ball_result(result, a, matrix, 1.0, 16.0 / 3.0)
    np.testing.assert_allclose(result.witness[1], 1.0 / 3.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(abs(result.witness[0]), np.sqrt(8.0 / 9.0), atol=1e-10)


def test_zero_top_linear_component_does_not_always_mean_hard_case():
    a, matrix = np.array([0.0, 6.0]), np.diag([2.0, 1.0])
    result = robust_ball_maximum(a, matrix, 1.0)
    # Формальный нижний компонент при lambda=4 равен 2 и вне шара.
    # Регулярный максимум достигается в (0,1): (6+1)^2=49.
    assert_ball_result(result, a, matrix, 1.0, 49.0)
    np.testing.assert_allclose(result.witness, [0.0, 1.0], rtol=0, atol=1e-10)


def test_pure_quadratic_maximum_uses_largest_singular_value():
    a, matrix = np.zeros(2), np.diag([7.0, 2.0])
    result = robust_ball_maximum(a, matrix, 1.0)
    assert_ball_result(result, a, matrix, 1.0, 49.0)
    np.testing.assert_allclose(abs(result.witness[0]), 1.0, atol=1e-10)


def test_near_hard_case_keeps_a_nonzero_top_linear_component():
    epsilon = 1e-7
    a, matrix = np.array([epsilon, 1.0]), np.diag([2.0, 1.0])
    result = robust_ball_maximum(a, matrix, 1.0, rtol=1e-12, maxiter=256)
    feasible = np.array([np.sqrt(8.0 / 9.0), 1.0 / 3.0])
    feasible_residual = a + matrix @ feasible
    feasible_value = float(feasible_residual @ feasible_residual)
    assert np.linalg.norm(result.witness) <= 1.0 + 1e-10
    assert result.witness[0] > 0.5
    assert result.attained_estimate >= feasible_value - 2e-10
    assert result.attained_estimate > 16.0 / 3.0 + 3.0 * epsilon
    assert result.dual_upper_estimate >= feasible_value - 2e-10
    assert result.gap_estimate <= 2e-10
    assert result.converged_estimate
    assert result.certified is False


@pytest.mark.parametrize("radius", [0.0, 2.0])
def test_empty_model_class_keeps_central_bias(radius):
    a, matrix = np.array([3.0, 4.0]), np.empty((2, 0))
    result = robust_ball_maximum(a, matrix, radius)
    assert_ball_result(result, a, matrix, radius, 25.0)


def test_zero_radius_keeps_bias_with_nonzero_model_directions():
    a, matrix = np.array([3.0, 4.0]), np.array([[7.0, 1.0], [2.0, -3.0]])
    result = robust_ball_maximum(a, matrix, 0.0)
    assert_ball_result(result, a, matrix, 0.0, 25.0)
    np.testing.assert_array_equal(result.witness, [0.0, 0.0])


def affine_case():
    """Полностью заданный случай с разными пространствами и фактором ранга один."""
    return {
        "E": np.array([[1.0, 0.3], [-0.2, 0.8]]),
        "offset": np.array([0.2, -0.3]),
        "O": np.array([[0.5, 1.0], [-1.0, 0.4]]),
        "T": np.array([[0.8, 0.0], [0.1, 0.6]]),
        "Cxi": CovarianceFactor(np.array([[1.0, 0.0], [0.3, 0.6]])),
        "Ce": CovarianceFactor(np.array([[0.5, 0.0], [0.1, 0.7]])),
        "L": np.array([[1.0, 0.2], [0.0, 2.0]]),
        "B": np.array([[1.0, 0.4], [0.1, -0.2]]),
        "Ceta": CovarianceFactor(np.array([[0.3], [0.7]])),
        "rho": 1.25,
    }


def assert_affine_components_equal(left, right):
    for field in (
        "parameter_variance",
        "noise_variance",
        "value_estimate",
        "upper_estimate",
    ):
        np.testing.assert_allclose(
            getattr(left, field), getattr(right, field), rtol=2e-10, atol=2e-11
        )


def test_scalar_affine_risk_is_nine_with_all_three_nonzero_contributions():
    result = affine_goal_risk(
        np.array([[2.0]]),
        np.array([1.0]),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.ones(1),
        np.ones(1),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.ones(1),
        0.5,
    )
    # (2*1-1)^2=1, Var(2*e)=4, max_{|eta|<=0.5}(1+2*eta)^2=4.
    np.testing.assert_allclose(result.parameter_variance, 1.0, atol=1e-13)
    np.testing.assert_allclose(result.noise_variance, 4.0, atol=1e-13)
    np.testing.assert_allclose(result.robust.attained_estimate, 4.0, atol=1e-12)
    np.testing.assert_allclose(result.value_estimate, 9.0, atol=1e-12)
    np.testing.assert_allclose(result.upper_estimate, 9.0, atol=1e-11)
    assert result.converged_estimate


def test_correlated_parameter_cancellation_and_noise_variance_one_half():
    result = affine_goal_risk(
        np.array([[1.0, -1.0]]),
        np.zeros(1),
        np.eye(2),
        np.zeros((1, 2)),
        np.ones((2, 2)),
        np.array([[1.0, 0.75], [0.75, 1.0]]),
        np.ones((1, 1)),
        np.zeros((2, 1)),
        CovarianceFactor(np.empty((1, 0))),
        1.0,
    )
    # Общая случайная компонента двух параметров уничтожается разностью.
    # Для ошибки измерений Var(e_0-e_1)=1+1-2*0.75=0.5.
    np.testing.assert_allclose(result.parameter_variance, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.noise_variance, 0.5, atol=1e-14)
    np.testing.assert_allclose(result.robust.attained_estimate, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.value_estimate, 0.5, atol=1e-13)
    assert result.converged_estimate


def test_zero_raw_covariance_classes_keep_only_transformed_bias():
    result = affine_goal_risk(
        np.ones((1, 1)),
        np.array([2.0]),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.zeros(1),
        np.zeros(1),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.zeros(1),
        3.0,
    )
    assert result.parameter_variance == result.noise_variance == 0.0
    np.testing.assert_allclose(result.value_estimate, 4.0, atol=1e-13)


def test_affine_offset_is_transformed_by_goal_map_inside_robust_maximum():
    result = affine_goal_risk(
        np.zeros((2, 1)),
        np.array([3.0, 4.0]),
        np.zeros((1, 1)),
        np.zeros((2, 1)),
        CovarianceFactor(np.empty((1, 0))),
        CovarianceFactor(np.empty((1, 0))),
        np.array([[2.0, 0.0]]),
        np.zeros((1, 1)),
        CovarianceFactor(np.empty((1, 0))),
        4.0,
    )
    assert result.parameter_variance == result.noise_variance == 0.0
    np.testing.assert_allclose(result.value_estimate, 36.0, atol=1e-13)
    np.testing.assert_allclose(result.upper_estimate, 36.0, atol=1e-13)


def test_rectangular_covariance_factor_owns_readonly_input():
    raw = np.array([[1.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    original = raw.copy()
    factor = CovarianceFactor(raw)
    assert factor.factor.shape == (2, 3)
    assert not np.shares_memory(factor.factor, raw)
    raw[:] = 99
    np.testing.assert_array_equal(factor.factor, original)
    with pytest.raises(ValueError):
        factor.factor[0, 0] = 99


def test_factor_rotation_and_rectangular_representations_preserve_affine_risk():
    case = affine_case()
    result = affine_goal_risk(**case)
    raw_covariances = dict(case)
    for name in ("Cxi", "Ce"):
        root = case[name].factor
        raw_covariances[name] = root @ root.T
    assert_affine_components_equal(result, affine_goal_risk(**raw_covariances))
    rotated = dict(case)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    for name in ("Cxi", "Ce"):
        root = case[name].factor
        rectangular = np.column_stack((root @ rotation, np.zeros(root.shape[0])))
        rotated[name] = CovarianceFactor(rectangular)
    rotated["Ceta"] = CovarianceFactor(-case["Ceta"].factor)
    assert_affine_components_equal(result, affine_goal_risk(**rotated))


def test_general_affine_changes_of_all_three_coordinate_spaces_preserve_risk():
    case = affine_case()
    originals = {
        name: value.copy() for name, value in case.items() if isinstance(value, np.ndarray)
    }
    result = affine_goal_risk(**case)
    output_change = np.array([[2.0, 0.5], [0.0, 1.0]])
    observation_change = np.array([[1.0, 0.2], [0.3, 1.5]])
    parameter_change = np.array([[1.2, 0.0], [0.4, 0.8]])
    inverse_output = np.linalg.inv(output_change)
    inverse_observation = np.linalg.inv(observation_change)
    inverse_parameter = np.linalg.inv(parameter_change)
    changed = affine_goal_risk(
        output_change @ case["E"] @ inverse_observation,
        output_change @ case["offset"],
        observation_change @ case["O"] @ inverse_parameter,
        output_change @ case["T"] @ inverse_parameter,
        CovarianceFactor(parameter_change @ case["Cxi"].factor),
        CovarianceFactor(observation_change @ case["Ce"].factor),
        case["L"] @ inverse_output,
        observation_change @ case["B"],
        case["Ceta"],
        case["rho"],
    )
    assert_affine_components_equal(result, changed)
    for name, original in originals.items():
        np.testing.assert_array_equal(case[name], original)
    np.testing.assert_allclose(
        result.value_estimate,
        result.parameter_variance + result.noise_variance + result.robust.attained_estimate,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.upper_estimate,
        result.parameter_variance + result.noise_variance + result.robust.dual_upper_estimate,
        atol=1e-13,
    )


@pytest.mark.parametrize("history_row,expected", [([0.0, 1.0], 0.0), ([1.0, 0.0], 0.5)])
def test_history_only_separates_source_when_it_observes_imitating_field(history_row, expected):
    # На основном интервале A_y=(1,0)^T имитируется первым компонентом поля.
    # История с A_h=0 должна наблюдать именно этот компонент.
    A = np.array([[0.0], [1.0], [0.0]])
    nuisance = np.vstack((history_row, np.eye(2)))
    original_A, original_nuisance = A.copy(), nuisance.copy()
    result = profile_information(A, nuisance, np.eye(3))
    np.testing.assert_allclose(result.information, [[expected]], atol=2e-13)
    np.testing.assert_allclose(result.projected.T @ result.projected, [[expected]], atol=2e-13)
    assert result.nuisance_rank == 2
    np.testing.assert_array_equal(A, original_A)
    np.testing.assert_array_equal(nuisance, original_nuisance)


def test_repeated_independent_history_increases_profile_information():
    first = profile_information(np.array([[0.0], [1.0]]), np.ones((2, 1)), np.eye(2))
    second = profile_information(np.array([[0.0], [0.0], [1.0]]), np.ones((3, 1)), np.eye(3))
    np.testing.assert_allclose(first.information, [[0.5]], atol=1e-13)
    np.testing.assert_allclose(second.information, [[2.0 / 3.0]], atol=1e-13)
    assert second.information[0, 0] > first.information[0, 0]


def test_profile_handles_exact_nuisance_rank_deficiency_without_ridge():
    A = np.array([[1.0], [0.0]])
    nuisance = np.array([[1.0, 2.0], [1.0, 2.0]])
    result = profile_information(A, nuisance, np.eye(2))
    assert result.nuisance_rank == 1
    np.testing.assert_allclose(result.projected, [[0.5], [-0.5]], atol=2e-13)
    np.testing.assert_allclose(result.information, [[0.5]], atol=2e-13)
    np.testing.assert_allclose(nuisance.T @ result.projected, np.zeros((2, 1)), atol=2e-13)


def test_profile_numerical_rank_uses_declared_relative_threshold():
    A, nuisance = np.array([[0.0], [1.0]]), np.diag([1.0, 1e-12])
    coarse = profile_information(A, nuisance, np.eye(2), relative_threshold=1e-10)
    fine = profile_information(A, nuisance, np.eye(2), relative_threshold=1e-14)
    assert coarse.nuisance_rank == 1 and fine.nuisance_rank == 2
    np.testing.assert_allclose(coarse.information, [[1.0]], atol=1e-13)
    np.testing.assert_allclose(fine.information, [[0.0]], atol=1e-13)
    np.testing.assert_allclose(coarse.nuisance_threshold, 1e-10, rtol=1e-13)
    np.testing.assert_allclose(fine.nuisance_threshold, 1e-14, rtol=1e-13)


def test_profile_correlated_metric_matches_independent_scalar_schur_complement():
    # Sigma^{-1}=[[4,-1],[-1,2]]/7; A=(1,0), C=(0,1).
    # S=4/7-((-1/7)^2)/(2/7)=1/2.
    result = profile_information(
        np.array([[1.0], [0.0]]),
        np.array([[0.0], [1.0]]),
        np.array([[2.0, 1.0], [1.0, 4.0]]),
    )
    np.testing.assert_allclose(result.information, [[0.5]], atol=2e-13)


def test_design_selection_supports_three_goals_nonunit_baselines_and_permutations():
    baselines = {"arrival": 5.0, "stock": 10.0, "flux": 20.0}
    ratios = {
        "alpha": (0.2, 0.8, 0.8),
        "beta": (0.8, 0.2, 0.8),
        "gamma": (0.8, 0.8, 0.2),
        "delta": (0.6, 0.6, 0.6),
        "dominated": (0.8, 0.8, 0.8),
    }
    candidates = [
        DesignCandidate(
            name,
            2.0,
            {goal: ratio * baselines[goal] for goal, ratio in zip(baselines, values)},
        )
        for name, values in ratios.items()
    ]
    selection = select_designs(candidates, baselines, budgets=(0.0, 2.0), mode="exact")
    assert selection[0].winner is None and selection[0].ranking == ()
    chosen = selection[1]
    assert chosen.budget == 2.0 and chosen.mode == "exact"
    assert chosen.winner == "delta"
    assert [rank.identifier for rank in chosen.ranking] == [
        "delta",
        "alpha",
        "beta",
        "gamma",
        "dominated",
    ]
    assert chosen.goal_winners == {"arrival": "alpha", "stock": "beta", "flux": "gamma"}
    assert set(chosen.pareto) == {"alpha", "beta", "gamma", "delta"}
    np.testing.assert_allclose(chosen.ranking[0].maximum, 0.6, atol=1e-14)
    np.testing.assert_allclose(chosen.ranking[0].total, 1.8, atol=1e-14)
    for rank in chosen.ranking:
        np.testing.assert_allclose(
            [rank.normalized[goal] for goal in baselines],
            ratios[rank.identifier],
            atol=1e-14,
        )
    reversed_result = select_designs(
        list(reversed(candidates)),
        dict(reversed(list(baselines.items()))),
        budgets=(2.0,),
        mode="exact",
    )[0]
    assert reversed_result.winner == chosen.winner
    assert [rank.identifier for rank in reversed_result.ranking] == [
        rank.identifier for rank in chosen.ranking
    ]
    assert reversed_result.goal_winners == chosen.goal_winners
    assert set(reversed_result.pareto) == set(chosen.pareto)


def test_budget_modes_and_owned_risk_mappings():
    risks = {"load": 5.0, "delay": 10.0, "energy": 15.0}
    candidate = DesignCandidate("small", 1.0, risks)
    risks["load"] = 500.0
    assert candidate.risks["load"] == 5.0
    baselines = {"load": 5.0, "delay": 5.0, "energy": 5.0}
    other = DesignCandidate("large", 2.0, {"load": 1.0, "delay": 1.0, "energy": 1.0})
    exact = select_designs([candidate, other], baselines, budgets=(1.5,), mode="exact")[0]
    at_most = select_designs([candidate, other], baselines, budgets=(1.5,), mode="at_most")[0]
    assert exact.winner is None and exact.ranking == ()
    assert at_most.winner == "small"
    np.testing.assert_allclose(at_most.ranking[0].maximum, 3.0, atol=1e-14)
    baselines["load"] = 5000.0
    assert at_most.ranking[0].normalized["load"] == 1.0


def test_selection_rejects_normalization_that_erases_strict_risk_order():
    candidates = [
        DesignCandidate("a", 1.0, {"state": 2e-200}),
        DesignCandidate("z", 1.0, {"state": 1e-200}),
    ]
    with pytest.raises(ValueError, match="positive normalized risk underflows"):
        select_designs(candidates, {"state": 1e200}, budgets=(1.0,))
    selection = select_designs(candidates, {"state": 1.0}, budgets=(1.0,))[0]
    assert selection.winner == "z"
    assert selection.goal_winners == {"state": "z"}
    assert selection.pareto == ("z",)
    assert [row.identifier for row in selection.ranking] == ["z", "a"]


def test_three_mask_table_has_exact_risks_and_a_greedy_counterexample():
    rows = {"A": [2.0, 0.0], "B": [1.0, np.sqrt(2.0)], "C": [1.0, -np.sqrt(2.0)]}
    expected = {
        "A": 6.0 / 5.0,
        "B": 5.0 / 4.0,
        "C": 5.0 / 4.0,
        "AB": 9.0 / 16.0,
        "AC": 9.0 / 16.0,
        "BC": 8.0 / 15.0,
        "ABC": 12.0 / 35.0,
    }
    candidates = []
    for size in (1, 2, 3):
        for names in combinations(rows, size):
            name = "".join(names)
            observation = np.array([rows[key] for key in names])
            estimator = np.linalg.solve(np.eye(2) + observation.T @ observation, observation.T)
            risk = affine_goal_risk(
                estimator,
                np.zeros(2),
                observation,
                np.eye(2),
                np.eye(2),
                np.eye(size),
                np.eye(2),
                np.zeros((size, 1)),
                CovarianceFactor(np.empty((1, 0))),
                0.0,
            )
            # Для этой линейной задачи риск равен tr((I+O^T O)^{-1});
            # семь значений выше получены независимыми дробными вычислениями.
            np.testing.assert_allclose(risk.value_estimate, expected[name], atol=2e-13)
            candidates.append(DesignCandidate(name, size, {"state": risk.value_estimate}))
    first, second = select_designs(candidates, {"state": 2.0}, budgets=(1, 2), mode="exact")
    assert first.winner == "A" and second.winner == "BC"
    greedy_extensions = [
        candidate
        for candidate in candidates
        if len(candidate.identifier) == 2 and first.winner in candidate.identifier
    ]
    greedy = select_designs(greedy_extensions, {"state": 2.0}, budgets=(2,), mode="exact")[0]
    assert greedy.winner in {"AB", "AC"}
    assert greedy.ranking[0].maximum > second.ranking[0].maximum


@pytest.mark.parametrize("bad", [True, -1.0, np.nan, np.inf, "1"])
def test_robust_maximum_rejects_invalid_radius(bad):
    with pytest.raises((TypeError, ValueError)):
        robust_ball_maximum(np.ones(1), np.ones((1, 1)), bad)


@pytest.mark.parametrize("options", [{"maxiter": 0}, {"maxiter": True}, {"rtol": -1.0}])
def test_robust_maximum_rejects_invalid_numerical_controls(options):
    with pytest.raises((TypeError, ValueError)):
        robust_ball_maximum(np.ones(1), np.ones((1, 1)), 1.0, **options)


@pytest.mark.parametrize(
    "a,matrix",
    [
        (np.ones((1, 1)), np.ones((1, 1))),
        (np.ones(2), np.ones((1, 1))),
        (np.array([np.nan]), np.ones((1, 1))),
        (np.ones(1), np.array([[np.inf]])),
        (np.array([1j]), np.ones((1, 1))),
        (np.ones(1), np.array([[True]])),
    ],
)
def test_robust_maximum_rejects_invalid_array_contracts(a, matrix):
    with pytest.raises((TypeError, ValueError)):
        robust_ball_maximum(a, matrix, 1.0)


@pytest.mark.parametrize("bad", [np.ones(2), np.array([[np.nan]]), np.array([[1j]]), [[True]]])
def test_covariance_factor_rejects_nonmatrix_or_nonreal_values(bad):
    with pytest.raises((TypeError, ValueError)):
        CovarianceFactor(bad)


@pytest.mark.parametrize(
    "name,bad",
    [
        ("Cxi", np.diag([1.0, -1e-4])),
        ("Ce", np.array([[1.0, 0.3], [0.0, 1.0]])),
        ("Ceta", np.diag([1.0, -1e-4])),
        ("offset", np.zeros((2, 1))),
        ("L", np.ones((1, 3))),
        ("E", np.array([[1.0, np.inf], [0.0, 1.0]])),
        ("O", np.ones((3, 2))),
        ("rho", np.nan),
    ],
)
def test_affine_risk_rejects_invalid_covariances_shapes_and_values(name, bad):
    case = affine_case()
    case[name] = bad
    with pytest.raises((TypeError, ValueError)):
        affine_goal_risk(**case)


def test_affine_risk_rejects_nonfinite_intermediate_arithmetic():
    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        affine_goal_risk(
            np.array([[1e308]]),
            np.zeros(1),
            np.array([[1e308]]),
            np.ones((1, 1)),
            np.ones(1),
            np.ones(1),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.ones(1),
            1.0,
        )


@pytest.mark.parametrize("covariance", [np.diag([1.0, 0.0]), np.diag([1.0, -1.0])])
def test_profile_requires_positive_definite_working_covariance(covariance):
    with pytest.raises((TypeError, ValueError)):
        profile_information(np.ones((2, 1)), np.ones((2, 1)), covariance)


@pytest.mark.parametrize("baseline", [0.0, -1.0, np.nan, np.inf])
def test_selection_rejects_invalid_baselines(baseline):
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"g": 1.0})], {"g": baseline}, budgets=(1,))


def test_selection_rejects_duplicate_ids_goal_mismatch_and_nonfinite_inputs():
    with pytest.raises((TypeError, ValueError)):
        select_designs(
            [
                DesignCandidate("same", 1, {"g": 1.0}),
                DesignCandidate("same", 2, {"g": 2.0}),
            ],
            {"g": 5.0},
            budgets=(2,),
        )
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"other": 1.0})], {"g": 5.0}, budgets=(1,))
    with pytest.raises((TypeError, ValueError)):
        select_designs(
            [
                DesignCandidate("one", 1, {"g": 1.0}),
                DesignCandidate("two", 2, {"other": 1.0}),
            ],
            {"g": 5.0},
            budgets=(2,),
        )
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", np.inf, {"g": 1.0})], {"g": 5.0}, budgets=(1,))
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"g": np.nan})], {"g": 5.0}, budgets=(1,))
    with pytest.raises((TypeError, ValueError)):
        select_designs([DesignCandidate("one", 1, {"g": 1.0})], {"g": 5.0}, budgets=(np.inf,))


def test_no_data_affine_baseline_is_five():
    result = affine_goal_risk(
        np.empty((1, 0)),
        np.array([1.0]),
        np.empty((0, 1)),
        np.array([[2.0]]),
        np.ones(1),
        np.empty((0, 0)),
        np.ones((1, 1)),
        np.empty((0, 0)),
        np.empty((0, 0)),
        1.0,
    )
    # Без наблюдений остаются Var(-2*xi)=4 и квадрат смещения 1.
    np.testing.assert_allclose(result.parameter_variance, 4.0, atol=1e-13)
    assert result.noise_variance == 0.0
    np.testing.assert_allclose(result.robust.attained_estimate, 1.0, atol=1e-13)
    np.testing.assert_allclose(result.value_estimate, 5.0, atol=1e-13)
    np.testing.assert_allclose(result.upper_estimate, 5.0, atol=1e-13)
    assert result.converged_estimate


def test_model_error_family_has_unbounded_greedy_ratio_as_penalty_vanishes():
    penalty = 1e-3
    full_observation = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    full_model_direction = np.array([[0.0], [1.0], [0.0]])
    candidates = []
    for size in (1, 2, 3):
        for indices in combinations(range(3), size):
            rows = list(indices)
            observation = full_observation[rows]
            estimator = np.linalg.solve(
                observation.T @ observation + penalty * np.eye(2), observation.T
            )
            risks = {}
            for coordinate, goal in enumerate(("first", "second")):
                risk = affine_goal_risk(
                    estimator,
                    np.zeros(2),
                    observation,
                    np.eye(2),
                    np.eye(2),
                    penalty**2 * np.eye(size),
                    np.eye(2)[coordinate : coordinate + 1],
                    full_model_direction[rows],
                    np.ones(1),
                    1.0,
                )
                assert risk.converged_estimate
                risks[goal] = risk.value_estimate
            identifier = "".join(str(index + 1) for index in indices)
            candidates.append(DesignCandidate(identifier, size, risks))

    assert {candidate.identifier for candidate in candidates} == {
        "1",
        "2",
        "3",
        "12",
        "13",
        "23",
        "123",
    }
    baseline = {"first": 1.0, "second": 1.0}
    singleton, pair = select_designs(candidates, baseline, budgets=(1, 2), mode="exact")
    assert singleton.winner == "2"
    assert pair.winner == "13"
    expected_optimal = 2.0 * penalty**2 / (1.0 + penalty) ** 2
    np.testing.assert_allclose(pair.ranking[0].maximum, expected_optimal, rtol=1e-10, atol=1e-15)
    optimum = next(candidate for candidate in candidates if candidate.identifier == "13")
    np.testing.assert_allclose(
        list(optimum.risks.values()),
        [expected_optimal, expected_optimal],
        rtol=1e-10,
        atol=1e-15,
    )
    # Начав с лучшего одиночного варианта 2, добавление без удаления
    # может дать только 12 или 23. Их модельное смещение не исчезает при lambda->0.
    extensions = [
        candidate
        for candidate in candidates
        if len(candidate.identifier) == 2 and "2" in candidate.identifier
    ]
    assert {candidate.identifier for candidate in extensions} == {"12", "23"}
    greedy = select_designs(extensions, baseline, budgets=(2,), mode="exact")[0]
    assert greedy.winner in {"12", "23"}
    assert greedy.ranking[0].maximum / pair.ranking[0].maximum > 1000.0


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_robust_maximum_rejects_underflow_or_overflow_of_nonzero_objective(magnitude):
    with pytest.raises(ValueError):
        robust_ball_maximum(np.array([magnitude]), np.zeros((1, 0)), 0.0)


def test_smallest_subnormal_noise_covariance_preserves_scaled_variance():
    tiny = np.nextafter(0.0, 1.0)
    covariances = (
        np.array([[tiny]]),
        np.array([tiny]),
        CovarianceFactor(np.array([[np.sqrt(tiny)]])),
    )
    results = [
        affine_goal_risk(
            np.array([[1e162]]),
            np.zeros(1),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros(1),
            covariance,
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.zeros(1),
            1.0,
        )
        for covariance in covariances
    ]
    for result in results:
        assert result.noise_variance > 1.0
        assert result.parameter_variance == 0.0
        assert result.robust.attained_estimate == 0.0
        np.testing.assert_allclose(result.value_estimate, result.noise_variance, rtol=1e-14)
        np.testing.assert_allclose(
            result.noise_variance, results[0].noise_variance, rtol=2e-14, atol=0.0
        )


def test_model_factor_chain_preserves_representable_risk_after_radius_scaling():
    result = affine_goal_risk(
        np.array([[1e-200]]),
        np.zeros(1),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros(1),
        np.zeros(1),
        np.ones((1, 1)),
        np.array([[1e-200]]),
        np.array([1e300]),
        1e250,
    )
    # B*sqrt(Ceta)=1e-50, E*(B*sqrt(Ceta))=1e-250;
    # умножение на радиус даёт 1. Раннее формирование E*B потеряло бы этот вклад.
    assert result.parameter_variance == result.noise_variance == 0.0
    np.testing.assert_allclose(result.value_estimate, 1.0, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(result.upper_estimate, 1.0, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(abs(result.robust.witness[0] / 1e250), 1.0, rtol=2e-12, atol=1e-13)
    assert result.converged_estimate


def test_hard_case_does_not_claim_unattainable_relative_tolerance():
    result = robust_ball_maximum(np.array([0.0, 1.0]), np.diag([2.0, 1.0]), 1.0, rtol=1e-30)
    assert not result.converged_estimate
    assert result.status in {"resolution_limit_estimate", "numerical_inconsistency"}
    assert result.certified is False
    np.testing.assert_allclose(result.attained_estimate, 16.0 / 3.0, atol=2e-12)
