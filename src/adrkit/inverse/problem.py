"""Обратная задача из прогноза, невязки, штрафа и границ."""

from adrkit.contracts import (
    require_capability,
    validate_map_capabilities,
    validate_scalar_capabilities,
)
from adrkit.spaces import ArraySpace

from .constraints import box_bounds
from .objectives import compose_maps, evaluate_score, make_objective
from .optimizers import solve_lbfgsb


def _check_selection(selection):
    if selection is None:
        return
    for name in ("full_layout", "selected_layout"):
        if not isinstance(getattr(selection, name, None), ArraySpace):
            raise TypeError(f"selection {name} must be an ArraySpace record")
    for method in ("take", "scatter"):
        if not callable(getattr(selection, method, None)):
            raise TypeError(f"selection must provide callable {method}()")


class InverseProblem:
    """Обратная задача из прогноза, функционала потерь и штрафа.

    Parameters
    ----------
    forward : MapCapabilities
        Прогноз между пространствами параметров и наблюдений.
    loss : ScalarCapabilities
        Функционал потерь на пространстве наблюдений после selection.
    penalty : ScalarCapabilities or None, optional
        Штраф на пространстве параметров с уже включённым весом.
    selection : Selection or None, optional
        Выбор строк наблюдений перед вычислением потерь.
    bounds : sequence of pairs or None, optional
        Границы параметров в порядке C.

    Notes
    -----
    Производные — координатные ковекторы. Неотрицательность
    и регуляризация задаются переданными границами и штрафом.
    """

    def __init__(self, forward, loss, *, penalty=None, selection=None, bounds=None):
        validate_map_capabilities(forward, name="forward")
        validate_scalar_capabilities(loss, name="loss")
        if penalty is not None:
            validate_scalar_capabilities(penalty, name="penalty")
        _check_selection(selection)
        objective = make_objective(forward, selection, loss, penalty)
        checked_bounds = tuple(box_bounds(bounds, forward.domain.size))
        self._forward = forward
        self._loss = loss
        self._penalty = penalty
        self._objective = objective
        self._bounds = checked_bounds

    @classmethod
    def from_maps(cls, *stages, loss, penalty=None, selection=None, bounds=None):
        """Создать обратную задачу из последовательности отображений.

        Parameters
        ----------
        *stages
            Этапы MapCapabilities в порядке применения.
        loss : ScalarCapabilities
            Потери на пространстве конечного наблюдения.
        penalty : ScalarCapabilities or None, optional
            Штраф первого пространства параметров.
        selection : Selection or None, optional
            Выбор строк конечного наблюдения.
        bounds : sequence of pairs or None, optional
            Покомпонентные границы параметров.

        Returns
        -------
        problem : InverseProblem
            Задача с составным прямым отображением.
        """
        for stage in stages:
            validate_map_capabilities(stage, name="forward")
        return cls(
            compose_maps(*stages), loss, penalty=penalty, selection=selection, bounds=bounds
        )

    @property
    def parameter_space(self):
        """Пространство параметров обратной задачи."""
        return self._forward.domain

    @property
    def forward(self):
        """Прямое отображение с сохранёнными по ссылке операциями."""
        return self._forward

    @property
    def objective(self):
        """ScalarCapabilities функционала для совместимого оптимизатора."""
        return self._objective

    @property
    def bounds(self):
        """Кортеж пар границ параметров в порядке C."""
        return self._bounds

    def predict(self, parameters):
        """Вычислить полный прогноз по параметрам.

        Parameters
        ----------
        parameters : array_like
            Конечная точка формы parameter_space.shape в единицах
            пространства.

        Returns
        -------
        result : ndarray
            Прогноз формы forward.codomain.shape до выбора строк.
        """
        x = self.parameter_space.array(parameters, name="parameters", copy=True)
        return self.forward.codomain.array(
            self.forward.predict.predict(x), name="prediction", copy=True
        )

    def value(self, parameters):
        """Вычислить функционал со штрафом.

        Parameters
        ----------
        parameters : array_like
            Конечная точка формы parameter_space.shape в единицах
            пространства.

        Returns
        -------
        result : float
            Значение потерь и штрафа; бесконечности допустимы, NaN
            отклоняется.
        """
        return evaluate_score(self.objective, parameters)

    def _require_cotangent(self):
        require_capability(self._forward.vjp, operation="forward VJP", consumer="InverseProblem")
        require_capability(
            self._loss.cotangent, operation="loss cotangent", consumer="InverseProblem"
        )
        if self._penalty is not None:
            require_capability(
                self._penalty.cotangent, operation="penalty cotangent", consumer="InverseProblem"
            )
        return require_capability(
            self.objective.cotangent, operation="smooth cotangent", consumer="InverseProblem"
        )

    def cotangent(self, parameters):
        """Вычислить координатную производную функционала.

        Parameters
        ----------
        parameters : array_like
            Конечная точка формы parameter_space.shape в единицах
            пространства.

        Returns
        -------
        result : ndarray
            Ковектор формы parameter_space.shape.
        """
        derivative = self._require_cotangent()
        x = self.parameter_space.array(parameters, name="parameters", copy=True)
        return self.parameter_space.array(derivative.cotangent(x), name="cotangent", copy=True)

    def solve(self, initial, *, options=None):
        """Подогнать параметры методом L-BFGS-B.

        Parameters
        ----------
        initial : array_like
            Конечная начальная точка формы parameter_space.shape.
        options : Mapping or None, optional
            Параметры L-BFGS-B для scipy.optimize.minimize.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Решение x и координатный градиент jac как плоские массивы в
            порядке C.
        """
        self._require_cotangent()
        return solve_lbfgsb(self.objective, initial, bounds=self.bounds, options=options)
