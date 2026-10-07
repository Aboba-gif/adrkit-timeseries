"""Методы заполнения, их параметры и состояния обучения."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .numerics import HGBFiller, KalmanDLM, SoftImpute, fill_linear, fill_pchip

@dataclass(frozen=True)
class NoFitState:
    """Сохранить настройки интерполяции без обучения.

    Parameters
    ----------
    kind : {'linear', 'pchip'}
        Метод интерполяции.
    edge_policy : str
        Правило обработки края соответствующего метода.
    """
    kind: str
    edge_policy: str


@dataclass(frozen=True)
class InterpolationImputer:
    """Заполнять каждый ряд линейной или PCHIP-интерполяцией.

    Parameters
    ----------
    kind : {'linear', 'pchip'}
        Метод для каждого поста и канала.
    edge_policy : str
        Для ``linear``: ``'nearest'`` продолжает крайнее наблюдение, ``'abstain'``
        оставляет края ``NaN``. Для ``pchip``: ``'extrapolate'`` продолжает крайние
        кубические участки, ``'abstain'`` оставляет края ``NaN``.

    Attributes
    ----------
    no_fit_state : NoFitState
        Состояние, которое следует явно передать в ``predict`` как ``fitted``.
    """
    kind: str
    edge_policy: str

    def __post_init__(self):
        from .contracts import choice
        choice(self.kind, {"linear", "pchip"}, "imputer.kind")
        _interpolation_parameters(self.kind, {"edge_policy": self.edge_policy})

    @property
    def no_fit_state(self):
        """Вернуть настройки интерполяции для передачи в ``predict``."""
        return NoFitState(self.kind, self.edge_policy)

    def predict(self, inputs, *, fitted):
        """Заполнить пропуски панели с выбранным правилом обработки краёв.

        Parameters
        ----------
        inputs : MaskedInputs
            Видимые входы формы (n_times, n_sites, n_channels) в исходных
            единицах.
        fitted : NoFitState
            Состояние, равное ``no_fit_state`` этого метода.

        Returns
        -------
        result : ImputationResult
            Панель с точным сохранением видимых наблюдений; ``unavailable`` для
            недоступного заполнения, ``uncertainty=None``.

        Notes
        -----
        Ряд без наблюдений остаётся недоступным. PCHIP требует хотя бы
        два наблюдения; при одном сохраняется только оно. Интерполяция
        использует позиции на регулярной временной сетке.
        """
        from .contracts import DataContractError, MaskedInputs, result_from_inputs
        if not isinstance(inputs, MaskedInputs):
            raise DataContractError("predict requires MaskedInputs, never panel/truth")
        if not isinstance(fitted, NoFitState) or fitted != self.no_fit_state:
            raise DataContractError("explicit matching no-fit state required")
        prediction = np.full(inputs.values.shape, np.nan)
        for site in range(inputs.values.shape[1]):
            for channel in range(inputs.values.shape[2]):
                series = inputs.values[:, site, channel]
                observed = np.flatnonzero(np.isfinite(series))
                if len(observed) == 0:
                    continue
                if self.kind == "linear":
                    filled, _ = fill_linear(series)
                elif len(observed) >= 2:
                    filled, _ = fill_pchip(series)
                else:
                    filled = series.copy()  # Для кубического интерполянта нужны хотя бы два узла.
                # Копия нужна для записи: pandas может вернуть неизменяемое представление.
                
                filled = np.array(filled, dtype=float, copy=True)
                if self.edge_policy == "abstain":
                    filled[:observed[0]] = np.nan
                    filled[observed[-1]+1:] = np.nan
                prediction[:, site, channel] = filled
        # Численный метод не должен менять видимые наблюдения.
        prediction[inputs.visible] = inputs.values[inputs.visible]
        available = np.isfinite(prediction)
        status = np.where(inputs.visible, "observed", np.where(available, "imputed", "unavailable"))
        return result_from_inputs(inputs, prediction, availability=available, status=status)


@dataclass(frozen=True)
class ImputerFamily:
    """Задать проверку параметров и конструктор метода заполнения.

    Parameters
    ----------
    validate_parameters : callable
        Принимает независимый словарь параметров и возвращает
        нормализованный словарь, представимый конечным JSON.
    constructor : callable
        Принимает нормализованный словарь и возвращает объект с
        ``predict(inputs, fitted=...)``.

    Notes
    -----
    Обучаемый метод также предоставляет ``fit(train_inputs, rng=...)``.
    Метод без обучения предоставляет ``no_fit_state``. Внешние семейства
    передаются явно через ``families``; они не заменяют встроенные имена.
    """
    validate_parameters: object
    constructor: object

    def __post_init__(self):
        from .contracts import DataContractError
        if not callable(self.validate_parameters) or not callable(self.constructor):
            raise DataContractError("family requires explicit Python validator and constructor")


def _interpolation_parameters(kind, parameters):
    """Проверить правило края для выбранного вида интерполяции."""
    from .contracts import choice, closed
    closed(parameters, {"edge_policy"}, where=kind + ".parameters")
    allowed = {"abstain", "nearest"} if kind == "linear" else {"abstain", "extrapolate"}
    return {"edge_policy": choice(parameters["edge_policy"], allowed, "edge_policy")}


def _families(additional):
    """Соединить встроенные и явно переданные семейства методов."""
    from .contracts import DataContractError, text
    result = {
        "linear": ImputerFamily(lambda p: _interpolation_parameters("linear", p),
                                lambda p: InterpolationImputer("linear", p["edge_policy"])),
        "pchip": ImputerFamily(lambda p: _interpolation_parameters("pchip", p),
                               lambda p: InterpolationImputer("pchip", p["edge_policy"])),
        "kalman_dlm": ImputerFamily(lambda p: KalmanParameters.from_dict(p).to_dict(),
                                    lambda p: KalmanImputer(KalmanParameters.from_dict(p))),
        "soft_impute": ImputerFamily(lambda p: SoftImputeParameters.from_dict(p).to_dict(),
                                     lambda p: SoftImputeAdapter(SoftImputeParameters.from_dict(p))),
        "hgb": ImputerFamily(lambda p: HGBParameters.from_dict(p).to_dict(),
                             lambda p: HGBImputer(HGBParameters.from_dict(p))),
    }
    if additional is not None:
        if type(additional) is not dict:
            raise DataContractError("families requires an explicit Python dict")
        for kind, family in additional.items():
            text(kind, "family.kind")
            if kind in result or not isinstance(family, ImputerFamily):
                raise DataContractError("duplicate built-in kind or invalid ImputerFamily")
            result[kind] = family
    return result


def validate_imputer_spec(spec, *, families=None):
    """Проверить и нормализовать запрос метода заполнения.

    Parameters
    ----------
    spec : dict
        Поля ``kind`` и ``parameters``: имя семейства и его словарь параметров.
    families : dict of str to ImputerFamily, optional
        Дополнительные Python-семейства с именами, отличными от
        встроенных.

    Returns
    -------
    normalized : dict
        Независимые ``kind`` и нормализованные ``parameters``, представимые
        конечным JSON.

    Notes
    -----
    Выбранное семейство проверяет смысл параметров. Конструктор метода
    и вычисления с данными на этом этапе не вызываются.
    """
    import json
    from .contracts import DataContractError, closed, text
    closed(spec, {"kind", "parameters"}, where="imputer")
    kind = text(spec["kind"], "imputer.kind")
    available = _families(families)
    if kind not in available:
        raise DataContractError(f"unsupported imputer kind: {kind}")
    if type(spec["parameters"]) is not dict:
        raise DataContractError("imputer.parameters must be an object")
    try:
        owned = json.loads(json.dumps(spec["parameters"], allow_nan=False))
        parameters = available[kind].validate_parameters(owned)
        if type(parameters) is not dict:
            raise DataContractError("family validator must return a parameters object")
        return {"kind": kind, "parameters": json.loads(json.dumps(parameters, allow_nan=False))}
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"invalid {kind} parameters: {exc}") from exc


def build_imputer(spec, *, families=None):
    """Создать метод заполнения из проверенного запроса.

    Parameters
    ----------
    spec : dict
        Поля ``kind`` и ``parameters``: имя семейства и его словарь параметров.
    families : dict of str to ImputerFamily, optional
        Дополнительные Python-семейства с именами, отличными от
        встроенных.

    Returns
    -------
    imputer : object
        Метод с операцией ``predict``; состояние для неё получает ``fit`` либо
        ``no_fit_state`` соответствующего семейства.

    Notes
    -----
    Сборка не читает источники, не обучает метод и не вычисляет прогноз.
    Встроены ``linear``, ``pchip``, ``kalman_dlm``, ``soft_impute`` и ``hgb``. Семейство
    может задавать собственный метод и смысл стандартного отклонения.
    """
    from .contracts import DataContractError
    normalized = validate_imputer_spec(spec, families=families)
    result = _families(families)[normalized["kind"]].constructor(normalized["parameters"])
    if not callable(getattr(result, "predict", None)):
        raise DataContractError("imputer family must provide predict")
    return result


def _positive_integer(value, name):
    """Вернуть положительное целое число либо отклонить значение."""
    from .contracts import DataContractError
    if type(value) is not int or value <= 0:
        raise DataContractError(f"{name}: positive integer required")
    return value


@dataclass(frozen=True)
class KalmanParameters:
    """Задать обучение и режим заполнения модели Калмана.

    Parameters
    ----------
    period : int
        Положительный период гармоник в шагах сетки.
    harmonics : int
        Положительное число гармоник.
    maxiter : int
        Положительный предел итераций ``L-BFGS-B`` при обучении каждого
        ряда.
    mode : {'interior', 'tail', 'head'}
        Сглаживание по всему ряду, прямой проход или проход по
        обращённому ряду.
    initial_variance : float, optional
        Положительная начальная дисперсия каждой компоненты состояния в
        квадрате численной единицы ряда; по умолчанию 10.
    x0 : array_like, shape (5,), optional
        Начальные параметры оптимизатора; ``None`` выбирает начальную точку
        из дисперсии обучающего ряда.

    Notes
    -----
    AR обозначает авторегрессионную компоненту первого порядка.
    Порядок ``x0``: логарифмы дисперсий уровня, гармоник и AR-компоненты,
    параметр AR перед tanh, логарифм дисперсии наблюдения. Начальный
    уровень численного фильтра использует среднее всего переданного ряда;
    название ``tail`` само по себе не задаёт прогноз в реальном времени.
    """
    period: int
    harmonics: int
    maxiter: int
    mode: str
    initial_variance: float = field(default=10.0, kw_only=True)
    x0: tuple[float, ...] | None = field(default=None, kw_only=True)

    def __post_init__(self):
        from .contracts import DataContractError, choice
        for key in ("period", "harmonics", "maxiter"):
            _positive_integer(getattr(self, key), key)
        choice(self.mode, {"interior", "tail", "head"}, "kalman_dlm.mode")
        variance = self.initial_variance
        if (isinstance(variance, (bool, np.bool_))
                or not isinstance(variance, (int, float, np.integer, np.floating))):
            raise DataContractError("kalman_dlm.initial_variance: finite positive number required")
        try:
            variance = float(variance)
        except OverflowError as exc:
            raise DataContractError("kalman_dlm.initial_variance: finite positive number required") from exc
        if not np.isfinite(variance) or variance <= 0:
            raise DataContractError("kalman_dlm.initial_variance: finite positive number required")
        object.__setattr__(self, "initial_variance", variance)
        if self.x0 is not None:
            values = np.asarray(self.x0, dtype=object)
            if (values.shape != (5,) or any(isinstance(v, (bool, np.bool_))
                    or not isinstance(v, (int, float, np.integer, np.floating)) for v in values)):
                raise DataContractError("kalman_dlm.x0: five finite real numbers required")
            try:
                point = tuple(float(v) for v in values)
            except OverflowError as exc:
                raise DataContractError("kalman_dlm.x0: five finite real numbers required") from exc
            if not np.isfinite(point).all():
                raise DataContractError("kalman_dlm.x0: five finite real numbers required")
            object.__setattr__(self, "x0", point)

    @classmethod
    def from_dict(cls, value):
        """Создать проверенные параметры из словаря полей класса."""
        from .contracts import closed
        return cls(**closed(value, {"period", "harmonics", "maxiter", "mode", "initial_variance", "x0"},
                            optional={"initial_variance", "x0"}, where="kalman_dlm.parameters"))

    def to_dict(self):
        """Вернуть независимый словарь нормализованных параметров."""
        from dataclasses import asdict
        result = asdict(self)
        result["x0"] = None if self.x0 is None else list(self.x0)
        return result


@dataclass(frozen=True)
class SoftImputeParameters:
    """Задать матрицу циклов и параметры заполнения по сингулярному разложению.

    Parameters
    ----------
    lam : float
        Неотрицательный порог уменьшения сингулярных чисел в единице
        исходного ряда.
    rank_max : int
        Положительный предел числа сохраняемых компонент.
    max_iter : int
        Положительный предел итераций.
    tol : float
        Неотрицательный множитель критерия остановки по изменению
        целевой функции.
    cycle_length : int
        Положительное число последовательных шагов ряда в одной строке
        матрицы.

    Notes
    -----
    SVD обозначает сингулярное разложение матрицы.
    Строки начинаются с первого элемента ряда; последняя дополняется
    ``NaN``. ``cycle_length`` не привязывает позиции к календарным суткам.
    """
    lam: float
    rank_max: int
    max_iter: int
    tol: float
    cycle_length: int

    def __post_init__(self):
        from .contracts import DataContractError
        for key in ("rank_max", "max_iter", "cycle_length"):
            _positive_integer(getattr(self, key), key)
        for key in ("lam", "tol"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise DataContractError(f"{key}: finite nonnegative number required")

    @classmethod
    def from_dict(cls, value):
        """Создать проверенные параметры из словаря полей класса."""
        from .contracts import closed
        return cls(**closed(value, {"lam", "rank_max", "max_iter", "tol", "cycle_length"},
                            where="soft_impute.parameters"))

    def to_dict(self):
        """Вернуть независимый словарь нормализованных параметров."""
        from dataclasses import asdict
        return asdict(self)


def input_layout(inputs):
    """Получить описание осей, часов и единиц входной панели.

    Parameters
    ----------
    inputs : MaskedInputs
        Видимые входы с явной спецификацией панели.

    Returns
    -------
    layout : str
        JSON с ``sites``, ``channels``, ``units``, ``frequency``, ``origin``, ``clock``,
        ``timezone`` и ``calendar``; ``start``/``stop`` и идентификаторы выборки не
        входят.
    """
    import json
    from .contracts import DataContractError, MaskedInputs
    if not isinstance(inputs, MaskedInputs):
        raise DataContractError("expected MaskedInputs, never panel/truth")
    layout = inputs.spec.to_dict()
    return json.dumps({key: layout[key] for key in (
        "sites", "channels", "units", "frequency", "origin", "clock", "timezone", "calendar"
    )}, sort_keys=True)


def _imputation_result(inputs, prediction, uncertainty=None):
    """Сохранить видимые значения и построить привязанный результат."""
    from .contracts import result_from_inputs
    prediction[inputs.visible] = inputs.values[inputs.visible]
    available = np.isfinite(prediction)
    status = np.where(inputs.visible, "observed", np.where(available, "imputed", "unavailable"))
    return result_from_inputs(inputs, prediction, availability=available, status=status, uncertainty=uncertainty)


@dataclass(frozen=True)
class KalmanState:
    """Сохранить обученные параметры Калмана для каждого ряда.

    Parameters
    ----------
    parameters : KalmanParameters
        Настройки метода, с которыми обучено состояние.
    protocol_id, population_id : str
        Протокол и идентификатор обучающей выборки.
    layout : str
        Описание осей, часов и единиц из ``input_layout``.
    theta : array_like, shape (n_sites, n_channels, 5)
        Пять конечных параметров каждого ряда в порядке, описанном у
        ``KalmanParameters``.

    Notes
    -----
    ``theta`` копируется в буфер без права записи. Диапазон времени
    обучающей выборки не ограничивает последующий диапазон прогноза.
    """
    parameters: KalmanParameters
    protocol_id: str
    population_id: str
    layout: str
    theta: np.ndarray

    def __post_init__(self):
        from .contracts import DataContractError, snapshot
        shape = np.shape(self.theta)
        if len(shape) != 3 or shape[-1] != 5 or not np.isfinite(self.theta).all():
            raise DataContractError("invalid Kalman fitted coefficients")
        object.__setattr__(self, "theta", snapshot(self.theta, dtype=float, shape=shape, name="theta"))


@dataclass(frozen=True)
class KalmanImputer:
    """Обучать и применять модель Калмана отдельно по постам и каналам.

    Parameters
    ----------
    parameters : KalmanParameters
        Настройки обучения, режима прогноза и начальной ковариации.
    """
    parameters: KalmanParameters

    def fit(self, train_inputs, *, rng):
        """Оценить пять параметров модели для каждого обучающего ряда.

        Parameters
        ----------
        train_inputs : MaskedInputs
            Видимые входы обучающей выборки ``population_kind='train'``, форма
            (n_times, n_sites, n_channels).
        rng : numpy.random.Generator
            Явный генератор случайных чисел общего интерфейса.

        Returns
        -------
        fitted : KalmanState
            Коэффициенты по постам и каналам с привязкой к параметрам,
            протоколу и осям/единицам.

        Notes
        -----
        Каждый ряд требует минимум два видимых наблюдения с положительной
        дисперсией. Оптимизация каждого ряда выполняется ``L-BFGS-B``; её ошибка
        прерывает обучение. Этот детерминированный метод не расходует ``rng``.
        """
        from .contracts import DataContractError
        layout = input_layout(train_inputs)
        if train_inputs.population_kind != "train" or not isinstance(rng, np.random.Generator):
            raise DataContractError("fit requires explicit train population and RNG")
        values = train_inputs.values
        
        if any(np.isfinite(values[:, i, j]).sum() < 2 or np.nanvar(values[:, i, j]) <= 0
               for i in range(values.shape[1]) for j in range(values.shape[2])):
            raise DataContractError("Kalman fit needs varying visible training observations per series")
        theta = np.empty((*values.shape[1:], 5))
        p = self.parameters
        for i in range(values.shape[1]):
            for j in range(values.shape[2]):
                model = KalmanDLM(period=p.period, harmonics=p.harmonics,
                                  initial_variance=p.initial_variance)
                model.fit(values[:, i, j], x0=p.x0, maxiter=p.maxiter)
                theta[i, j] = model.theta
        return KalmanState(p, train_inputs.protocol_id, train_inputs.population_id, layout, theta)

    def predict(self, inputs, *, fitted):
        """Заполнить панель с обученными коэффициентами Калмана.

        Parameters
        ----------
        inputs : MaskedInputs
            Видимые ряды формы (n_times, n_sites, n_channels) в единицах
            входных каналов.
        fitted : KalmanState
            Состояние с теми же параметрами, протоколом, осями, часами и
            единицами.

        Returns
        -------
        result : ImputationResult
            Прогноз и модельное стандартное отклонение в единицах каналов.
            Видимые наблюдения сохраняются точно; ряд без наблюдений
            остаётся недоступным.

        Notes
        -----
        ``interior`` применяет сглаживание, ``tail`` — проход вперёд, ``head`` —
        проход по обращённому ряду. Стандартное отклонение (SD) включает
        дисперсию нового наблюдения
        при фиксированных ``theta``; неопределённость оценивания ``theta`` не включена.
        Модельный SD сохраняется также в видимых позициях.
        """
        from .contracts import DataContractError
        layout = input_layout(inputs)
        if (not isinstance(fitted, KalmanState) or fitted.parameters != self.parameters or
                fitted.protocol_id != inputs.protocol_id or fitted.layout != layout or
                fitted.theta.shape != (*inputs.values.shape[1:], 5)):
            raise DataContractError("Kalman fitted state binding mismatch")
        prediction = np.full(inputs.values.shape, np.nan)
        uncertainty = np.full(inputs.values.shape, np.nan)
        p = self.parameters
        for i in range(prediction.shape[1]):
            for j in range(prediction.shape[2]):
                y = inputs.values[:, i, j]
                if not np.isfinite(y).any():
                    continue  # Среднее для начального уровня нельзя вычислить без наблюдений.
                model = KalmanDLM(period=p.period, harmonics=p.harmonics,
                                  theta=fitted.theta[i, j].copy(), initial_variance=p.initial_variance)
                result = model.smooth(y) if p.mode == "interior" else model.filter_only(y, mode=p.mode)
                prediction[:, i, j], uncertainty[:, i, j] = result
        return _imputation_result(inputs, prediction, uncertainty)


@dataclass(frozen=True)
class SoftImputeAdapter:
    """Заполнять каждый ряд через матрицу последовательных циклов.

    Parameters
    ----------
    parameters : SoftImputeParameters
        Ширина матрицы и параметры заполнения по сингулярному разложению.

    Attributes
    ----------
    no_fit_state : SoftImputeParameters
        Настройки метода, передаваемые в ``predict`` без отдельного
        обучения.
    """
    parameters: SoftImputeParameters

    @property
    def no_fit_state(self):
        """Вернуть параметры для прогноза без отдельного обучения."""
        return self.parameters

    def predict(self, inputs, *, fitted):
        """Заполнить пропуски каждого ряда методом ``SoftImpute``.

        Parameters
        ----------
        inputs : MaskedInputs
            Видимые входы формы (n_times, n_sites, n_channels) в исходных
            единицах.
        fitted : SoftImputeParameters
            Параметры, равные ``no_fit_state`` этого адаптера.

        Returns
        -------
        result : ImputationResult
            Прогноз с точным сохранением видимых значений и
            ``uncertainty=None``.

        Notes
        -----
        Если хотя бы один столбец матрицы циклов не содержит наблюдений,
        пропуски этого ряда остаются ``unavailable``. Обучающая выборка не нужна.
        """
        from .contracts import DataContractError
        input_layout(inputs)
        if not isinstance(fitted, SoftImputeParameters) or fitted != self.parameters:
            raise DataContractError("explicit matching SoftImpute no-fit state required")
        prediction = np.full(inputs.values.shape, np.nan)
        for i in range(prediction.shape[1]):
            for j in range(prediction.shape[2]):
                model = SoftImpute(**self.parameters.to_dict())
                y = inputs.values[:, i, j]
                if np.isfinite(model._to_matrix(y)).any(axis=0).all():
                    prediction[:, i, j], _ = model.fill(y)
        return _imputation_result(inputs, prediction)


def _validate_hgb_estimator(parameters):
    """Проверить численные настройки оценщика HGB."""
    from .contracts import DataContractError
    for name in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
        _positive_integer(getattr(parameters, name), name)
    if parameters.max_leaf_nodes < 2:
        raise DataContractError("hgb.max_leaf_nodes must be at least two")
    for name in ("learning_rate", "l2_regularization"):
        value = getattr(parameters, name)
        if (type(value) not in (int, float) or not np.isfinite(value) or
                value < 0 or (name == "learning_rate" and value == 0)):
            raise DataContractError(f"hgb.{name}: invalid finite value")
    if not (type(parameters.early_stopping) is bool or
            (type(parameters.early_stopping) is str and parameters.early_stopping == "auto")):
        raise DataContractError("hgb.early_stopping must be boolean or 'auto'")


def _hgb_estimator_parameters(parameters, seed):
    """Собрать параметры оценщика HGB с явным ``random_state``."""
    return {**{key: getattr(parameters, key) for key in (
        "max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization",
        "min_samples_leaf", "early_stopping")}, "random_state": seed}


@dataclass(frozen=True)
class HGBParameters:
    """Задать целевой канал и признаки HistGradientBoostingRegressor (HGB).

    Parameters
    ----------
    target : str
        Имя целевого канала панели.
    lags, leads : sequence of int
        Упорядоченные различные положительные сдвиги прошлого/будущего в
        шагах сетки; пустая последовательность разрешена.
    covs, calendar : sequence of str
        Имена дополнительных входных каналов. ``calendar`` выбирает уже
        переданные столбцы, не вычисляет календарь.
    use_leads : bool
        Включить признаки ``leads``. Требуется хотя бы один активный сдвиг
        к прошлым или будущим значениям.
    max_iter : int
        Положительный предел итераций бустинга.
    learning_rate : float
        Положительный темп обучения.
    max_leaf_nodes : int
        Предел числа листьев одного дерева, не менее двух.
    l2_regularization : float
        Неотрицательный вес L2-регуляризации листьев.
    min_samples_leaf : int
        Положительное минимальное число наблюдений в листе.
    early_stopping : bool or {'auto'}
        Правило ранней остановки ``HistGradientBoostingRegressor``.

    Notes
    -----
    Признаки сохраняют единицы входных каналов. Преобразования шкалы
    целевого значения и его ограничения не вводятся этим классом.
    """
    target: str
    lags: tuple
    leads: tuple
    covs: tuple
    calendar: tuple
    use_leads: bool
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    min_samples_leaf: int
    early_stopping: bool | str

    def __post_init__(self):
        from .contracts import DataContractError, strings, text
        text(self.target, "hgb.target")
        for name in ("lags", "leads"):
            values = getattr(self, name)
            if type(values) not in (tuple, list):
                raise DataContractError(f"hgb.{name}: ordered steps required")
            steps = tuple(_positive_integer(v, name) for v in values)
            if len(set(steps)) != len(steps):
                raise DataContractError(f"hgb.{name}: duplicate steps")
            object.__setattr__(self, name, steps)
        for name in ("covs", "calendar"):
            object.__setattr__(self, name, strings(getattr(self, name), name, nonempty=False))
        columns = self.covs + self.calendar
        lag_names = {f"lag{k}" for k in self.lags} | {f"lead{k}" for k in self.leads}
        if len(set(columns)) != len(columns) or set(columns) & (lag_names | {self.target}):
            raise DataContractError("hgb feature columns overlap each other, target or lag names")
        if type(self.use_leads) is not bool:
            raise DataContractError("hgb.use_leads must be boolean")
        
        if not self.lags and not (self.use_leads and self.leads):
            raise DataContractError("hgb requires an active lag or lead")
        _validate_hgb_estimator(self)

    @classmethod
    def from_dict(cls, value):
        """Создать проверенные параметры из словаря полей класса."""
        from .contracts import closed
        from dataclasses import fields
        return cls(**closed(value, {f.name for f in fields(cls)}, where="hgb.parameters"))

    def to_dict(self):
        """Вернуть независимый словарь настроек метода."""
        from dataclasses import asdict
        return asdict(self)

    def estimator_parameters(self, seed):
        """Вернуть настройки sklearn-оценщика с ``random_state`` равным ``seed``."""
        return _hgb_estimator_parameters(self, seed)


@dataclass(frozen=True)
class HGBEstimatorParameters:
    """Задать оценщик HGB для внешнего набора признаков.

    Parameters
    ----------
    target : str
        Имя целевого канала панели.
    max_iter : int
        Положительный предел итераций бустинга.
    learning_rate : float
        Положительный темп обучения.
    max_leaf_nodes : int
        Предел числа листьев одного дерева, не менее двух.
    l2_regularization : float
        Неотрицательный вес L2-регуляризации листьев.
    min_samples_leaf : int
        Положительное минимальное число наблюдений в листе.
    early_stopping : bool or {'auto'}
        Правило ранней остановки ``HistGradientBoostingRegressor``.

    Notes
    -----
    Состав, единицы и временные зависимости признаков задаёт ``FeatureSet``,
    переданный через ``hgb_family``.
    """
    target: str
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    min_samples_leaf: int
    early_stopping: bool | str

    def __post_init__(self):
        from .contracts import text
        text(self.target, "hgb.target")
        _validate_hgb_estimator(self)

    @classmethod
    def from_dict(cls, value):
        """Создать проверенные параметры из словаря полей класса."""
        from dataclasses import fields
        from .contracts import closed
        return cls(**closed(value, {f.name for f in fields(cls)}, where="hgb.parameters"))

    def to_dict(self):
        """Вернуть независимый словарь настроек метода."""
        from dataclasses import asdict
        return asdict(self)

    def estimator_parameters(self, seed):
        """Вернуть настройки sklearn-оценщика с ``random_state`` равным ``seed``."""
        return _hgb_estimator_parameters(self, seed)


@dataclass(frozen=True)
class HGBState:
    """Сохранить обученные модели HGB и привязку их признаков.

    Parameters
    ----------
    parameters : HGBParameters or HGBEstimatorParameters
        Настройки, с которыми обучены модели.
    protocol_id, population_id : str
        Протокол и идентификатор обучающей выборки.
    layout : str
        Оси, часы и единицы входов из ``input_layout``.
    seeds : tuple of int
        ``random_state`` каждого оценщика в порядке постов.
    _models : tuple of object
        По одной обученной модели на пост; состояние сохраняет глубокие
        копии.
    features : FeatureSet or None, optional
        Внешний набор признаков, либо ``None`` для встроенных сдвигов.
    feature_context : {'retrospective', 'causal'}, optional
        Контекст внешнего набора признаков.
    feature_names, feature_units : tuple of str, optional
        Имена и единицы внешних признаков в порядке столбцов; для
        встроенных признаков пусты.

    Attributes
    ----------
    models : tuple of object
        Глубокие копии сохранённых моделей для внешнего использования.
    """
    parameters: HGBParameters | HGBEstimatorParameters
    protocol_id: str
    population_id: str
    layout: str
    seeds: tuple
    _models: tuple = field(repr=False, compare=False)
    features: object | None = field(default=None, repr=False, compare=False)
    feature_context: str = "retrospective"
    feature_names: tuple = ()
    feature_units: tuple = ()

    def __post_init__(self):
        from copy import deepcopy
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "_models", tuple(deepcopy(m) for m in self._models))

    @property
    def models(self):
        """Вернуть независимые глубокие копии обученных моделей."""
        from copy import deepcopy
        return tuple(deepcopy(m) for m in self._models)


@dataclass(frozen=True)
class HGBImputer:
    """Заполнять целевой канал отдельной моделью HGB для каждого поста.

    Parameters
    ----------
    parameters : HGBParameters or HGBEstimatorParameters
        ``HGBParameters`` для встроенных сдвигов; ``HGBEstimatorParameters`` для
        внешних признаков.
    features : FeatureSet or None, optional
        Внешний набор признаков. При ``None`` используются встроенные сдвиги
        и каналы.
    feature_context : {'retrospective', 'causal'}, optional
        Контекст внешних признаков; встроенные признаки требуют
        ``retrospective``.

    Notes
    -----
    Для ``causal`` внешний ``FeatureSet`` должен объявлять ``uses_future=False``.
    Выбор признаков не задаёт общую причинность обучения и заполнения.
    """
    parameters: HGBParameters | HGBEstimatorParameters
    features: object | None = None
    feature_context: str = "retrospective"

    def __post_init__(self):
        from .contracts import DataContractError
        from .features import FeatureSet
        if self.feature_context not in ("retrospective", "causal"):
            raise DataContractError("feature context must be retrospective or causal")
        if self.features is None:
            if not isinstance(self.parameters, HGBParameters) or self.feature_context != "retrospective":
                raise DataContractError("built-in HGB features require HGBParameters and retrospective context")
        elif not isinstance(self.features, FeatureSet) or not isinstance(self.parameters, HGBEstimatorParameters):
            raise DataContractError("injected HGB features require FeatureSet and HGBEstimatorParameters")
        elif self.feature_context == "causal" and self.features.uses_future is not False:
            raise DataContractError("causal features require an explicit no-future dependency")

    def _columns(self, inputs):
        """Проверить входные каналы и вернуть индексы цели и дополнительных столбцов."""
        from .contracts import DataContractError
        input_layout(inputs)
        channels = inputs.spec.to_dict()["channels"]
        required = (self.parameters.target,)
        if self.features is None:
            required += self.parameters.covs + self.parameters.calendar
        if any(name not in channels for name in required):
            raise DataContractError("hgb requires all declared target/feature channels")
        return channels.index(self.parameters.target), {
            name: channels.index(name) for name in required[1:]
        }

    def _feature_values(self, inputs):
        """Вычислить внешние признаки в контексте этого метода."""
        return self.features.transform(inputs, context=self.feature_context)

    def fit(self, train_inputs, *, rng):
        """Обучить модель HGB на видимых целевых значениях каждого поста.

        Parameters
        ----------
        train_inputs : MaskedInputs
            Видимые входы обучающей выборки ``population_kind='train'``, форма
            (n_times, n_sites, n_channels).
        rng : numpy.random.Generator
            Явный генератор случайных чисел общего интерфейса.

        Returns
        -------
        fitted : HGBState
            По одной модели на пост с параметрами, осями/единицами и
            составом внешних признаков.

        Notes
        -----
        На каждом посту требуется хотя бы одно видимое целевое наблюдение.
        Для каждого оценщика ``rng`` выдаёт отдельный ``random_state``. Скрытые
        ответы не используются. Встроенные сдвиги при обучении сохраняют ``NaN``;
        внешний набор применяется через ``FeatureSet.transform``.
        """
        from .contracts import DataContractError
        target, columns = self._columns(train_inputs)
        if train_inputs.population_kind != "train" or not isinstance(rng, np.random.Generator):
            raise DataContractError("fit requires explicit train population and RNG")
        if not np.isfinite(train_inputs.values[:, :, target]).any(axis=0).all():
            raise DataContractError("hgb fit requires visible training targets at every site")
        p, models, seeds = self.parameters, [], []
        features = self._feature_values(train_inputs) if self.features is not None else None
        for site in range(train_inputs.values.shape[1]):
            seed = int(rng.integers(0, 2**32 - 1))
            if features is None:
                model = HGBFiller(list(p.lags), list(p.leads), list(p.covs), list(p.calendar),
                                  p.estimator_parameters(seed))
                frame = pd.DataFrame({name: train_inputs.values[:, site, j] for name, j in columns.items()})
                model.fit(train_inputs.values[:, site, target], frame, use_leads=p.use_leads)
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                visible = train_inputs.visible[:, site, target]
                model = HistGradientBoostingRegressor(**p.estimator_parameters(seed))
                model.fit(features.values[visible, site, :], train_inputs.values[visible, site, target])
            models.append(model)
            seeds.append(seed)
        return HGBState(p, train_inputs.protocol_id, train_inputs.population_id,
                        input_layout(train_inputs), tuple(seeds), tuple(models), self.features,
                        self.feature_context, features.names if features is not None else (),
                        features.units if features is not None else ())

    def predict(self, inputs, *, fitted):
        """Заполнить целевой канал обученными моделями HGB.

        Parameters
        ----------
        inputs : MaskedInputs
            Видимые входы формы (n_times, n_sites, n_channels) в исходных
            единицах.
        fitted : HGBState
            Состояние с совпадающими параметрами, протоколом,
            осями/часами/единицами и тем же объектом внешних признаков.

        Returns
        -------
        result : ImputationResult
            Прогноз целевого канала с сохранением видимых значений и
            остальных каналов; ``uncertainty=None``.

        Notes
        -----
        Встроенный HGB перед сдвигом к прошлым значениям продолжает последнее
        наблюдение, перед сдвигом к будущим — следующее. Внешний ``FeatureSet``
        применяется без такого продолжения,
        с теми же именами и единицами, что при ``fit``. Прогнозы обратно
        в признаки не подаются.
        """
        from .contracts import DataContractError
        target, columns = self._columns(inputs)
        if (not isinstance(fitted, HGBState) or fitted.parameters != self.parameters or
                fitted.protocol_id != inputs.protocol_id or fitted.layout != input_layout(inputs) or
                len(fitted._models) != inputs.values.shape[1] or fitted.features is not self.features or
                fitted.feature_context != self.feature_context):
            raise DataContractError("HGB fitted state binding mismatch")
        features = self._feature_values(inputs) if self.features is not None else None
        if features is not None and (features.names != fitted.feature_names or features.units != fitted.feature_units):
            raise DataContractError("HGB fitted feature layout/units mismatch")
        prediction = inputs.values.copy()
        for site, model in enumerate(fitted._models):
            y = inputs.values[:, site, target]
            if not np.isnan(y).any():
                continue  # При отсутствии пропусков predict не вызывается: sklearn не принимает ноль строк.
                          
            if features is None:
                frame = pd.DataFrame({name: inputs.values[:, site, j] for name, j in columns.items()})
                prediction[:, site, target], _ = model.fill(y, frame)
            else:
                missing = ~inputs.visible[:, site, target]
                prediction[missing, site, target] = model.predict(features.values[missing, site, :])
        return _imputation_result(inputs, prediction)


def hgb_family(features, *, context="retrospective"):
    """Создать семейство HGB с пользовательским набором признаков.

    Parameters
    ----------
    features : FeatureSet
        Набор признаков, одинаково применяемый при обучении и прогнозе.
    context : {'retrospective', 'causal'}, optional
        Временной контекст признаков; ``causal`` требует ``uses_future=False``.

    Returns
    -------
    family : ImputerFamily
        Семейство с ``HGBEstimatorParameters`` для передачи под выбранным
        именем в ``families``.
    """
    from .contracts import DataContractError
    from .features import FeatureSet
    if not isinstance(features, FeatureSet) or context not in ("retrospective", "causal"):
        raise DataContractError("hgb_family requires FeatureSet and explicit feature context")
    if context == "causal" and features.uses_future is not False:
        raise DataContractError("causal features require an explicit no-future dependency")
    return ImputerFamily(lambda p: HGBEstimatorParameters.from_dict(p).to_dict(),
                         lambda p: HGBImputer(HGBEstimatorParameters.from_dict(p), features, context))
