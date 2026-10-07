"""Численные методы заполнения одномерных рядов с пропусками NaN."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import PchipInterpolator



def fill_linear(y: np.ndarray) -> tuple[np.ndarray, None]:
    """Заполнить пропуски ряда линейной интерполяцией.

    Parameters
    ----------
    y : ndarray of float, shape (n,)
        Значения ряда в одной единице; ``NaN`` обозначает пропуск.
        Координата интерполяции — номер шага.

    Returns
    -------
    filled : ndarray of float, shape (n,)
        Ряд в исходной единице с сохранёнными наблюдениями. Пропуски на
        краях получают ближайшее наблюдение; полностью пустой ряд
        остаётся с ``NaN``.
    standard_deviation : None
        Стандартное отклонение этим методом не оценивается.
    """
    s = pd.Series(y)
    return s.interpolate("linear", limit_direction="both").to_numpy(), None


def fill_pchip(y: np.ndarray) -> tuple[np.ndarray, None]:
    """Заполнить пропуски ряда кусочно-кубической интерполяцией PCHIP.

    Parameters
    ----------
    y : ndarray of float, shape (n,)
        Значения ряда в одной единице; ``NaN`` обозначает пропуск. Требуются
        не менее двух наблюдений; координата — номер шага.

    Returns
    -------
    filled : ndarray of float, shape (n,)
        Ряд в исходной единице с сохранёнными наблюдениями. За крайними
        наблюдениями используется экстраполяция PCHIP.
    standard_deviation : None
        Стандартное отклонение этим методом не оценивается.
    """
    obs = ~np.isnan(y)
    x = np.arange(len(y))
    f = PchipInterpolator(x[obs], y[obs], extrapolate=True)
    out = y.copy()
    out[~obs] = f(x[~obs])
    return out, None



@dataclass
class KalmanDLM:
    r"""Моделировать ряд уровнем, гармониками и авторегрессией первого порядка.

    Parameters
    ----------
    period : int
        Положительный период первой гармоники в шагах ряда.
    harmonics : int, optional
        Неотрицательное число сезонных гармоник. По умолчанию 3.
    theta : ndarray of float, shape (5,), optional
        Пять конечных параметров: логарифмы численных значений дисперсий
        уровня, сезонной и AR-компоненты; параметр AR до tanh; логарифм
        численного значения дисперсии наблюдения. Логарифмы относятся к
        выбранной единице ряда. ``None`` означает отсутствие параметров.
    initial_variance : float, optional
        Положительная конечная начальная дисперсия каждой компоненты
        состояния в квадрате единицы ряда. По умолчанию 10.

    Attributes
    ----------
    Z : ndarray of float, shape (m,)
        Безразмерный вектор наблюдения. Создаётся в ``fit``, ``smooth``
        или ``filter_only``.
    T : ndarray of float, shape (m, m)
        Безразмерная матрица перехода: уровень, сезонные вращения и AR.

    Notes
    -----
    AR(1) обозначает авторегрессию первого порядка.
    В математической модели :math:`x_t` — состояние, :math:`y_t` —
    наблюдение; все компоненты состояния имеют единицу ряда. При
    фиксированных ``theta`` уравнения имеют вид

    .. math::

        \begin{aligned}
        x_t &= T x_{t-1} + w_t, & \operatorname{Cov}(w_t) &= Q, \\
        y_t &= Z x_t + v_t, & \operatorname{Var}(v_t) &= r.
        \end{aligned}

    Шумы :math:`w_t` и :math:`v_t` предполагаются центрированными
    гауссовыми, независимыми между шагами, друг от друга и от начального
    состояния. При заданных начальных среднем и ковариации начальное
    состояние также предполагается гауссовым. Это предпосылки модели,
    а не свойства, проверенные на данных.
    :math:`Q` и :math:`r` имеют единицу ряда в квадрате.

    Пусть :math:`K` равно ``harmonics``. Состояние содержит уровень,
    две компоненты каждой гармоники и компоненту AR. Его размер и углы
    сезонных вращений равны

    .. math::

        m = 2K+2, \qquad
        \omega_j = \frac{2\pi j}{\mathrm{period}}, \qquad j=1,\ldots,K.

    Диагональ :math:`Q` содержит :math:`q_{\rm level}`, по два
    :math:`q_{\rm season}` на гармонику и :math:`q_{\rm AR}`. Их численные
    значения и :math:`r` получаются экспонентой соответствующих ``theta``;
    безразмерный коэффициент AR равен :math:`\phi=\tanh(\theta_4)`, где
    индекс 4 отсчитывается от единицы.

    Начальный уровень равен среднему конечных наблюдений всего переданного
    ряда, остальные компоненты — нулю. Начальная ковариация :math:`P_0`
    задаётся как :math:`P_0=\mathtt{initial\_variance}\,I_m`, где
    :math:`I_m` — единичная матрица размера m. Нужен хотя бы один конечный
    отсчёт; даже прямой проход инициализируется ретроспективно.
    Переход применяется перед каждым обновлением, включая первый шаг.
    """
    period: int
    harmonics: int = 3
    theta: np.ndarray | None = None
    initial_variance: float = field(default=10.0, kw_only=True)
    Z: np.ndarray = field(init=False, repr=False)
    T: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if (isinstance(self.initial_variance, bool) or not isinstance(self.initial_variance, Real)
                or not np.isfinite(self.initial_variance) or self.initial_variance <= 0):
            raise ValueError("initial_variance must be a finite positive real number")

    def _build(self, theta):
        """Построить T, Z, Q и дисперсию наблюдения из пяти параметров ``theta``."""
        q_lvl, q_seas, q_ar, phi_raw, r = theta
        phi = np.tanh(phi_raw)
        K = self.harmonics
        m = 1 + 2 * K + 1
        T = np.zeros((m, m))
        Z = np.zeros(m)
        Q = np.zeros((m, m))
        T[0, 0] = 1
        Z[0] = 1
        Q[0, 0] = np.exp(q_lvl)
        for j in range(K):
            w = 2 * np.pi * (j + 1) / self.period
            i = 1 + 2 * j
            T[i:i + 2, i:i + 2] = [[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]]
            Z[i] = 1
            Q[i, i] = Q[i + 1, i + 1] = np.exp(q_seas)
        T[-1, -1] = phi
        Z[-1] = 1
        Q[-1, -1] = np.exp(q_ar)
        self.Z, self.T = Z, T
        return T, Z, Q, np.exp(r)

    def _filter(self, y, theta, store=True):
        """Вычислить логарифм правдоподобия и при ``store`` сохранить состояния фильтра."""
        T, Z, Q, R = self._build(theta)
        m = len(Z)
        n = len(y)
        a = np.zeros(m)
        a[0] = np.nanmean(y)
        P = np.eye(m) * self.initial_variance
        if store:
            at = np.zeros((n, m))
            Pt = np.zeros((n, m, m))
            af = np.zeros((n, m))
            Pf = np.zeros((n, m, m))
        ll = 0.0
        for k in range(n):
            # Переход выполняется до обновления по наблюдению, включая первый шаг.
            a = T @ a
            P = T @ P @ T.T + Q
            if store:
                at[k], Pt[k] = a, P
            if not np.isnan(y[k]):
                v = y[k] - Z @ a
                F = Z @ P @ Z + R
                Kg = P @ Z / F
                a = a + Kg * v
                P = P - np.outer(Kg, Z @ P)
                ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
            if store:
                af[k], Pf[k] = a, P
        if store:
            self._at, self._Pt, self._af, self._Pf = at, Pt, af, Pf
        return ll

    def fit(self, y: np.ndarray, x0=None, maxiter: int = 60):
        """Оценить параметры модели по правдоподобию фильтра Калмана.

        Parameters
        ----------
        y : ndarray of float, shape (n,)
            Ряд в одной единице с ``NaN`` вместо пропусков. Для начального
            приближения по умолчанию нужны наблюдения с ненулевой
            дисперсией.
        x0 : array_like of float, shape (5,), optional
            Начальные log(q_level), log(q_season), log(q_ar), atanh(phi),
            log(r). По умолчанию дисперсии равны соответственно 1e-4, 1e-5,
            0.3 и 0.2 от дисперсии наблюдений, а phi = 0.6.
        maxiter : int, optional
            Максимальное число итераций ``L-BFGS-B``; по умолчанию 60.

        Returns
        -------
        self : KalmanDLM
            Та же модель с сохранённым конечным вектором ``theta``.

        Notes
        -----
        При неуспешной оптимизации или неконечном результате вызывается
        ``RuntimeError``; прежний ``theta`` сохраняется. Начальные среднее и ковариация
        определены в ``KalmanDLM``. Ошибка оценки самих параметров не включается
        в стандартное отклонение (SD)
        методов ``smooth`` и ``filter_only``.
        """
        if x0 is None:
            v = np.nanvar(y)
            x0 = np.array([np.log(v * 1e-4), np.log(v * 1e-5), np.log(v * 0.3), np.arctanh(0.6), np.log(v * 0.2)])
        res = optimize.minimize(lambda th: -self._filter(y, th, store=False), x0, method="L-BFGS-B",
                                options={"maxiter": maxiter})
        error = f"Kalman fit did not return a usable solution (status {res.status}): {res.message}"
        if not res.success:
            raise RuntimeError(error)
        try:
            theta = np.asarray(res.x)
            fun = np.asarray(res.fun)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(error) from exc
        if (theta.shape != (5,) or theta.dtype.kind not in "fiu" or not np.isfinite(theta).all()
                or fun.shape != () or fun.dtype.kind not in "fiu" or not np.isfinite(fun)):
            raise RuntimeError(error)
        self.theta = theta.copy()
        return self

    def smooth(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Вычислить сглаженный сигнал и модельное стандартное отклонение.

        Parameters
        ----------
        y : ndarray of float, shape (n,)
            Конечные значения ряда в одной единице; ``NaN`` обозначает пропуск.
            Нужны хотя бы одно наблюдение и заданный ``theta``.

        Returns
        -------
        mean : ndarray of float, shape (n,)
            Среднее сигнала после фильтра и обратного сглаживания по всему
            ряду в единице y. Наблюдаемые позиции тоже заменяются средним модели.
        standard_deviation : ndarray of float, shape (n,)
            Модельное стандартное отклонение (SD) нового наблюдения с
            независимым шумом в единице y.

        Notes
        -----
        При предпосылках ``KalmanDLM`` параметры ``theta``, начальное среднее
        и начальная ковариация считаются фиксированными. Пусть
        :math:`P_t^s` — сглаженная ковариация состояния после учёта всего ряда.
        :math:`Z` — вектор наблюдения, :math:`r` — дисперсия его шума из той же
        модели. Возвращаемое стандартное отклонение вычисляется как

        .. math::

            \mathrm{SD}_t = \sqrt{Z P_t^s Z^{\mathsf T} + r}.

        Это SD нового модельного наблюдения, а не уже известного отсчёта.
        Неопределённость оценённых ``theta`` и начального среднего, выбранного
        по данным, не учитывается. ``KalmanImputer``
        отдельно сохраняет исходные видимые значения в итоговом заполнении.
        """
        if self.theta is None:
            raise RuntimeError("fit() first")
        self._filter(y, self.theta)
        T = self.T
        Z = self.Z
        n, m = self._af.shape
        a_s = self._af.copy()
        P_s = self._Pf.copy()
        for k in range(n - 2, -1, -1):
            J = self._Pf[k] @ T.T @ np.linalg.pinv(self._Pt[k + 1])
            a_s[k] = self._af[k] + J @ (a_s[k + 1] - self._at[k + 1])
            P_s[k] = self._Pf[k] + J @ (P_s[k + 1] - self._Pt[k + 1]) @ J.T
        yhat = a_s @ Z
        R = np.exp(self.theta[-1])
        sd = np.sqrt(np.einsum("i,kij,j->k", Z, P_s, Z) + R)
        return yhat, sd

    def filter_only(self, y: np.ndarray, mode: str = "tail") -> tuple[np.ndarray, np.ndarray]:
        r"""Вычислить прогноз до текущего обновления и стандартное отклонение.

        Parameters
        ----------
        y : ndarray of float, shape (n,)
            Конечные значения ряда в одной единице; ``NaN`` обозначает пропуск.
            Нужны хотя бы одно наблюдение и заданный ``theta``.
        mode : {"tail", "head"}, optional
            ``"tail"`` проходит ряд вперёд, ``"head"`` — в обратном порядке.
            По умолчанию ``"tail"``.

        Returns
        -------
        mean : ndarray of float, shape (n,)
            Среднее до обновления по текущему наблюдению, в единице y и
            исходном порядке времени.
        standard_deviation : ndarray of float, shape (n,)
            Модельное стандартное отклонение (SD) нового наблюдения до
            обновления, в единице y.

        Notes
        -----
        При предпосылках ``KalmanDLM`` параметры ``theta``, начальное среднее
        и начальная ковариация считаются фиксированными. Пусть
        :math:`P_t^-` — ковариация состояния перед текущим обновлением в
        выбранном направлении прохода. :math:`Z` и :math:`r` — вектор
        наблюдения и дисперсия его шума из модели класса. Возвращается

        .. math::

            \mathrm{SD}_t = \sqrt{Z P_t^- Z^{\mathsf T} + r}.

        Неопределённость оценённых ``theta`` и начального среднего, выбранного
        по данным, не учитывается. Начальный уровень
        использует среднее всего переданного ряда, поэтому ``"tail"``
        не является потоковым прогнозом. ``"head"`` использует более поздние
        наблюдения исходного ряда.
        """
        if mode not in ("head", "tail"):
            raise ValueError("mode must be 'head' or 'tail'")
        if self.theta is None:
            raise RuntimeError("fit() first")
        yy = y[::-1] if mode == "head" else y
        self._filter(yy, self.theta)
        yhat = self._at @ self.Z          # Прогноз до обновления по текущему наблюдению.
        R = np.exp(self.theta[-1])
        sd = np.sqrt(np.einsum("i,kij,j->k", self.Z, self._Pt, self.Z) + R)
        if mode == "head":
            yhat, sd = yhat[::-1], sd[::-1]
        return yhat, sd



@dataclass
class SoftImpute:
    r"""Заполнять матрицу циклов пороговым преобразованием сингулярных чисел.

    Parameters
    ----------
    cycle_length : int
        Положительное число шагов в строке матрицы. Неполный последний
        цикл дополняется ``NaN``.
    lam : float, optional
        Неотрицательный порог сингулярных чисел и коэффициент ядерной нормы
        в единице ряда. По умолчанию 2.
    rank_max : int, optional
        Положительный предел числа сохраняемых компонент. По умолчанию 8.
    max_iter : int, optional
        Положительный предел итераций. По умолчанию 200.
    tol : float, optional
        Неотрицательный безразмерный множитель численного условия остановки.
        По умолчанию 1e-4.

    Attributes
    ----------
    history : list of float
        Значения критерия после каждой итерации ``fill`` в квадрате единицы
        ряда. Устанавливается при вызове ``fill``.

    Notes
    -----
    Пусть :math:`X` — матрица циклов, :math:`\Omega` — индексы её конечных
    наблюдений, :math:`Z` — матрица текущей оценки той же формы и
    :math:`\lambda` равно ``lam``. Ядерная норма :math:`\lVert Z\rVert_*`
    есть сумма сингулярных чисел. Вычисляемый критерий равен

    .. math::

        F(Z) = \frac12\sum_{(i,j)\in\Omega}(X_{ij}-Z_{ij})^2
             + \lambda\lVert Z\rVert_*.

    На каждой итерации наблюдаемые элементы берутся из :math:`X`,
    пропуски — из предыдущей :math:`Z`. Сингулярные числа уменьшаются
    на ``lam`` с отсечением у нуля; сохраняется не более ``rank_max``
    компонент. Начальное заполнение использует среднее наблюдений своего
    столбца, поэтому каждому столбцу нужно хотя бы одно конечное наблюдение.

    Ограничение ``rank_max`` отличает этот алгоритм от SoftImpute без
    жёсткого ограничения ранга. Сходимость к глобальному минимуму критерия
    не гарантируется. Пусть :math:`F_k=F(Z_k)` — значение после итерации k
    и :math:`\varepsilon` равно ``tol``. Остановка проверяется по условию

    .. math::

        |F_k-F_{k-1}| < \varepsilon\max(1,F_k).

    Число 1 в пороге относится к численному масштабу критерия без
    нормирования ряда. Смена численной единицы может изменить остановку.
    """
    cycle_length: int
    lam: float = 2.0
    rank_max: int = 8
    max_iter: int = 200
    tol: float = 1e-4

    def _to_matrix(self, y):
        """Разложить ряд по циклам и дополнить неполный последний цикл значениями ``NaN``."""
        n = len(y)
        n_cycles = int(np.ceil(n / self.cycle_length))
        M = np.full(n_cycles * self.cycle_length, np.nan)
        M[:n] = y
        return M.reshape(n_cycles, self.cycle_length)

    def fill(self, y: np.ndarray) -> tuple[np.ndarray, None]:
        """Заполнить пропуски ряда по матрице циклов.

        Parameters
        ----------
        y : ndarray of float, shape (n,)
            Ряд в одной единице с ``NaN`` вместо пропусков. Каждому столбцу
            матрицы циклов нужно хотя бы одно наблюдение.

        Returns
        -------
        filled : ndarray of float, shape (n,)
            Ряд в исходной единице; меняются только позиции ``NaN``. Дополнение
            последнего цикла в результат не входит.
        standard_deviation : None
            Стандартное отклонение этим методом не оценивается.

        Notes
        -----
        Критерий, порог сингулярных чисел и условие остановки описаны в ``SoftImpute``.
        Значения критерия записываются в ``history``; они не являются оценкой ошибки
        по скрытым контрольным ответам.
        """
        X = self._to_matrix(y)
        Omega = ~np.isnan(X)
        col_mean = np.nanmean(X, axis=0)
        Z = np.where(Omega, X, col_mean[None, :])
        f_prev = np.inf
        self.history = []
        for it in range(self.max_iter):
            U, s, Vt = np.linalg.svd(np.where(Omega, X, Z), full_matrices=False)
            s_thr = np.maximum(s - self.lam, 0)[: self.rank_max]
            Z_new = (U[:, : self.rank_max] * s_thr) @ Vt[: self.rank_max]
            f = 0.5 * np.nansum((X - Z_new)[Omega] ** 2) + self.lam * s_thr.sum()
            self.history.append(f)
            if abs(f_prev - f) < self.tol * max(1.0, f):
                Z = Z_new
                break
            Z, f_prev = Z_new, f
        out = y.copy()
        flat = Z.ravel()[: len(y)]
        out[np.isnan(y)] = flat[np.isnan(y)]
        return out, None



@dataclass
class HGBFiller:
    """Заполнять ряд HistGradientBoostingRegressor по сдвигам и другим столбцам.

    Parameters
    ----------
    lags : list of int
        Положительные сдвиги к прошлым значениям y, в шагах ряда.
    leads : list of int
        Положительные сдвиги к будущим значениям y, в шагах ряда;
        используются, если ``fit`` вызван с ``use_leads=True``.
    covs : list of str
        Имена столбцов X с ковариатами; единицы каждого столбца задаёт
        вызывающий код.
    calendar : list of str
        Имена столбцов X с календарными признаками; их шкалы задаёт
        вызывающий код.
    params : dict, optional
        Аргументы ``HistGradientBoostingRegressor`` без дополнительного
        преобразования. По умолчанию пустой словарь.
    model_ : object or None, optional
        Регрессор, заменяемый при ``fit``. По умолчанию ``None``; обычный
        порядок использования — ``fit``, затем ``fill``.

    Attributes
    ----------
    use_leads_ : bool
        Признак использования будущих значений, устанавливаемый ``fit`` и
        используемый ``fill``.

    Notes
    -----
    HGB обозначает ансамблевую модель ``HistGradientBoostingRegressor``.
    """
    lags: list[int]
    leads: list[int]
    covs: list[str]
    calendar: list[str]
    params: dict = field(default_factory=dict)
    model_: object = None

    def _features(self, y: np.ndarray, X: pd.DataFrame, use_leads: bool) -> pd.DataFrame:
        """Построить lag/lead целевого ряда и добавить выбранные столбцы X."""
        s = pd.Series(y)
        F = {f"lag{l}": s.shift(l) for l in self.lags}
        if use_leads:
            F.update({f"lead{l}": s.shift(-l) for l in self.leads})
        F = pd.DataFrame(F)
        for c in self.covs + self.calendar:
            F[c] = X[c].to_numpy()
        return F

    def fit(self, y: np.ndarray, X: pd.DataFrame, use_leads: bool = True):
        """Обучить HGB-регрессор на наблюдаемых значениях ряда.

        Parameters
        ----------
        y : ndarray of float, shape (n,)
            Ряд в единице целевого канала с ``NaN`` вместо пропусков; нужны
            наблюдаемые значения.
        X : pandas.DataFrame, shape (n, p)
            Дополнительные столбцы с именами из ``covs`` и ``calendar``, в том же
            порядке строк, что y. Индекс не задаёт числовую координату
            сдвигов.
        use_leads : bool, optional
            Включить признаки будущих значений y. По умолчанию ``True``.

        Returns
        -------
        self : HGBFiller
            Та же модель с обученным ``model_`` и сохранённым ``use_leads_``.

        Notes
        -----
        В обучении участвуют строки с наблюдаемым y. Пропуски в сдвинутых
        признаках передаются оценщику без предварительного заполнения. Этот
        численный метод получает обычный y; скрытие контрольных ответов
        обеспечивается ``MaskedInputs`` в адаптере ``HGBImputer``.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor
        F = self._features(y, X, use_leads)
        ok = ~np.isnan(y)
        self.model_ = HistGradientBoostingRegressor(**self.params).fit(F[ok], y[ok])
        self.use_leads_ = use_leads
        return self

    def fill(self, y: np.ndarray, X: pd.DataFrame) -> tuple[np.ndarray, None]:
        """Заполнить пропуски ряда обученным HGB-регрессором.

        Parameters
        ----------
        y : ndarray of float, shape (n,)
            Конечные значения в единице целевого канала; ``NaN`` обозначает
            пропуск. Требуется хотя бы одна позиция ``NaN``.
        X : pandas.DataFrame, shape (n, p)
            Столбцы ``covs`` и ``calendar`` в том же порядке строк, что y,
            и с теми же шкалами, что при ``fit``.

        Returns
        -------
        filled : ndarray of float, shape (n,)
            Ряд в исходной единице с сохранёнными наблюдениями и прогнозами
            только в позициях ``NaN``.
        standard_deviation : None
            Стандартное отклонение этим методом не оценивается.

        Notes
        -----
        Перед сдвигом к прошлым значениям последнее наблюдение продолжается
        вперёд. Перед сдвигом к будущим значениям следующее наблюдение
        продолжается назад.
        Прогнозы не подставляются обратно в признаки. Требуются обученные
        ``model_`` и ``use_leads_``. Панельный ``HGBImputer`` сохраняет ряд
        без пропусков без вызова этого численного метода.
        """
        F = self._features(y, X, self.use_leads_)
        out = y.copy()
        miss = np.isnan(y)
        # Признаки строятся из наблюдений, поэтому прогнозы не подаются обратно.
        Ff = F.copy()
        s = pd.Series(y)
        for l in self.lags:
            Ff[f"lag{l}"] = s.ffill().shift(l).to_numpy()
        if self.use_leads_:
            for l in self.leads:
                Ff[f"lead{l}"] = s.bfill().shift(-l).to_numpy()
        out[miss] = self.model_.predict(Ff[miss])
        return out, None
