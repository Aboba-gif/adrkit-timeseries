"""Оценка свободного члена и наклона двух линейных зависимостей.

Численная оценка сравнивается с аналитическим решением квадратичной задачи.
"""
import numpy as np

from adrkit.contracts import MapCapabilities, ScalarCapabilities
from adrkit.inverse import InverseProblem
from adrkit.spaces import ArraySpace


class TwoChannelLine:
    """Два линейных канала со свободным членом и наклоном.

    Parameters
    ----------
    times : array_like, shape (m,)
        Безразмерные моменты наблюдений.

    Notes
    -----
    Параметры имеют форму (2, 2): канал, затем свободный член и наклон.
    Прогноз имеет форму (m, 2): момент, затем канал; единицы безразмерны.
    """
    def __init__(self, times):
        self.times = np.array(times, dtype=float, copy=True)
        self.domain = ArraySpace((2, 2), axes=("channel", "term"),
                                 coordinates=(("A", "B"), ("intercept", "slope")),
                                 units="dimensionless", binding="line-coefficients")
        self.codomain = ArraySpace((len(times), 2), axes=("sample", "channel"),
                                   coordinates=(tuple(times), ("A", "B")),
                                   units="dimensionless", binding="curve-observations")

    def predict(self, theta):
        """Вычислить два линейных канала.

        Parameters
        ----------
        theta : ndarray, shape (2, 2)
            Свободный член и наклон для каждого канала.

        Returns
        -------
        prediction : ndarray, shape (m, 2)
            Безразмерные прогнозы в порядке времени и канала.
        """
        return theta[:, 0][None, :] + self.times[:, None]*theta[:, 1][None, :]

    def vjp(self, at, cotangent):
        """Перенести ковектор наблюдений к параметрам прямых.

        Parameters
        ----------
        at : ndarray, shape (2, 2)
            Точка параметров; производная линейной модели от неё не
            зависит.
        cotangent : ndarray, shape (m, 2)
            Ковектор наблюдений по времени и каналу.

        Returns
        -------
        covector : ndarray, shape (2, 2)
            Ковектор по свободному члену и наклону каждого канала.
        """
        return np.column_stack((cotangent.sum(axis=0), self.times @ cotangent))

    def capabilities(self):
        """Предоставить прогноз и VJP как MapCapabilities."""
        return MapCapabilities(self, self.domain, self.codomain, vjp=self)


class SquaredError:
    """Квадратичная ошибка наблюдений в евклидовой метрике.

    Parameters
    ----------
    observed : array_like
        Наблюдения формы layout.shape.
    layout : ArraySpace
        Пространство наблюдений.
    """
    def __init__(self, observed, layout):
        self.observed = layout.array(observed, copy=True)
        self.layout = layout

    def value(self, predicted):
        """Вычислить половину суммы квадратов ошибок.

        Parameters
        ----------
        predicted : ndarray
            Прогноз формы layout.shape.

        Returns
        -------
        value : numpy.floating
            Квадратичная ошибка наблюдений.
        """
        return .5*np.sum((predicted-self.observed)**2)

    def cotangent(self, predicted):
        """Вычислить координатный ковектор квадратичной ошибки.

        Parameters
        ----------
        predicted : ndarray
            Прогноз формы layout.shape.

        Returns
        -------
        covector : ndarray
            Разность прогноза и наблюдений той же формы.
        """
        return predicted-self.observed

    def capabilities(self):
        """Предоставить ошибку и ковектор как ScalarCapabilities."""
        return ScalarCapabilities(self, self.layout, self)


class RidgePenalty:
    """Квадратичный штраф безразмерных коэффициентов.

    Parameters
    ----------
    strength : float
        Вес половины суммы квадратов коэффициентов.
    layout : ArraySpace
        Пространство коэффициентов.
    """
    def __init__(self, strength, layout):
        self.strength, self.layout = strength, layout

    def value(self, theta):
        """Вычислить квадратичный штраф коэффициентов.

        Parameters
        ----------
        theta : ndarray
            Коэффициенты формы layout.shape.

        Returns
        -------
        value : numpy.floating
            Взвешенная половина суммы квадратов.
        """
        return .5*self.strength*np.sum(theta**2)

    def cotangent(self, theta):
        """Вычислить координатный ковектор штрафа.

        Parameters
        ----------
        theta : ndarray
            Коэффициенты формы layout.shape.

        Returns
        -------
        covector : ndarray
            Коэффициенты, умноженные на strength, той же формы.
        """
        return self.strength*theta

    def capabilities(self):
        """Предоставить штраф и ковектор как ScalarCapabilities."""
        return ScalarCapabilities(self, self.layout, self)


def main():
    """Подогнать два линейных канала с покомпонентными границами."""
    model = TwoChannelLine([-1., 0., 1.])
    truth = np.array([[1., 2.], [-.5, .25]])
    observed = model.predict(truth)
    loss = SquaredError(observed, model.codomain)
    penalty = RidgePenalty(.2, model.domain)
    problem = InverseProblem(model.capabilities(), loss.capabilities(),
                             penalty=penalty.capabilities(),
                             bounds=[(None, None), (-1., 1.), (None, None), (None, None)])
    result = problem.solve(np.zeros(model.domain.shape), options={"gtol": 1e-10, "ftol": 1e-14})
    if not result.success:
        raise RuntimeError(result.message)
    estimate = result.x.reshape(problem.parameter_space.shape)
    # Столбцы единицы и времени ортогональны; решение покоординатное.
    expected = truth*np.array([3./3.2, 2./2.2])
    expected[0, 1] = 1.  # Верхняя граница для наклона в канале A.
    np.testing.assert_allclose(estimate, expected, atol=1e-8, rtol=1e-8)
    print("Оценки [свободный член, наклон] по каналам:")
    print(estimate)
    print("Оценка совпадает с аналитическим решением задачи с заданными границами.")
    print("Строгая выпуклость этой задачи обеспечивает единственность решения.")


if __name__ == "__main__":
    main()
