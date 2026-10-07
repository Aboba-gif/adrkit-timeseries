"""Покомпонентные границы параметров обратной задачи."""
import numpy as np


def gradient_mapping(theta, gradient):
    r"""Вычислить отображение проекционного градиента.

    Parameters
    ----------
    theta, gradient : array_like
        Точка и евклидов градиент одинаковой формы; theta
        неотрицательна.

    Returns
    -------
    mapping : ndarray
        Отображение с единичным шагом той же формы.

    Notes
    -----
    Для проекции :math:`P_+` на неотрицательный ортант:

    .. math::

        g_P(\theta)=\theta-P_+(\theta-g).

    Здесь :math:`g` — переданный gradient.
    """
    return np.asarray(theta)-np.maximum(0., np.asarray(theta)-gradient)


def box_bounds(bounds, size):
    """Подготовить покомпонентные границы для SciPy.

    Parameters
    ----------
    bounds : sequence of pairs or None
        Нижняя и верхняя границы каждой координаты; None задаёт
        отсутствие границы.
    size : int
        Число координат.

    Returns
    -------
    bounds : list of tuple
        Пары границ в порядке координат; обратные и недопустимые
        границы отклоняются.
    """
    if bounds is None:
        return [(None, None)]*size
    if len(bounds) != size:
        raise ValueError("One bound pair per coefficient required")
    result = []
    for low, high in bounds:
        low = -np.inf if low is None else float(low)
        high = np.inf if high is None else float(high)
        if np.isnan(low) or np.isnan(high) or low > high or low == np.inf or high == -np.inf:
            raise ValueError("Invalid coefficient bounds")
        result.append((low, high))
    return result
