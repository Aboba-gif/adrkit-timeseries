"""Сумма, масштабирование и перенос скалярного функционала через отображение."""
import numpy as np

from adrkit.inverse.functionals import pullback_functional, scalar_from_functions, scale_functional, sum_functionals
from adrkit.maps import map_from_functions
from adrkit.spaces import ArraySpace


def main():
    """Показать композицию, сумму и масштабирование функционалов."""
    domain = ArraySpace((2, 2), axes=("axis-0", "axis-1"),
        coordinates=((0, 1), (0, 1)), units="1", binding="fixed")
    quartic = scalar_from_functions(domain=domain,
        value=lambda x: np.sum((x-.3)**4)/4, cotangent=lambda x: (x-.3)**3)
    exponential = map_from_functions(domain=domain, codomain=domain, predict=np.exp,
        jvp=lambda at, h: np.exp(at)*h, vjp=lambda at, z: np.exp(at)*z)
    pulled = pullback_functional(quartic, exponential)
    point = np.array([[.2, -.1], [.3, .7]])
    direction = np.array([[.5, .4], [-.3, .1]])
    expected_covector = np.exp(point)*(np.exp(point)-.3)**3
    np.testing.assert_allclose(pulled.cotangent.cotangent(point), expected_covector, rtol=1e-14)
    step = 1e-5
    numerical = (pulled.value.value(point+step*direction)-pulled.value.value(point-step*direction))/(2*step)
    np.testing.assert_allclose(numerical, np.sum(expected_covector*direction), rtol=1e-9, atol=1e-10)
    center = np.array([[.4, -.5], [.1, .2]])
    quadratic = scalar_from_functions(domain=domain,
        value=lambda x: np.sum((x-center)**2)/2, cotangent=lambda x: x-center)
    functional = sum_functionals(scale_functional(quadratic, 2.5), scale_functional(quartic, -.7))
    point = np.array([[.1, .8], [-.4, .7]])
    np.testing.assert_allclose(functional.value.value(point),
        1.25*np.sum((point-center)**2)-.175*np.sum((point-.3)**4), rtol=1e-14)
    np.testing.assert_allclose(functional.cotangent.cotangent(point),
        2.5*(point-center)-.7*(point-.3)**3, rtol=1e-14)
    print(functional.value.value(point))


if __name__ == "__main__":
    main()
