"""Нелинейное отображение tanh с явно объявленными производными."""
import numpy as np

from adrkit.maps import map_from_functions
from adrkit.spaces import ArraySpace


def main():
    """Показать прогноз tanh и действие его производной."""
    layout = ArraySpace((3,), axes=("entry",), coordinates=((0, 1, 2),), units="1")
    gain = .5
    mapping = map_from_functions(domain=layout, codomain=layout,
        predict=lambda x: np.tanh(gain*x),
        jvp=lambda at, h: gain*(1-np.tanh(gain*at)**2)*h,
        vjp=lambda at, z: gain*(1-np.tanh(gain*at)**2)*z)
    point, direction = np.array([.4, .6, .8]), np.array([.2, -.1, .3])
    step = 1e-6
    numerical = (mapping.predict.predict(point+step*direction)
                 - mapping.predict.predict(point-step*direction))/(2*step)
    np.testing.assert_allclose(mapping.jvp.jvp(point, direction), numerical, rtol=1e-8, atol=1e-9)
    print(mapping.predict.predict(point))


if __name__ == "__main__":
    main()
