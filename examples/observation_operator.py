"""Парное усреднение с прямой и сопряжённой производными."""
import numpy as np

from adrkit.maps import map_from_functions
from adrkit.spaces import ArraySpace


def main():
    """Показать парное усреднение и перенос ковектора."""
    pairs = np.array([[0, 1], [1, 3], [3, 4]])
    domain = ArraySpace((5,), axes=("entry",), coordinates=((0, 1, 2, 3, 4),), units="1")
    codomain = ArraySpace((3,), axes=("pair_row",), coordinates=((0, 1, 2),), units="1")

    def average(x):
        return np.mean(x[pairs], axis=1)

    def transpose(cotangent):
        result = np.zeros(domain.size)
        np.add.at(result, pairs.ravel(), np.repeat(cotangent/2, 2))
        return result

    mapping = map_from_functions(domain=domain, codomain=codomain, predict=average,
        jvp=lambda at, direction: average(direction), vjp=lambda at, cotangent: transpose(cotangent))
    point = np.arange(5, dtype=float) + .3
    direction = np.array([.2, -.1, .3, .4, -.2])
    covector = np.array([.3, -.2, .8])
    np.testing.assert_allclose(covector @ mapping.jvp.jvp(point, direction),
                              direction @ mapping.vjp.vjp(point, covector), rtol=1e-14)
    print(mapping.predict.predict(point))


if __name__ == "__main__":
    main()
