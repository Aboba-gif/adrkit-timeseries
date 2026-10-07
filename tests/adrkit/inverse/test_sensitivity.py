"""Взвешенные сингулярные числа и привязка линеаризации к точке."""
from types import SimpleNamespace

import numpy as np
import pytest

from adrkit.contracts import MapCapabilities
from adrkit.inverse.misfit import covariance_metric
from adrkit.inverse.regularization import CoefficientMetric
from adrkit.inverse.sensitivity import SensitivityLimits, sensitivity_spectrum
from adrkit.spaces import ArraySpace, matrix_map


class NonlinearSurrogate:
    r"""Нелинейное отображение двух координат с сохранением точек производной.

    Attributes
    ----------
    calls : list of ndarray
        Копии точек, в которых запрошено действие якобиана.

    Notes
    -----
    Прямое отображение задано формулой

    .. math::

        T(x)=\begin{pmatrix}x_0^2+x_1\\3x_0-x_1^2\\x_0x_1\end{pmatrix}.
    """
    def __init__(self):
        self.calls = []

    def predict(self, x):
        return np.array([x[0]**2+x[1], 3*x[0]-x[1]**2, x[0]*x[1]])

    def jvp(self, at, h):
        self.calls.append(at.copy())
        return np.array([[2*at[0],1],[3,-2*at[1]],[at[1],at[0]]])@h


class FixedWhitening:
    r"""Преобразование строк по фиксированной ковариационной матрице.

    Parameters
    ----------
    covariance : array_like of shape (n, n)
        Симметричная положительно определённая ковариационная матрица.
    layout : ArraySpace
        Пространство строк преобразуемой матрицы.

    Notes
    -----
    При разложении Холецкого :math:`\Sigma=LL^\mathsf{T}` метод
    ``whiten_matrix`` возвращает :math:`L^{-1}B`.
    """
    def __init__(self, covariance, layout):
        self.layout = layout
        self.factor = np.linalg.cholesky(covariance)

    def whiten_matrix(self, matrix):
        return np.linalg.solve(self.factor, matrix)


class ForeignHashWhitening(FixedWhitening):
    @property
    def whitening_hash(self):
        raise AssertionError('spectrum read an unrelated whitening identifier')


def setup():
    d = ArraySpace((2,), axes=('source',), coordinates=(('left','right'),), units='kg/s', binding='source-1')
    y = ArraySpace((3,), axes=('observation',), coordinates=(('B','C','A'),), units='ug/m3', binding='rows-1')
    op = NonlinearSurrogate()
    caps = MapCapabilities(op,d,y,op)
    G = np.array([[3.,.7],[.7,2.]])
    sigma = np.array([[2.,.3,0],[.3,1.,.1],[0,.1,4.]])
    return caps, CoefficientMetric(G,domain=d), covariance_metric(sigma,layout=y), G, sigma


@pytest.mark.parametrize('whitener_type', [None, FixedWhitening, ForeignHashWhitening],
                         ids=['covariance', 'external', 'foreign_hash'])
def test_whitened_svd_matches_independent_dense_algebra_and_owned_point(whitener_type):
    caps, metric, noise, G, sigma = setup()
    if whitener_type is not None:
        noise = whitener_type(sigma, caps.codomain)
    point = np.array([.6,-.2])
    original = point.copy()
    result = sensitivity_spectrum(caps,point,coefficient_metric=metric,observation_metric=noise,
                                  limits=SensitivityLimits(2,6,2))
    J = np.array([[1.2,1],[3,.4],[-.2,.6]])
    expected = np.linalg.svd(np.linalg.solve(np.linalg.cholesky(sigma),J) @
                             np.linalg.inv(np.linalg.cholesky(G).T),compute_uv=False)
    np.testing.assert_allclose(result.singular_values,expected,rtol=2e-14,atol=1e-14)
    assert result.rank == 2 and result.nullity == 0
    np.testing.assert_allclose(result.threshold, 1e-10*expected[0], rtol=2e-14, atol=0.)
    np.testing.assert_allclose(result.condition, expected[0]/expected[1], rtol=2e-14, atol=0.)
    assert result.condition_reason is None
    assert result.coefficient_space is caps.domain
    assert result.observation_space is caps.codomain
    assert not np.shares_memory(result.at, point)
    with pytest.raises(ValueError):
        result.at[0] = 99
    with pytest.raises(ValueError):
        result.singular_values[0] = 99
    point[:] = 99
    np.testing.assert_array_equal(result.at,original)
    for at in caps.jvp.calls:
        np.testing.assert_array_equal(at,original)
    changed = np.nextafter(original,np.inf)
    second = sensitivity_spectrum(caps,changed,coefficient_metric=metric,observation_metric=noise,
                                  limits=SensitivityLimits(2,6,2))
    np.testing.assert_array_equal(second.at, changed)
    assert not np.array_equal(second.at, result.at)
    for at in caps.jvp.calls[-2:]:
        np.testing.assert_array_equal(at,changed)


@pytest.mark.parametrize('change', ['rows','units','binding'])
def test_sensitivity_rejects_misbound_metric_before_jvp(change):
    caps, metric, noise, _, sigma = setup()
    y = caps.codomain
    wrong = ArraySpace(y.shape,axes=y.axes,
        coordinates=(('A','C','B'),) if change=='rows' else y.coordinates,
        units='mg/m3' if change=='units' else y.units,
        binding='rows-2' if change=='binding' else y.binding)
    with pytest.raises(ValueError,match='Incompatible'):
        sensitivity_spectrum(caps,[.6,-.2],coefficient_metric=metric,
            observation_metric=covariance_metric(sigma,layout=wrong),limits=SensitivityLimits(2,6,2))
    assert caps.jvp.calls == []


@pytest.mark.parametrize('limits', [SensitivityLimits(1,6,2),SensitivityLimits(2,5,2),SensitivityLimits(2,6,1)])
def test_limits_are_checked_before_derivatives(limits):
    caps, metric, noise, _, _ = setup()
    with pytest.raises(ValueError,match='limits'):
        sensitivity_spectrum(caps,[.6,-.2],coefficient_metric=metric,observation_metric=noise,limits=limits)
    assert caps.jvp.calls == []


@pytest.mark.parametrize('root', [
    np.eye(3), [[1., 0.], [.3, 1.]], [[1., 0.], [0., 0.]],
    [[1., 0.], [0., -1.]], [[1., np.nan], [0., 1.]],
    [[1., np.inf], [0., 1.]], np.eye(2, dtype=complex),
])
def test_invalid_coefficient_factor_rejected_before_derivatives(root):
    caps, _, noise, _, _ = setup()
    metric = SimpleNamespace(domain=caps.domain, root=root)
    with pytest.raises(ValueError, match='Coefficient whitening'):
        sensitivity_spectrum(caps, [.6, -.2], coefficient_metric=metric,
                             observation_metric=noise, limits=SensitivityLimits(2,6,2))
    assert caps.jvp.calls == []


def test_zero_spectrum_preserves_unresolved_condition():
    caps, metric, noise, _, _ = setup()
    zero = matrix_map(np.zeros((3,2)), domain=caps.domain, codomain=caps.codomain)
    result = sensitivity_spectrum(zero, [.6, -.2], coefficient_metric=metric,
                                  observation_metric=noise, limits=SensitivityLimits(2,6,2))
    np.testing.assert_array_equal(result.singular_values, np.zeros(2))
    assert result.rank == 0 and result.nullity == 2 and result.threshold == 0.
    assert result.condition is None and result.condition_reason == 'zero resolved rank'


def test_explicit_unit_change_and_row_permutation_preserve_metric_spectrum():
    caps, metric, noise, G, sigma = setup()
    point = np.array([.6,-.2])
    limits = SensitivityLimits(2,6,2)
    first = sensitivity_spectrum(caps,point,coefficient_metric=metric,observation_metric=noise,limits=limits)
    # Коэффициенты выражаются в g/s вместо kg/s, наблюдения — в mg/m3. Неявных преобразований нет:
    # J, обе метрики, точка и идентификаторы строк преобразуются вместе.
    input_scale, output_scale = 1000., .001
    order = [2,0,1]
    d = ArraySpace((2,),axes=caps.domain.axes,coordinates=caps.domain.coordinates,units='g/s',binding='source-2')
    y = ArraySpace((3,),axes=caps.codomain.axes,coordinates=(('A','B','C'),),units='mg/m3',binding='rows-2')
    class Converted:
        def predict(self,x):
            return output_scale*caps.predict.predict(x/input_scale)[order]

        def jvp(self,at,h):
            return output_scale*caps.jvp.jvp(at/input_scale,h/input_scale)[order]
    op = Converted()
    converted = sensitivity_spectrum(MapCapabilities(op,d,y,op),input_scale*point,
        coefficient_metric=CoefficientMetric(G/input_scale**2,domain=d),
        observation_metric=covariance_metric(output_scale**2*sigma[np.ix_(order,order)],layout=y),limits=limits)
    np.testing.assert_allclose(converted.singular_values,first.singular_values,rtol=3e-14,atol=1e-14)
    assert converted.rank==first.rank and converted.nullity==first.nullity
    assert converted.coefficient_space is d and converted.observation_space is y
    assert converted.coefficient_space != first.coefficient_space
    assert converted.observation_space != first.observation_space
    np.testing.assert_array_equal(converted.at, input_scale*point)
