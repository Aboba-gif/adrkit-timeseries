"""Неподдерживаемые операции явно отклоняются до численных вычислений."""
import numpy as np
import pytest

from adrkit.contracts import MapCapabilities, ScalarCapabilities
from adrkit.errors import CapabilityError
from adrkit.inverse.misfit import working_quadratic, working_precision
from adrkit.inverse.objectives import compose_maps, make_objective
from adrkit.inverse.optimizers import solve_lbfgsb
from adrkit.inverse.regularization import CoefficientMetric
from adrkit.inverse.sensitivity import SensitivityLimits, sensitivity_spectrum
from adrkit.spaces import ArraySpace


def layout(**changes):
    options = dict(axes=('row',), coordinates=(('A','B'),), units='kg', binding='fixed')
    options.update(changes)
    return ArraySpace((2,), **options)


class ForbiddenPredict:
    def predict(self, x):
        raise AssertionError('preflight must not predict')


class JVPOnly(ForbiddenPredict):
    def jvp(self, at, h):
        return 2*at*h


def test_predict_only_rejects_sensitivity_and_smooth_optimizer():
    domain = layout()
    caps = MapCapabilities(ForbiddenPredict(), domain, domain)
    misfit = working_quadratic([0,0], [1,2], layout=domain)
    objective = make_objective(caps, None, misfit)
    assert objective.cotangent is None
    with pytest.raises(CapabilityError, match='smooth cotangent'):
        solve_lbfgsb(objective, [1,2])
    with pytest.raises(CapabilityError, match='JVP'):
        sensitivity_spectrum(caps, [1,2], coefficient_metric=CoefficientMetric(np.eye(2), domain=domain),
            observation_metric=misfit.metric, limits=SensitivityLimits(2,4,2))


def test_jvp_only_surrogate_has_spectrum_without_adjoint_or_predict():
    domain = layout()
    op = JVPOnly()
    caps = MapCapabilities(op, domain, domain, op)
    misfit = working_quadratic([0,0], [1,1], layout=domain)
    result = sensitivity_spectrum(caps, [1,3], coefficient_metric=CoefficientMetric(np.eye(2), domain=domain),
        observation_metric=misfit.metric, limits=SensitivityLimits(2,4,2))
    np.testing.assert_array_equal(result.singular_values, [6.,2.])
    assert make_objective(caps, None, misfit).cotangent is None


@pytest.mark.parametrize('changes', [dict(units='g'), dict(scale=1000),
    dict(coordinates=(('B','A'),)), dict(binding='different'), dict(dtype=np.float32)])
def test_same_shape_different_identity_rejected_before_predict(changes):
    left, right = layout(), layout(**changes)
    with pytest.raises(ValueError, match='Incompatible'):
        compose_maps(MapCapabilities(ForbiddenPredict(), left, left),
                     MapCapabilities(ForbiddenPredict(), right, right))


def test_precision_only_cannot_pretend_to_whiten():
    domain = layout()
    op = JVPOnly()
    with pytest.raises(CapabilityError, match='whitening'):
        sensitivity_spectrum(MapCapabilities(op,domain,domain,op), [1,2],
            coefficient_metric=CoefficientMetric(np.eye(2),domain=domain),
            observation_metric=working_precision(lambda r:r,layout=domain),
            limits=SensitivityLimits(2,4,2))


def test_nonsmooth_scalar_cannot_be_used_by_smooth_optimizer():
    class Absolute:
        def value(self, x):
            raise AssertionError('must reject before value')
    with pytest.raises(CapabilityError, match='cotangent'):
        solve_lbfgsb(ScalarCapabilities(Absolute(),layout()), [1,2])
