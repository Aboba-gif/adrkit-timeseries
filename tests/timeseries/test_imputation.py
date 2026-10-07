"""Внешний метод заполнения передаётся как явное семейство без доступа к контрольным значениям."""
import numpy as np
import pytest

from timeseries.contracts import DataContractError, MaskedInputs, closed, conceal, result_from_inputs
from timeseries.data import build_data
from timeseries.imputation import ImputerFamily, build_imputer


class VisibleMean:
    """Заполнение пропусков средним видимых значений каждого ряда.

    Parameters
    ----------
    parameters : dict
        Словарь с `offset` — численной добавкой к среднему. Она
        применяется к каждому каналу в его численных единицах.
    """
    def __init__(self, parameters):
        self.offset = parameters['offset']

    def predict(self, inputs, *, fitted):
        """Заполнить невидимые ячейки средним видимых значений по времени.

        Parameters
        ----------
        inputs : MaskedInputs
            Панель формы (n_times, n_sites, n_channels) с видимыми значениями
            в единицах каналов и ``NaN`` во всех невидимых позициях.
        fitted : None
            Отсутствие состояния обучения для этого тестового метода.

        Returns
        -------
        ImputationResult
            Панель той же формы и привязки. Видимые значения сохраняются;
            искусственно скрытые и естественно отсутствующие значения
            заменяются средним видимых значений своего поста и канала
            по времени с добавкой `offset`. Ряд без видимых значений
            остаётся ``NaN``. Оценка неопределённости отсутствует.
        """
        assert isinstance(inputs,MaskedInputs) and fitted is None
        assert not hasattr(inputs,'truth')
        value = np.nanmean(inputs.values,axis=0,keepdims=True)+self.offset
        prediction = np.where(inputs.visible,inputs.values,np.broadcast_to(value,inputs.values.shape))
        return result_from_inputs(inputs, prediction)


def validate(parameters):
    closed(parameters,{'offset'},where='visible_mean')
    if type(parameters['offset']) not in (int,float) or not np.isfinite(parameters['offset']):
        raise DataContractError('finite offset required')
    return dict(parameters)


def test_external_imputer_sees_only_concealed_values(csv_case):
    specs,panel_spec = csv_case(3)
    panel = build_data(specs,panel_spec=panel_spec).load().panel
    hide = np.zeros(panel.values.shape,dtype=bool)
    hide[2,0,0] = True
    inputs,truth = conceal(panel,artificial_mask=hide,mit_mask=np.zeros_like(hide),
        protocol_id='B08',population_id='heldout',population_kind='test')
    families = {'visible_mean':ImputerFamily(validate,VisibleMean)}
    spec = {'kind':'visible_mean','parameters':{'offset':.5}}
    model = build_imputer(spec,families=families)
    spec['parameters']['offset'] = 999
    result = model.predict(inputs,fitted=None)
    assert np.isnan(inputs.values[2,0,0])
    assert result.prediction[2,0,0] == 13.5  # Среднее двух видимых значений с добавкой; скрытое значение исключено.
    np.testing.assert_array_equal(result.prediction[inputs.visible],inputs.values[inputs.visible])
    assert result.uncertainty is None
    assert not hasattr(model,'fit')
    with pytest.raises(DataContractError):
        build_imputer({'kind':'visible_mean','parameters':{'offset':0,'unused':1}},families=families)


@pytest.mark.parametrize('kind', ['linear','pchip'])
def test_builtin_no_fit_preserves_visible_and_missing_semantics(csv_case,kind):
    specs,panel_spec = csv_case()
    panel = build_data(specs,panel_spec=panel_spec).load().panel
    empty = np.zeros(panel.values.shape,dtype=bool)
    inputs,_ = conceal(panel,artificial_mask=empty,mit_mask=empty,
        protocol_id='B08',population_id='export',population_kind='export')
    model = build_imputer({'kind':kind,'parameters':{'edge_policy':'abstain'}})
    result = model.predict(inputs,fitted=model.no_fit_state)
    np.testing.assert_allclose(result.prediction[:,:,0],[[10,1],[12,3],[14,5],[16,7]],rtol=0,atol=1e-14)
    with pytest.raises(DataContractError,match='state'):
        model.predict(inputs,fitted=None)
