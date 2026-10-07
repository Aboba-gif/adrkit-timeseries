"""Интерполяция без обучения и проверка исходных байтов."""
import copy

import hashlib

import importlib.util

import json

from pathlib import Path

import numpy as np

import pytest

from adrkit.config import JSONRecord

from adrkit.contracts import ScalarCapabilities

from adrkit.errors import CapabilityError, ConfigError

from tests.configuration.fixtures.imputation import load, run

from adrkit.inverse.regularization import p1_matrices

def data_config(tmp_path):
    raw = b'time,value\n2020-01-01 00:00:00,1\n2020-01-01 01:00:00,\n2020-01-01 02:00:00,3\n'
    path = tmp_path / 'tiny.csv'
    path.write_bytes(raw)
    panel = dict(start='2020-01-01 00:00:00', stop='2020-01-01 03:00:00', frequency='1h',
        origin='2020-01-01 00:00:00', clock='naive', timezone=None, calendar='proleptic_gregorian',
        sites=['toy'], channels=['x'], units={'x':'1'}, duplicates='reject')
    source = dict(id='tiny',path='tiny.csv',sha256=hashlib.sha256(raw).hexdigest(),
        format={'kind':'csv','parameters':{'separator':',','encoding':'utf-8'}},
        columns={'time':['time'],'values':{'x':'value'}},
        time={'parser':'format','format':'%Y-%m-%d %H:%M:%S','dst_policy':'fail',
              **{k:panel[k] for k in ('timezone','calendar','clock','frequency','origin')}},
        site={'kind':'fixed','id':'toy'},units={'x':'1'},missing_values=[''],route='strict')
    cfg = {'schema_version':'adrkit.experiment/1','pipeline':'imputation','science_schema_version':'adrkit.imputation/1',
        'science':{'data':{'sources':[source],'panel':panel},'imputer':{'kind':'linear','parameters':{'edge_policy':'nearest'}},
                   'protocol_id':'engineering-tiny','population_id':'export-tiny'}}
    target = tmp_path/'data.json'
    target.write_text(json.dumps(cfg))
    return load([str(target)])

@pytest.mark.parametrize('kind', ['linear', 'pchip'])
def test_imputation_caller_and_hash(tmp_path, kind):
    cfg = data_config(tmp_path)
    if kind == 'pchip':
        cfg = load([str(tmp_path/'data.json')], overrides=['/science/imputer={"kind":"pchip","parameters":{"edge_policy":"abstain"}}'])
    result = run(cfg, mode='smoke')['result']
    np.testing.assert_array_equal(result['prediction'], [[[1]],[[2]],[[3]]])
    assert result['fit'] is False
    (tmp_path/'tiny.csv').write_text('changed bytes')
    with pytest.raises(ConfigError, match='mismatch'):
        run(cfg, mode='smoke')

def test_validation_before_data_io_or_outputs(tmp_path):
    cfg = data_config(tmp_path)
    (tmp_path/'tiny.csv').unlink()
    assert run(cfg, mode='validate', output_root=tmp_path/'unused')['validated']
    assert not (tmp_path/'unused').exists()
    with pytest.raises(ValueError, match='imputer'):
        cfg = load([str(tmp_path/'data.json')], overrides=['/science/imputer={"kind":"trained","parameters":{}}'])
        run(cfg, mode='run')
