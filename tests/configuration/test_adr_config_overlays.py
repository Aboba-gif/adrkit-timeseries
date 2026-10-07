"""Тождество полной и дополненной конфигураций; отклонение некорректных входов до выполнения действий."""
from copy import deepcopy
import json

import pytest

from adrkit.errors import ConfigError
from tests.configuration.fixtures.composition.adr_actions import assemble, main, run
from adrkit.config.composition import load


def write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def test_full_and_overlays_have_same_identity_and_outputs(tmp_path,composition_config):
    full = write(tmp_path/'full.json',composition_config)
    base = deepcopy(composition_config)
    base['science']['regularizer']['parameters']['alpha']=.1
    basefile = write(tmp_path/'base.json',base)
    overlay = write(tmp_path/'overlay.json',{'schema_version':'adrkit.experiment/1',
        'science':{'regularizer':{'kind':'h1','parameters':{'alpha':.3}}}})
    direct, merged = load([full]),load([basefile,overlay])
    assert direct.effective_science_sha256 == merged.effective_science_sha256
    assert direct.to_dict() == merged.to_dict()
    assert direct.provenance.to_dict() != merged.provenance.to_dict()
    assert run(direct,mode='smoke') == run(merged,mode='run')
    dest = tmp_path/'results'
    assert main(['--mode','run','--config',basefile,overlay,'--output-root',str(dest)])==0
    saved = json.loads((dest/'result.json').read_text())
    assert saved['result'] == run(direct,mode='run')['result']
    with pytest.raises(ConfigError,match='new directory'):
        run(direct,mode='run',output_root=dest)


@pytest.mark.parametrize('override', [
    '/science/source/parameters/time_unit=""', '/science/source/parameters/scale=true',
    '/science/source/parameters/expression="import os"', '/science/surrogate/kind="auto_import"',
    '/science/regularizer={"kind":"h1","parameters":{"alpha":1}}',
    '/science/regularizer/parameters/tau=0', '/science/optimizer/maxiter=2.5',
    '/science/covariance=[[1,2,0],[2,1,0],[0,0,1]]', '/science/target=[1,2]',
    '/science/bounds=[[2,1],[0,1]]', '/science/sensitivity/max_jvp_calls=0',
    '/execution/workers=2', '/science/unknown=0',
])
def test_invalid_config_fails_before_run(tmp_path,composition_config,override):
    path = write(tmp_path/'full.json',composition_config)
    with pytest.raises((ValueError,TypeError)):
        assemble(load([path],overrides=[override]))


@pytest.mark.parametrize('bad', ['unknown','type','kind_change'])
def test_invalid_prefix_cannot_be_repaired_by_later_overlay(tmp_path,composition_config,bad):
    first = deepcopy(composition_config)
    if bad=='unknown':
        first['science']['source']['parameters']['extra']=1
        repair = {'science':{'source':composition_config['science']['source']}}
    elif bad=='type':
        first['science']['source']['parameters']['scale']='two'
        repair = {'science':{'source':{'kind':'p1','parameters':{'scale':2}}}}
    else:
        repair = {'science':{'regularizer':{'kind':'quadratic','parameters':{'alpha':.3}}}}
    repair['schema_version']='adrkit.experiment/1'
    with pytest.raises(ConfigError):
        load([write(tmp_path/'bad.json',first),write(tmp_path/'repair.json',repair)])


@pytest.mark.parametrize('text', ['{"pipeline":"composition","pipeline":"imputation"}',
    '{"science":{"initial":[NaN]}}', '{"science":{"initial":[Infinity]}}'])
def test_non_strict_json_is_rejected(tmp_path,text):
    path=tmp_path/'bad.json'
    path.write_text(text)
    with pytest.raises(ConfigError):
        load([str(path)])
