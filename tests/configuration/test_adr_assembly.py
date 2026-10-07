"""Сборка ADR, внешние семейства, производные и отдельные действия."""

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

from adrkit.config.composition import Family, load
from tests.configuration.fixtures.composition.adr_actions import assemble, main, run

from adrkit.inverse.regularization import p1_matrices

HERE = Path(__file__).resolve().parent / "fixtures/composition"
CONFIGS = HERE / "configs"
ADAPTER_FILES = {'source': 'bernstein_source', 'observation': 'pair_average_observation', 'regularizer': 'quartic_penalty', 'surrogate': 'tanh_prediction'}

def extension(name):
    spec = importlib.util.spec_from_file_location("example_" + name, HERE / (ADAPTER_FILES[name] + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def config(name=None, overrides=(), inputs=None):
    files = [str(CONFIGS / "full.json")]
    if name:
        files.append(str(CONFIGS / (name + ".json")))
    return load(files, overrides=overrides, inputs=inputs)

def test_full_overlay_and_independent_solution():
    full = config()
    merged = load([str(CONFIGS / (n + ".json")) for n in ("base", "overlay")])
    assert full.to_dict() == merged.to_dict()
    assert full.effective_science_sha256 == merged.effective_science_sha256
    a, b = run(full, mode="smoke"), run(merged, mode="run")
    assert a == b
    assert full.provenance.to_dict() != merged.provenance.to_dict()
    # Базисные функции, матрицы массы и жёсткости собраны вручную, без конструкторов библиотеки.
    A = np.array([[1,0,0],[.5,.5,0],[0,1,0],[0,.5,.5],[0,0,1]])
    M = np.array([[2,1,0],[1,4,1],[0,1,2]]) / 6
    K = np.array([[1,-1,0],[-1,2,-1],[0,-1,1]])
    G = M + .25*K
    expected = np.linalg.solve(A.T@A + .1*G, A.T@np.array([1,1.5,2,1.5,1]) + .1*G@np.array([.2,.3,.2]))
    assert a['result']['optimizer']['success']
    np.testing.assert_allclose(a['result']['coefficients'], expected, rtol=1e-9, atol=1e-10)

@pytest.mark.parametrize('name', ['source', 'regularizer', 'observation', 'surrogate'])
def test_external_derivatives_and_run(name):
    module = extension(name)
    cfg = config(name, inputs=module.FAMILIES)
    assembled = assemble(cfg, inputs=module.FAMILIES)
    x, h = np.array([.4,.6,.8]), np.array([.2,-.1,.3])
    mapping = assembled.prediction
    eps = 1e-6
    numerical = (mapping.predict.predict(x+eps*h)-mapping.predict.predict(x-eps*h))/(2*eps)
    tangent = mapping.jvp.jvp(x, h)
    np.testing.assert_allclose(tangent, numerical, rtol=1e-8, atol=1e-9)
    if mapping.vjp is not None:
        z = np.arange(mapping.codomain.size, dtype=float)+.3
        np.testing.assert_allclose(z@tangent, h@mapping.vjp.vjp(x,z), rtol=1e-12, atol=1e-12)
        objective = assembled.objective
        difference = (objective.value.value(x+eps*h)-objective.value.value(x-eps*h))/(2*eps)
        np.testing.assert_allclose(h@objective.cotangent.cotangent(x), difference, rtol=1e-8, atol=1e-9)
    result = run(cfg, mode='smoke', inputs=module.FAMILIES)['result']
    assert result.get('optimizer', {}).get('success', True)
    if name == 'surrogate':
        assert result['capabilities']['vjp'] is False
    if name == 'source':
        np.testing.assert_array_equal(module.UserTruth().value([0,.5,1,1.5,2]), cfg.to_dict()['science']['target'])

def test_jvp_only_rejects_solve_before_any_prediction(monkeypatch):
    module = extension('surrogate')
    def forbidden(*args):
        raise AssertionError('prediction performed before capability rejection')
    monkeypatch.setattr(module.TanhSurrogate, 'predict', forbidden)
    cfg = config('surrogate', ['/science/action="solve"'], module.FAMILIES)
    with pytest.raises(CapabilityError, match='VJP'):
        run(cfg, mode='run', inputs=module.FAMILIES)

def test_predict_only_and_nonsmooth_are_not_adjoint_capable():
    module = extension('surrogate')
    cfg = config('surrogate', ['/science/surrogate/parameters/capabilities="predict"', '/science/action="score"'], module.FAMILIES)
    assert not run(cfg, mode='run', inputs=module.FAMILIES)['result']['capabilities']['jvp']
    with pytest.raises(CapabilityError, match='JVP'):
        run(config('surrogate', ['/science/surrogate/parameters/capabilities="predict"'], module.FAMILIES), mode='run', inputs=module.FAMILIES)
    class Absolute:
        def value(self, x):
            return float(np.sum(np.abs(x)))
    families = {'regularizer': {'absolute': Family(lambda p, partial: p,
                 lambda p,b,d: ScalarCapabilities(Absolute(), d))}}
    replacement = '/science/regularizer={"kind":"absolute","parameters":{}}'
    score = config(overrides=[replacement, '/science/action="score"'], inputs=families)
    assert run(score, mode='smoke', inputs=families)['result']['score'] == 5.25
    with pytest.raises(CapabilityError):
        assemble(config(overrides=[replacement], inputs=families), inputs=families)

@pytest.mark.parametrize('override', [
    '/science/source/parameters/scale=-1', '/science/regularizer/parameters/tau=0',
    '/science/unknown=1', '/science/optimizer/maxiter=true',
    '/science/target=[1,2]', '/science/bounds=[[2,1]]',
    '/science/covariance=[[1,2],[2,1]]',
    '/science/source={"kind":"p1","parameters":{}}',
    '/science/sensitivity/max_columns=0', '/execution/workers=2',
])
def test_invalid_before_execution(override):
    with pytest.raises((ValueError, TypeError)):
        assemble(config(overrides=[override]))

def test_sensitivity_limits_reject_without_jvp():
    with pytest.raises(ConfigError, match='limits'):
        assemble(config(overrides=['/science/action="sensitivity"', '/science/sensitivity/max_entries=1']))

def test_l2_h1_manual_integrals_nonzero_center():
    r"""Проверить штраф относительно ненулевого центра по независимым интегралам.

    Notes
    -----
    Для разности с центром :math:`d(t)=1+2t` на интервале от 0 до 2
    интегралы значений и производной равны

    .. math::

        \int_0^2 d(t)^2\,dt=\frac{62}{3},\qquad
        \int_0^2 d'(t)^2\,dt=8.
    """
    for kind, tau in [('l2',0), ('h1',.5)]:
        overrides = ['/science/regularizer='+json.dumps({'kind':kind,'parameters':{'center':[.2,.3,.2],'alpha':.1,'tau':tau}})]
        c = assemble(config(overrides=overrides))
        penalty = c.objective.value.penalty
        np.testing.assert_allclose(penalty.value.value(np.array([1.2,3.3,5.2])), .05*(62/3+tau*tau*8), rtol=1e-14)

def test_output_and_cli(tmp_path, capsys):
    dest = tmp_path/'new-output'
    assert main(['--mode','smoke','--config',str(CONFIGS/'base.json'),str(CONFIGS/'overlay.json'),
                 '--output-root',str(dest)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads((dest/'result.json').read_text()) == report
    assert json.loads((dest/'config-record.json').read_text())['provenance']['inputs']
    with pytest.raises(ConfigError, match='new directory'):
        run(config(), mode='run', output_root=dest)
