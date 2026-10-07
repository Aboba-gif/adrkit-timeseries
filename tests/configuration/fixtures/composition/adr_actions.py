"""Отдельные действия ADR для проверки конфигураций и внешних семейств."""
import argparse
import json

from adrkit.config import verify_resources
from adrkit.config.composition import assemble as assemble_components, load
from adrkit.errors import ConfigError
from adrkit.inverse.objectives import evaluate_score
from adrkit.inverse.optimizers import solve_lbfgsb
from adrkit.inverse.sensitivity import SensitivityLimits, sensitivity_spectrum
from tests.configuration.fixtures._results import output_destination, write_report


def assemble(config, *, inputs=None):
    return assemble_components(config, inputs=inputs)


def run_action(assembled):
    c, s = assembled, assembled.science
    point = c.initial
    result = {"action": s["action"], "capabilities": {
        "predict": True, "jvp": c.prediction.jvp is not None, "vjp": c.prediction.vjp is not None,
        "cotangent": c.objective.cotangent is not None}}
    if s["action"] == "solve":
        solved = solve_lbfgsb(c.objective, point, bounds=s["bounds"], options=s["optimizer"])
        point = solved.x
        result["optimizer"] = {"success": bool(solved.success), "status": int(solved.status),
                               "message": str(solved.message), "iterations": int(solved.nit),
                               "evaluations": int(solved.nfev)}
    if s["action"] == "sensitivity":
        p = s["sensitivity"]
        spectrum = sensitivity_spectrum(c.prediction, point, coefficient_metric=c.coefficient_metric,
            observation_metric=c.observation_metric, relative_threshold=p["relative_threshold"],
            limits=SensitivityLimits(**{k: p[k] for k in ("max_columns", "max_entries", "max_jvp_calls")}))
        result["sensitivity"] = {"singular_values": spectrum.singular_values.tolist(),
            "rank": spectrum.rank, "nullity": spectrum.nullity, "condition": spectrum.condition,
            "condition_reason": spectrum.condition_reason}
    result.update(coefficients=point.tolist(), score=evaluate_score(c.objective, point),
                  prediction=c.prediction.codomain.array(c.prediction.predict.predict(point)).tolist())
    return result


def run(config, *, mode, inputs=None, output_root=None):
    """Проверить сборку либо выполнить объявленное конечное действие."""
    if mode not in ("validate", "smoke", "run"):
        raise ConfigError("supported engineering modes: validate, smoke, run")
    assembled = assemble(config, inputs=inputs)
    report = {"effective_config": config.to_dict(),
              "effective_science_sha256": config.effective_science_sha256}
    if mode == "validate":
        report["validated"] = True
        return report
    destination = output_destination(config, output_root)
    verify_resources(config)
    report["result"] = run_action(assembled)
    write_report(config, report, destination)
    return report


def main(argv=None, *, inputs=None):
    """Вызвать отдельное действие ADR с явной конфигурацией."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('validate', 'smoke', 'run'))
    parser.add_argument('--config', action='append', nargs='+', required=True)
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--config-root')
    parser.add_argument('--output-root')
    args = parser.parse_args(argv)
    try:
        config = load([p for group in args.config for p in group], inputs=inputs,
                      overrides=args.overrides, config_root=args.config_root,
                      output_root=args.output_root)
        report = run(config, mode=args.mode, inputs=inputs)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if report.get('result', {}).get('optimizer', {}).get('success', True) else 1
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')
