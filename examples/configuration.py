"""Собрать и решить малую обратную задачу из явного JSON-файла."""
import argparse
from pathlib import Path

from adrkit.config.composition import assemble, load
from adrkit.errors import ConfigError
from adrkit.inverse.optimizers import solve_lbfgsb


def main(argv=None):
    """Решить малую обратную задачу из JSON-конфигурации.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы командной строки; None использует sys.argv.

    Returns
    -------
    status : int
        Ноль после успешного решения и вывода коэффициентов.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON-файл составной задачи")
    args = parser.parse_args(argv)
    config = load([Path(args.config).resolve()])
    if config.to_dict()["execution"]["output_root"] is not None:
        raise ConfigError("пример выводит результат на экран и не записывает файлы")
    components = assemble(config)
    if components.science["action"] != "solve":
        raise ConfigError("пример выполняет только действие solve")
    result = solve_lbfgsb(components.objective, components.initial,
        bounds=components.science["bounds"], options=components.science["optimizer"])
    if not result.success:
        raise RuntimeError(result.message)
    print("Коэффициенты:", result.x)
    print("Значение функционала:", result.fun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
