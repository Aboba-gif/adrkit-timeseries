"""Проверка JSON-параметров семейств составной задачи."""
from adrkit.errors import ConfigError

from .loader import ScienceSchema
from .validation import JSONRecord, closed_object, integer, number, string

__all__ = ["fields", "vector", "matrix", "composition_schema"]


def fields(value, names, *, partial, where="parameters"):
    """Проверить поля объекта параметров семейства.

    Parameters
    ----------
    value : dict
        Объект со строковыми ключами.
    names : collection of str
        Разрешённые поля; при полной проверке все обязательны.
    partial : bool
        Допустить отсутствие разрешённых полей.
    where : str, optional
        Имя объекта в сообщении об ошибке.

    Returns
    -------
    parameters : dict
        Исходный проверенный словарь.
    """
    return closed_object(value, set(names), set() if partial else set(names), where=where)


def vector(value):
    """Проверить числовой вектор JSON.

    Parameters
    ----------
    value : list
        Непустой список конечных чисел без bool.

    Returns
    -------
    vector : list of float
        Значения в исходном порядке.
    """
    if type(value) is not list or not value:
        raise ConfigError("expected a nonempty numeric array")
    return [number(v) for v in value]


def matrix(value):
    """Проверить прямоугольную числовую матрицу JSON.

    Parameters
    ----------
    value : list of lists
        Непустые строки одинаковой длины с конечными числами.

    Returns
    -------
    matrix : list of lists of float
        Значения в исходном порядке строк и столбцов.
    """
    if type(value) is not list or not value:
        raise ConfigError("expected a nonempty numeric matrix")
    rows = [vector(row) for row in value]
    if len({len(row) for row in rows}) != 1:
        raise ConfigError("matrix rows must have equal lengths")
    return rows


def _p1_parameters(p, *, partial):
    fields(p, {"knots", "times", "time_unit", "coefficient_unit", "value_unit", "scale"}, partial=partial)
    p = dict(p)
    for key in ("knots", "times"):
        if key in p:
            p[key] = vector(p[key])
            if any(a >= b for a, b in zip(p[key], p[key][1:])):
                raise ConfigError(f"{key} must be strictly increasing")
    if "knots" in p and len(p["knots"]) < 2:
        raise ConfigError("P1 requires at least two knots")
    for key in ("time_unit", "coefficient_unit", "value_unit"):
        if key in p:
            string(p[key], where=key)
    if "scale" in p:
        p["scale"] = number(p["scale"], positive=True)
    if "knots" in p and "times" in p and (p["times"][0] < p["knots"][0] or p["times"][-1] > p["knots"][-1]):
        raise ConfigError("sample times must lie within the source interval")
    return p


def _empty(p, *, partial):
    return dict(fields(p, set(), partial=partial))


def _selection_parameters(p, *, partial):
    fields(p, {"indices"}, partial=partial)
    if "indices" in p:
        if type(p["indices"]) is not list or not p["indices"]:
            raise ConfigError("indices require a nonempty list")
        for i in p["indices"]:
            integer(i, minimum=0)
        if len(set(p["indices"])) != len(p["indices"]):
            raise ConfigError("selection indices must be unique")
    return dict(p)


def _penalty_parameters(p, *, partial):
    fields(p, {"center", "alpha", "tau"}, partial=partial)
    p = dict(p)
    if "center" in p:
        p["center"] = vector(p["center"])
    for key in ("alpha", "tau"):
        if key in p:
            p[key] = number(p[key], minimum=0)
    return p


def _l2_parameters(p, *, partial):
    p = _penalty_parameters(p, partial=partial)
    if p.get("tau", 0) != 0:
        raise ConfigError("L2 requires tau=0")
    return p


def _h1_parameters(p, *, partial):
    p = _penalty_parameters(p, partial=partial)
    if "tau" in p and p["tau"] <= 0:
        raise ConfigError("full temporal H1 requires tau>0")
    return p


def _quadratic_parameters(p, *, partial):
    fields(p, {"center", "alpha", "gram"}, partial=partial)
    p = dict(p)
    if "center" in p:
        p["center"] = vector(p["center"])
    if "alpha" in p:
        p["alpha"] = number(p["alpha"], minimum=0)
    if "gram" in p:
        p["gram"] = matrix(p["gram"])
    return p


def composition_schema(*, families):
    """Создать схему составной задачи из семейств компонентов.

    Parameters
    ----------
    families : Mapping
        Таблица ролей и семейств Family с операциями validate и
        build.

    Returns
    -------
    schema : ScienceSchema
        Схема adrkit.composition/1; нормализация вызывает validate,
        но не строит компоненты.
    """
    available = {role: dict(choices) for role, choices in families.items()}

    def normalize(science, context):
        def partial(path):
            return context.partial and not any(path == p or path.startswith(p + "/") for p in context.complete_paths)

        keys = {*available, "target", "covariance", "coefficient_metric", "initial", "bounds", "action", "optimizer", "sensitivity"}
        fields(science, keys, partial=partial("/science"), where="science")
        s = JSONRecord(science).to_dict()
        for role, choices in available.items():
            if role not in s:
                continue
            path = "/science/" + role
            spec = fields(s[role], {"kind", "parameters"}, partial=partial(path))
            if "kind" not in spec:
                if spec:
                    raise ConfigError(f"{role} parameters require kind")
                continue
            if type(spec["kind"]) is not str or spec["kind"] not in choices:
                raise ConfigError(f"unsupported {role} kind: {spec['kind']!r}")
            if "parameters" in spec:
                spec["parameters"] = choices[spec["kind"]].validate(spec["parameters"], partial=partial(path + "/parameters"))
        for key in ("initial", "target"):
            if key in s:
                s[key] = vector(s[key])
        for key in ("covariance", "coefficient_metric"):
            if key in s:
                s[key] = matrix(s[key])
        if "bounds" in s and s["bounds"] is not None:
            if type(s["bounds"]) is not list or not s["bounds"]:
                raise ConfigError("bounds require null or a list of pairs")
            for pair in s["bounds"]:
                if type(pair) is not list or len(pair) != 2:
                    raise ConfigError("bounds require pairs")
                for value in pair:
                    if value is not None:
                        number(value)
                if None not in pair and pair[0] > pair[1]:
                    raise ConfigError("inverted bounds")
        if "action" in s and s["action"] not in ("score", "solve", "sensitivity"):
            raise ConfigError("action must be score, solve or sensitivity")
        if "optimizer" in s:
            options = fields(s["optimizer"], {"maxiter", "maxfun", "ftol", "gtol"}, partial=partial("/science/optimizer"))
            for key, value in options.items():
                options[key] = integer(value, minimum=1) if key in ("maxiter", "maxfun") else number(value, positive=True)
        if "sensitivity" in s:
            limits = fields(s["sensitivity"], {"max_columns", "max_entries", "max_jvp_calls", "relative_threshold"}, partial=partial("/science/sensitivity"))
            for key, value in limits.items():
                limits[key] = number(value, minimum=0) if key == "relative_threshold" else integer(value, minimum=1)
        return s

    return ScienceSchema("composition", "adrkit.composition/1", normalize, lambda s: ())
