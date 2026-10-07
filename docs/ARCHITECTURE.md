# Архитектура

`adrkit` и `timeseries` — независимые Python-пакеты одной библиотечной
поставки. `adrkit` содержит уравнение и компоненты обратных задач, `timeseries` —
данные и методы обработки временных рядов. Примеры в `examples` показывают
небольшие задачи на искусственных входах; тесты находятся в `tests`.

## Уравнение и обратная задача

| Компонент | Назначение |
|---|---|
| [ArraySpace](../src/adrkit/spaces.py) | Форма массива, порядок координат, единицы и привязка данных |
| [ADRProblem](../src/adrkit/pde.py) | Уравнение и заданные коэффициенты |
| [ScalarField, VectorField, ReactionLaw](../src/adrkit/coefficients.py) | Поля коэффициентов и локальная реакция с производной |
| [StateResidual](../src/adrkit/state_residuals.py) | Добавочный дискретный член и полный якобиан по состоянию |
| [BoundedSolver](../src/adrkit/backends/bounded.py), [CachedSolver](../src/adrkit/backends/cached.py) | Решение прямой задачи и производные по аддитивному входу |
| [sources](../src/adrkit/sources.py), [loads](../src/adrkit/loads.py) | Базис источника, средние нагрузки и известная предыстория |
| [spaces.matrix_map](../src/adrkit/spaces.py), [maps.map_from_functions](../src/adrkit/maps.py), [state_maps](../src/adrkit/state_maps.py) | Отображения из матриц, функций и внешнего решателя |
| [GridObservation](../src/adrkit/observations/grid.py), [Selection](../src/adrkit/selection.py) | Наблюдения и выбор координат |
| [GridPrediction, SelectedPrediction](../src/adrkit/predictions.py) | Сеточный прогноз концентрации и выбор координат прогноза |
| [functionals](../src/adrkit/inverse/functionals.py), [objectives](../src/adrkit/inverse/objectives.py) | Скалярные функционалы и их композиция |
| [misfit](../src/adrkit/inverse/misfit.py), [regularization](../src/adrkit/inverse/regularization.py) | Ковариационная невязка и квадратичные штрафы |
| [linalg](../src/adrkit/inverse/linalg.py) | Факторизация положительно определённых матриц, отбеливание и корень Грама |
| [projected](../src/adrkit/inverse/projected.py), [tikhonov](../src/adrkit/inverse/tikhonov.py) | Проекционная оптимизация и неотрицательная регуляризация Тихонова |
| [sensitivity](../src/adrkit/inverse/sensitivity.py) | Спектр линейной или линеаризованной задачи в заданных метриках |
| [InverseProblem](../src/adrkit/inverse/problem.py), [optimizers](../src/adrkit/inverse/optimizers.py) | Сборка обратной задачи и гладкая оптимизация |
| [config.composition](../src/adrkit/config/composition.py) | Проверка JSON и создание компонентов ADR |

Размерность массива, его координаты и единицы задаются явно.
Для градиентной оптимизации нужны прогноз и VJP — действие
транспонированной производной. Спектральный анализ чувствительности
использует JVP — действие производной на направление; VJP для этой
операции не требуется. Эти операции предоставляет автор компонента.
Метрика пространства задаётся отдельно; VJP возвращает координатный
ковектор.

Общие факторизации в `inverse.linalg` используются метриками и решателем
Тихонова. Этот модуль не зависит от оптимизации или анализа чувствительности.

Сборка компонентов не выполняет прогноз или оптимизацию. Пример полного вызова:

```text
python -B examples/configuration.py --config examples/configuration.json
```

В Python дополнительные семейства передаются через `Family` и словарь `inputs`.
JSON выбирает имена и параметры из этого словаря, без импорта пользовательского
кода. Подробные ограничения решателей и производных: [EXTENDING](EXTENDING.md).

Общий формат `adrkit.experiment/1` хранит параметры в `science` и адрес результатов
в `execution.output_root`. `ScienceSchema` задаёт проверки и ресурсы в Python,
`ResourcePath` связывает путь с контрольной суммой файла. `EffectiveConfig`
сохраняет действующие параметры и их происхождение: файлы, наложения, замены
и значения по умолчанию. Правило нормализации JSON имеет версию `adrkit-json/1`.
Чтение конфигураций, объединение параметров и проверка файлов ресурсов — отдельные
вызовы `read_configs`, `compose_config` и `verify_resources`. Они не создают
каталог результатов и не запускают решение. Относительные пути привязаны к
объявленному источнику; `$`, `~`, `[` и `]` читаются буквально.

## Временные ряды

| Компонент | Назначение |
|---|---|
| [contracts](../src/timeseries/contracts.py), [data](../src/timeseries/data.py) | Оси панели, исходные наблюдения, маски и чтение CSV |
| [features](../src/timeseries/features.py) | Признаки и объявление их зависимости от будущих данных |
| [imputation](../src/timeseries/imputation.py), [numerics](../src/timeseries/numerics.py) | Конструкторы методов и численные операции заполнения |
| [metrics](../src/timeseries/metrics.py) | Ошибки по каналам и постам |
| [pipeline](../src/timeseries/pipeline.py), [config](../src/timeseries/config.py) | Сборка обработки и явные этапы обучения, прогноза и оценки |

Чтение данных, построение метода, обучение, прогноз и запись результатов —
отдельные вызовы. `conceal` отделяет контрольные значения от видимых входов.
Методы и признаки получают только эти входы.

Пример обработки собственного состава каналов:

```text
python -B examples/time_series_pipeline.py
```

Спецификации и конструкторы описаны в [TIME_SERIES](TIME_SERIES.md).
Для другого формата данных нужен адаптер источника; для иной непрерывной модели
нужны соответствующая дискретизация, производные и математическое обоснование.
