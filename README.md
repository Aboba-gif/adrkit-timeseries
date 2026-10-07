# adrkit-timeseries

Библиотека численных задач и обработки временных рядов. Python-пакет `adrkit`
содержит уравнение переноса, диффузии и реакции, решатели и компоненты обратных
задач. Пакет `timeseries` содержит чтение данных, признаки, методы заполнения
пропусков и метрики. Их можно использовать независимо.

## Установка

Нужен Python ≥3.12. Из папки библиотеки в PowerShell:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -c constraints-tested.txt "."
. .venv/Scripts/Activate.ps1
```

В Linux и WSL используйте `python3` и `.venv/bin/python`, затем активируйте
окружение командой `source .venv/bin/activate`.
`constraints-tested.txt` задаёт версии зависимостей для Python 3.12.14 в Windows.
Для разработки и запуска тестов вместо `"."` установите `".[test]"`.
Команды ниже используют `python` из выбранного окружения.

## Организация кода

| Каталог | Содержимое |
|---|---|
| `src/adrkit/` | Уравнение, коэффициенты, решатели, наблюдения, функционалы и оптимизация |
| `src/timeseries/` | Данные, признаки, заполнение пропусков, метрики и обработка |
| `examples/` | Короткие примеры на искусственных данных |
| `tests/` | Тесты общего API и малых численных задач |
| `docs/` | Устройство компонентов и руководства |

Коэффициенты, источники, отображения, функционалы и методы обработки задаются
отдельными компонентами. Чтение входов, сборка задачи, вычисление и запись
результатов выполняются отдельными вызовами. В устанавливаемый пакет входят
`adrkit` и `timeseries`; примеры и тесты остаются в исходной поставке.

Описание интерфейсов: [архитектура](docs/ARCHITECTURE.md),
[расширение ADR](docs/EXTENDING.md), [временные ряды](docs/TIME_SERIES.md).

## Примеры

| Файл | Операция |
|---|---|
| [prescribed_coefficients.py](examples/prescribed_coefficients.py) | Задать коэффициенты и локальную реакцию, решить прямую задачу |
| [source_basis.py](examples/source_basis.py) | Использовать собственный базис источника |
| [observation_operator.py](examples/observation_operator.py) | Задать усреднение и его сопряжённую производную |
| [nonlinear_map.py](examples/nonlinear_map.py) | Создать нелинейное отображение с явными производными |
| [composed_regularization.py](examples/composed_regularization.py) | Составить скалярный функционал из нескольких членов |
| [custom_inverse_problem.py](examples/custom_inverse_problem.py) | Восстановить параметры двух линейных зависимостей |
| [time_series_pipeline.py](examples/time_series_pipeline.py) | Построить признаки, заполнить пропуски и вычислить метрики |
| [configuration.py](examples/configuration.py) | Собрать и решить малую задачу из JSON |

```text
python -B examples/prescribed_coefficients.py
python -B examples/time_series_pipeline.py
python -B examples/configuration.py --config examples/configuration.json
```

Остальные примеры запускаются без аргументов таким же способом. Они используют
искусственные данные и печатают результат. Для запуска примера конфигурации из
другой рабочей папки передайте абсолютные пути к скрипту и JSON.

## Тесты

```text
python -X utf8 -B -m pytest -q
```

Код распространяется по [MIT-лицензии](LICENSE).
