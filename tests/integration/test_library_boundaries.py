"""Общие библиотеки не зависят от программ конкретных исследований."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'src'


@pytest.mark.parametrize('package,forbidden', [
    ('adrkit', {'timeseries', 'experiments', 'examples', 'tests'}),
    ('timeseries', {'adrkit', 'experiments', 'examples', 'tests'}),
])
def test_generic_package_imports_independently(package, forbidden):
    forbidden = set(forbidden) | {"experiments"}
    modules = []
    for path in sorted((SOURCE / package).rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split('.')[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                roots = {node.module.split('.')[0]}
            else:
                continue
            assert not roots.intersection(forbidden), path
        parts = path.relative_to(SOURCE).with_suffix('').parts
        modules.append('.'.join(parts[:-1] if parts[-1] == '__init__' else parts))
    # Отдельный интерпретатор проверяет зависимости, возникающие при импорте.
    script = ('import importlib,sys; '
              f'[importlib.import_module(name) for name in {modules!r}]; '
              f'assert not {{name.split(".")[0] for name in sys.modules}}.intersection({forbidden!r})')
    result = subprocess.run([sys.executable, '-B', '-c', script],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
