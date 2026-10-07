"""Значения нескольких CSV, маски, точное происхождение строк и проверка их идентификаторов."""
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from timeseries.contracts import DataContractError, DuplicateObservationError, SourceIntegrityError
from timeseries.data import build_data


@pytest.mark.parametrize('shards', [1,3,7])
@pytest.mark.parametrize('reverse', [False,True])
def test_csv_partition_order_does_not_change_panel(csv_case, shards, reverse):
    specs, panel = csv_case(shards)
    if reverse:
        specs.reverse()
    result = build_data(specs,panel_spec=panel).load()
    expected = np.array([[10,1],[np.nan,np.nan],[14,5],[16,7.]])[:,:,None]
    np.testing.assert_array_equal(result.panel.values,expected)
    np.testing.assert_array_equal(result.panel.original_observed,np.isfinite(expected))
    assert not result.panel.row_present[1,0,0]  # Строка отсутствует.
    assert result.panel.raw_missing[1,1,0]       # Строка присутствует с пустым значением.
    assert len(result.panel.row_origins) == 7
    by_id = {s['id']:s for s in specs}
    for entry in result.panel.row_origins:
        source = by_id[entry.row.source_id]
        with Path(source['path']).open(newline='') as stream:
            raw = list(csv.reader(stream))[entry.row.row_number-1]
        assert raw == [entry.row.raw_time,entry.row.site,entry.row.raw_value]
        assert entry.row.sha256 == source['sha256']


def test_inert_data_builder_then_exact_bytes_validation(csv_case):
    specs, panel = csv_case()
    assembly = build_data(specs,panel_spec=panel)
    specs[0]['units']['x'] = 'changed'
    path = Path(assembly.specs[0].to_dict()['path'])
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(SourceIntegrityError):
        assembly.load()


@pytest.mark.parametrize('failure', ['duplicate_id','units','clock','unknown'])
def test_data_contract_rejects_ambiguous_inputs(csv_case,failure):
    specs, panel = csv_case()
    if failure=='duplicate_id':
        specs.append(deepcopy(specs[0]))
    elif failure=='units':
        specs[0]['units']['x']='mg/m3'
    elif failure=='clock':
        specs[0]['time']['frequency']='30min'
    else:
        specs[0]['silently_fill']=True
    with pytest.raises(DataContractError):
        build_data(specs,panel_spec=panel)


def test_cross_file_duplicate_is_not_arbitrarily_deduplicated(csv_case,tmp_path):
    specs, panel = csv_case()
    second = deepcopy(specs[0])
    duplicate = tmp_path/'duplicate.csv'
    duplicate.write_bytes(Path(second['path']).read_bytes())
    second.update(id='second',path=str(duplicate))
    with pytest.raises(DuplicateObservationError):
        build_data([*specs,second],panel_spec=panel).load()
