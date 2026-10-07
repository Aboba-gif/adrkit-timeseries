"""Малые сгенерированные входы, без копирования научных данных."""
import csv
import hashlib
import pytest


@pytest.fixture
def csv_case(tmp_path):
    panel = dict(start='2020-01-01 00:00:00', stop='2020-01-01 04:00:00',
        frequency='1h', origin='2020-01-01 00:00:00', clock='naive', timezone=None,
        calendar='proleptic_gregorian', sites=['Z','A'], channels=['x'], units={'x':'ug/m3'}, duplicates='reject')
    rows = [('2020-01-01 00:00:00','A','1'), ('2020-01-01 00:00:00','Z','10'),
            ('2020-01-01 01:00:00','A',''), ('2020-01-01 02:00:00','Z','14'),
            ('2020-01-01 02:00:00','A','5'), ('2020-01-01 03:00:00','Z','16'),
            ('2020-01-01 03:00:00','A','7')]

    def write(shards=1):
        specs = []
        for i in range(shards):
            path = tmp_path/str(shards)/str(i)/'same.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w',newline='') as out:
                writer = csv.writer(out)
                writer.writerow(['time','site','value'])
                writer.writerows(rows[i::shards])
            specs.append(dict(id=f'part-{i}',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                format={'kind':'csv','parameters':{'separator':',','encoding':'utf-8'}},
                columns={'time':['time'],'values':{'x':'value'}},
                time={'parser':'format','format':'%Y-%m-%d %H:%M:%S','dst_policy':'fail',
                      **{k:panel[k] for k in ('timezone','calendar','clock','frequency','origin')}},
                site={'kind':'column','column':'site'},units={'x':'ug/m3'},missing_values=[''],route='strict'))
        return specs, panel.copy()
    return write
