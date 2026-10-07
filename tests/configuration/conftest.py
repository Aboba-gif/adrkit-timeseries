"""Полная конфигурация составной задачи для проверок сборки."""
import pytest

@pytest.fixture
def composition_config():
    return dict(schema_version='adrkit.experiment/1',pipeline='composition',
        science_schema_version='adrkit.composition/1',science=dict(
            source=dict(kind='p1',parameters=dict(knots=[0,1],times=[0,.5,1],time_unit='s',
                coefficient_unit='1',value_unit='kg/s',scale=2)),
            surrogate=dict(kind='identity',parameters={}),observation=dict(kind='identity',parameters={}),
            regularizer=dict(kind='h1',parameters=dict(center=[.2,.4],alpha=.3,tau=.2)),
            initial=[.4,.8],target=[1,2,1],covariance=[[2,0,0],[0,1,0],[0,0,3]],
            coefficient_metric=[[2,.1],[.1,1]],bounds=None,action='score',
            optimizer=dict(maxiter=30,maxfun=60,ftol=1e-12,gtol=1e-9),
            sensitivity=dict(max_columns=2,max_entries=6,max_jvp_calls=2,relative_threshold=1e-10)))
