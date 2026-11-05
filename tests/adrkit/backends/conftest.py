"""Малый протокол для проверки сеточного решателя."""

import pytest


@pytest.fixture
def small_protocol():
    return dict(
        model=dict(
            horizon=1.0,
            diffusion=0.2,
            velocity=[0.2, -0.1],
            linear_loss=0.1,
            reaction_gamma=0.4,
            reaction_c_star=0.8,
            source_position=[0.0, 0.0],
        ),
        control=dict(
            knots=[0.0, 0.5, 1.0],
            regularization_time_scale=0.3,
            regularization_alpha=0.2,
            prior_coefficients=0.35,
        ),
        observations=dict(
            centers=[[0.0, 0.0], [0.3, -0.2]],
            spatial_standard_deviation=0.6,
            snapshot_times=[1.0, 0.5],
            time_window=[0.0, 1.0],
        ),
    )
