"""Малый протокол для проверки сеточного решателя."""
import pytest

@pytest.fixture
def small_protocol():
    return dict(model=dict(horizon=1.,diffusion=.2,velocity=[.2,-.1],linear_loss=.1,
                          reaction_gamma=.4,reaction_c_star=.8,source_position=[0.,0.]),
        control=dict(knots=[0.,.5,1.],regularization_time_scale=.3,
                     regularization_alpha=.2,prior_coefficients=.35),
        observations=dict(centers=[[0.,0.],[.3,-.2]],spatial_standard_deviation=.6,
                          snapshot_times=[1.,.5],time_window=[0.,1.]))
