import numpy as np
import pytest

from src.filters.smc.auxiliary_pf import AuxiliaryParticleFilter

from src.models.lgm import LGModel, LGModelParams, LGModelState
from src.filters.smc.resampling import systematic_resampling

def test_auxiliary_particle_filter():
    rng = np.random.default_rng(42)

    # Define model parameters
    theta = LGModelParams(a=0.9, b=1.0, sigma_x=1.0, sigma_y=1.0)

    # Create model and particle filter
    model = LGModel(rng=rng)
    n_particles = 1000
    apf = AuxiliaryParticleFilter(model, n_particles, systematic_resampling)

    # Asserting initialization
    assert apf.N == n_particles
    assert apf.model == model
    assert apf.resampler == systematic_resampling

    # Generate synthetic observations
    T = 10
    x_true = model.sample_initial_state(theta)
    y = []
    for t in range(T):
        x_true = model.sample_next_state(theta, x_true)
        y_t = model.sample_observation(theta, x_true)
        y.append(y_t)
    y = np.array(y).flatten()

    # Run particle filter
    history = apf.run(y, theta)

    # Basic assertions
    assert len(history) == T+1
    for t in range(T+1):
        particles, weights, indices, loglik = history[t]
        assert isinstance(particles, LGModelState)
        assert particles.x_t.shape == (n_particles,)
        assert weights.shape == (n_particles,)
        assert np.isclose(weights.sum(), 1.0)
        assert indices.shape == (n_particles,) or indices.shape == (0,)  # At t=0, indices is an empty list
        assert isinstance(loglik, float)