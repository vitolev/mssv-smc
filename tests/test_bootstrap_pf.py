import numpy as np
import pytest

from src.filters.smc.bootstrap_pf import BootstrapParticleFilter

from src.models.lgm import LGModel, LGModelParams, LGModelState
from src.models.mssv import MSSVModel, MSSVParams, MSSVState
from src.filters.smc.resampling import systematic_resampling

def test_bootstrap_particle_filter_lgm():
    rng = np.random.default_rng(42)

    # Define model parameters
    theta = LGModelParams(a=0.9, b=1.0, sigma_x=1.0, sigma_y=1.0)

    # Create model and particle filter
    model = LGModel(rng=rng)
    n_particles = 1000
    bpf = BootstrapParticleFilter(model, n_particles, systematic_resampling)

    # Asserting initialization
    assert bpf.N == n_particles
    assert bpf.model == model
    assert bpf.resampler == systematic_resampling

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
    history = bpf.run(y, theta)

    smoothing_traj, _ = bpf.smoothing_trajectories(history, n_traj=5)
    assert len(smoothing_traj) == T+1

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

        assert isinstance(smoothing_traj[t], LGModelState)
        assert len(smoothing_traj[t]) == 5

def test_bootstrap_particle_filter_mssv():
    rng = np.random.default_rng(42)

    # Define model parameters
    theta = MSSVParams.from_mu(
        mu=[0.0, 1.0],
        phi=0.9,
        sigma_eta=0.1,
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    # Create model and particle filter
    model = MSSVModel(rng=rng)
    n_particles = 1000
    bpf = BootstrapParticleFilter(model, n_particles, systematic_resampling)

    # Asserting initialization
    assert bpf.N == n_particles
    assert bpf.model == model
    assert bpf.resampler == systematic_resampling

    # Generate synthetic observations
    T = 10
    state_true = model.sample_initial_state(theta, size=1)
    y = []
    for t in range(T):
        state_true = model.sample_next_state(theta, state_true)
        y_t = model.sample_observation(theta, state_true)
        y.append(y_t)
    y = np.array(y).flatten()

    # Run particle filter
    history = bpf.run(y, theta)

    smoothing_traj, _ = bpf.smoothing_trajectories(history, n_traj=5)
    assert len(smoothing_traj) == T+1

    # Basic assertions
    assert len(history) == T+1
    for t in range(T+1):
        particles, weights, indices, loglik = history[t]
        assert isinstance(particles, MSSVState)
        assert particles.h_t.shape == (n_particles,)
        assert particles.s_t.shape == (n_particles, len(theta.mu))
        assert weights.shape == (n_particles,)
        assert np.isclose(weights.sum(), 1.0)
        assert indices.shape == (n_particles,) or indices.shape == (0,)  # At t=0, indices is an empty list
        assert isinstance(loglik, float)

        assert isinstance(smoothing_traj[t], MSSVState)
        assert len(smoothing_traj[t]) == 5

def test_conditional_bootstrap_particle_filter():
    rng = np.random.default_rng(42)

    # Define model parameters
    theta = LGModelParams(a=0.9, b=1.0, sigma_x=1.0, sigma_y=1.0)

    # Create model and particle filter
    model = LGModel(rng=rng)
    n_particles = 1000
    bpf = BootstrapParticleFilter(model, n_particles, systematic_resampling)

    # Generate synthetic observations
    T = 10
    x_true = model.sample_initial_state(theta)
    true_path = [x_true]
    y = []
    for t in range(T):
        x_true = model.sample_next_state(theta, x_true)
        y_t = model.sample_observation(theta, x_true)
        y.append(y_t)
        true_path.append(x_true)
    y = np.array(y).flatten()

    # Run conditional particle filter with the true state as the conditioned path
    history = bpf.run_conditional(y, theta, true_path)

    smoothing_traj, _ = bpf.smoothing_trajectories(history, n_traj=5)
    assert len(smoothing_traj) == T+1

    # Basic assertions (similar to previous test)
    assert len(history) == T+1
    for t in range(T+1):
        particles, weights, indices, loglik = history[t]
        assert isinstance(particles, LGModelState)
        assert particles.x_t.shape == (n_particles,)
        assert weights.shape == (n_particles,)
        assert np.isclose(weights.sum(), 1.0)
        assert indices.shape == (n_particles,) or indices.shape == (0,)  # At t=0, indices is an empty list
        assert isinstance(loglik, float)

        assert isinstance(smoothing_traj[t], LGModelState)
        assert len(smoothing_traj[t]) == 5

def test_conditional_bootstrap_particle_filter_mssv():
    rng = np.random.default_rng(42)

    # Define model parameters
    theta = MSSVParams.from_mu(
        mu=[0.0, 1.0],
        phi=0.9,
        sigma_eta=0.1,
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    # Create model and particle filter
    model = MSSVModel(rng=rng)
    n_particles = 1000
    bpf = BootstrapParticleFilter(model, n_particles, systematic_resampling)

    # Generate synthetic observations
    T = 10
    state_true = model.sample_initial_state(theta, size=1)
    true_path = [state_true]
    y = []
    for t in range(T):
        state_true = model.sample_next_state(theta, state_true)
        y_t = model.sample_observation(theta, state_true)
        y.append(y_t)
        true_path.append(state_true)
    y = np.array(y).flatten()

    # Run conditional particle filter with the true state as the conditioned path
    history = bpf.run_conditional(y, theta, true_path)

    smoothing_traj, _ = bpf.smoothing_trajectories(history, n_traj=5)
    assert len(smoothing_traj) == T+1

    # Basic assertions (similar to previous test)
    assert len(history) == T+1
    for t in range(T+1):
        particles, weights, indices, loglik = history[t]
        assert isinstance(particles, MSSVState)
        assert particles.h_t.shape == (n_particles,)
        assert particles.s_t.shape == (n_particles, len(theta.mu))
        assert weights.shape == (n_particles,)
        assert np.isclose(weights.sum(), 1.0)
        assert indices.shape == (n_particles,) or indices.shape == (0,)  # At t=0, indices is an empty list
        assert isinstance(loglik, float)

        smoothing_state = smoothing_traj[t]
        assert isinstance(smoothing_state, MSSVState)
        assert len(smoothing_state) == 5