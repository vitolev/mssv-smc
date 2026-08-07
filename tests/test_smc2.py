import numpy as np
import pytest
import os
import h5py
from pathlib import Path

from src.models.mssv import MSSVModel, MSSVParams
from src.models.lgm import LGModel, LGModelParams
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.smc2.smc2 import SMC2
from src.filters.smc.resampling import systematic_resampling

def test_smc2_mssv():
    rng = np.random.default_rng(42)

    # Define model parameters
    theta = MSSVParams.from_mu(
        mu=[0.0, 1.0],
        phi=0.9,
        eta2=0.1,
        P=[[0.9, 0.1], [0.2, 0.8]]
    )
    model = MSSVModel(rng)

    # Generate synthetic data
    T = 10
    state_true = model.sample_initial_state(theta, size=1)
    y = []
    for t in range(T):
        state_true = model.sample_next_state(theta, state_true)
        y_t = model.sample_observation(theta, state_true)
        y.append(y_t)
    y = np.array(y).flatten()

    # Initialize Particle Filter and PIMH
    N_x = 100
    bpf = BootstrapParticleFilter(model, N_x, systematic_resampling)

    proposal_params = {
        "mode": "independent"
    }

    kwargs_prior = {
        "K": theta.K,
    }

    N_theta = 100
    smc2 = SMC2(bpf, N_theta, kwargs_prior=kwargs_prior, proposal_params=proposal_params)

    # Make test directory to store HDF5 output
    smc2_test_dir = Path(__file__).parent / "smc2_test_output"
    smc2_test_dir.mkdir(exist_ok=True)

    smc2.run(y, output_dir=smc2_test_dir)

    assert os.path.exists(os.path.join(smc2_test_dir, "state_history.h5"))
    assert os.path.exists(os.path.join(smc2_test_dir, "theta_history.h5"))

    with h5py.File(os.path.join(smc2_test_dir, "state_history.h5"), "r") as h5f:
        assert "x_particles" in h5f
        assert "trajectories" in h5f
        assert "x_particles_pred" in h5f

    with h5py.File(os.path.join(smc2_test_dir, "theta_history.h5"), "r") as h5f:
        assert "theta" in h5f
        assert "logweights" in h5f
        assert "ess" in h5f
        assert "resampled_times" in h5f

    # Clean up test directory
    for file in smc2_test_dir.iterdir():
        file.unlink()
    smc2_test_dir.rmdir()

def test_smc2_lgm():
    rng = np.random.default_rng(42)

    # Define model parameters
    theta = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)
    model = LGModel(rng)

    # Generate synthetic data
    T = 10
    state_true = model.sample_initial_state(theta, size=1)
    y = []
    for t in range(T):
        state_true = model.sample_next_state(theta, state_true)
        y_t = model.sample_observation(theta, state_true)
        y.append(y_t)
    y = np.array(y).flatten()

    # Initialize Particle Filter and PIMH
    N_x = 100
    bpf = BootstrapParticleFilter(model, N_x, systematic_resampling)

    proposal_params = {
        "mode": "independent"
    }

    N_theta = 100
    smc2 = SMC2(bpf, N_theta, proposal_params=proposal_params)

    # Make test directory to store HDF5 output
    smc2_test_dir = Path(__file__).parent / "smc2_test_output"
    smc2_test_dir.mkdir(exist_ok=True)

    smc2.run(y, output_dir=smc2_test_dir)

    assert os.path.exists(os.path.join(smc2_test_dir, "state_history.h5"))
    assert os.path.exists(os.path.join(smc2_test_dir, "theta_history.h5"))

    with h5py.File(os.path.join(smc2_test_dir, "state_history.h5"), "r") as h5f:
        assert "x_particles" in h5f
        assert "trajectories" in h5f
        assert "x_particles_pred" in h5f

    with h5py.File(os.path.join(smc2_test_dir, "theta_history.h5"), "r") as h5f:
        assert "theta" in h5f
        assert "logweights" in h5f
        assert "ess" in h5f
        assert "resampled_times" in h5f

    # Clean up test directory
    for file in smc2_test_dir.iterdir():
        file.unlink()
    smc2_test_dir.rmdir()