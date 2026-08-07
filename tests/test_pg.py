import numpy as np
import pytest
import os
import h5py
from pathlib import Path

from src.models.mssv import MSSVModel, MSSVParams
from src.models.lgm import LGModel, LGModelParams
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.pmcmc.pg import ParticleGibbsSampler
from src.filters.smc.resampling import systematic_resampling

def test_pg_mssv():
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
    n_particles = 1000
    bpf = BootstrapParticleFilter(model, n_particles, systematic_resampling)

    kwargs_prior = {
        "K": theta.K,
    }
    
    pg = ParticleGibbsSampler(bpf, kwargs_prior=kwargs_prior)

    # Make test directory to store HDF5 output
    pg_test_dir = Path(__file__).parent / "pg_test_output"
    pg_test_dir.mkdir(exist_ok=True)

    pg.run(y, n_iter=10, n_chain=2, output_dir=pg_test_dir, burnin=5)

    # Assert that the HDF5 file was created
    assert os.path.exists(os.path.join(pg_test_dir, "chain_0.h5"))
    assert os.path.exists(os.path.join(pg_test_dir, "chain_1.h5"))

    # Open file
    with h5py.File(os.path.join(pg_test_dir, "chain_0.h5"), "r") as h5f:
        # Check datasets
        assert "thetas" in h5f
        assert "trajectories" in h5f
        assert "logmarliks" in h5f

        # Check shapes
        assert h5f["thetas"].shape[0] == 5  # n_iter - burnin
        assert h5f["trajectories"].shape[0] == 5
        assert h5f["logmarliks"].shape[0] == 5

    with h5py.File(os.path.join(pg_test_dir, "chain_1.h5"), "r") as h5f:
        # Check datasets
        assert "thetas" in h5f
        assert "trajectories" in h5f
        assert "logmarliks" in h5f

        # Check shapes
        assert h5f["thetas"].shape[0] == 5  # n_iter - burnin
        assert h5f["trajectories"].shape[0] == 5
        assert h5f["logmarliks"].shape[0] == 5

    # Clean up test directory
    for file in pg_test_dir.iterdir():
        file.unlink()
    pg_test_dir.rmdir()



# Currently commented out because the LGModel does not support the conditional proposal mode, which is required for the Particle Gibbs sampler.
# Once the LGModel is updated to support this, the test can be uncommented and run.

# def test_pg_lgm():
#     rng = np.random.default_rng(42)

#     # Define model parameters
#     theta = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)
#     model = LGModel(rng)

#     # Generate synthetic data
#     T = 10
#     state_true = model.sample_initial_state(theta, size=1)
#     y = []
#     for t in range(T):
#         state_true = model.sample_next_state(theta, state_true)
#         y_t = model.sample_observation(theta, state_true)
#         y.append(y_t)
#     y = np.array(y).flatten()

#     # Initialize Particle Filter and PIMH
#     n_particles = 1000
#     bpf = BootstrapParticleFilter(model, n_particles, systematic_resampling)
    
#     pg = ParticleGibbsSampler(bpf)

#     # Make test directory to store HDF5 output
#     pg_test_dir = Path(__file__).parent / "pg_test_output"
#     pg_test_dir.mkdir(exist_ok=True)

#     pg.run(y, n_iter=10, n_chain=2, output_dir=pg_test_dir, burnin=5)

#     # Assert that the HDF5 file was created
#     assert os.path.exists(os.path.join(pg_test_dir, "chain_0.h5"))
#     assert os.path.exists(os.path.join(pg_test_dir, "chain_1.h5"))

#     # Open file
#     with h5py.File(os.path.join(pg_test_dir, "chain_0.h5"), "r") as h5f:
#         # Check datasets
#         assert "thetas" in h5f
#         assert "trajectories" in h5f
#         assert "logmarliks" in h5f

#         # Check shapes
#         assert h5f["thetas"].shape[0] == 5  # n_iter - burnin
#         assert h5f["trajectories"].shape[0] == 5
#         assert h5f["logmarliks"].shape[0] == 5

#     with h5py.File(os.path.join(pg_test_dir, "chain_1.h5"), "r") as h5f:
#         # Check datasets
#         assert "thetas" in h5f
#         assert "trajectories" in h5f
#         assert "logmarliks" in h5f

#         # Check shapes
#         assert h5f["thetas"].shape[0] == 5  # n_iter - burnin
#         assert h5f["trajectories"].shape[0] == 5
#         assert h5f["logmarliks"].shape[0] == 5

#     # Clean up test directory
#     for file in pg_test_dir.iterdir():
#         file.unlink()
#     pg_test_dir.rmdir()