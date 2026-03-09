# General imports
import logging
import os
from pathlib import Path
from src.utils.log import setup_main_logging
import numpy as np

# Experiment specific imports
from src.models.mssv import MSSVModelParams, MSSVModel
from src.filters.pmcmc.pmmh import ParticleMarginalMetropolisHastings
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.smc.resampling import systematic_resampling
from src.data_generation.simulate_data import simulate_data

def main():
    name = "pmmh_bpf_synth_T_200"

    logger = setup_main_logging(name)
    logger.info("=" * 60)
    logger.info("Particle Marginal Metropolis-Hastings (PMMH) algorithm with Bootstrap Particle Filter (BPF) on synthetic data with T=200")
    logger.info("=" * 60)

    logger.info("Project overview:")
    logs_dir = Path(os.environ['ROOT_DIR']) / 'experiments' / name / 'logs'
    data_dir = Path(os.environ['ROOT_DIR']) / 'data'
    logger.info(f"- Logs dir: {logs_dir}")
    logger.info(f"- Data dir: {data_dir}")

    logger.info("=" * 60)
    logger.info("STARTING PIPELINE")
    logger.info("=" * 60 + "\n")

    # Random seed
    rng = np.random.default_rng(123)
    # Initialize model
    model = MSSVModel(rng=rng)

    # True parameters
    true_theta = MSSVModelParams(
        mu=[-1.0, 1.6],
        phi=0.95,
        sigma_eta=0.1,
        P=[[0.95, 0.05],
            [0.05, 0.95]]
    )

    T = 200

    x_true, y = simulate_data(model, true_theta, T, rng)

    h_true = [x.h_t for x in x_true]
    h_true = np.array(h_true)
    s_true = [x.s_t for x in x_true]
    s_true = np.array(s_true)

    logger.info(f"True parameters: {true_theta}")
    logger.info(f"True log-volatility shape: {h_true.shape}")
    logger.info(f"True regimes shape: {s_true.shape}")
    logger.info(f"Observations (returns) shape: {y.shape}")

    # BPF initialization
    N = 3000
    bpf = BootstrapParticleFilter(model, N, resampler=systematic_resampling)
    logger.info(f"\nInitialized Bootstrap Particle Filter")
    logger.info(f"N = {N}")

    # Test log marginal likelihood mean and variance for fixed parameters
    logmarliks_bpf = []
    for _ in range(1000): 
        history = bpf.run(y, true_theta)
        logmarlik = history[-1][3]
        logmarliks_bpf.append(logmarlik)

    logger.info("-" * 60)

    logger.info(f"BPF log marginal likelihood")
    logger.info(f"Mean: {np.mean(logmarliks_bpf)}")
    logger.info(f"Variance: {np.var(logmarliks_bpf)}")

    logger.info(f"\nBPF marginal likelihood")
    logger.info(f"Mean: {np.mean(np.exp(logmarliks_bpf))}")
    logger.info(f"Variance: {np.var(np.exp(logmarliks_bpf))}")

    logger.info("-" * 60)

    kwargs_for_sampling = {
        "step_mu": 0.2,
        "step_delta": 0.4,
        "step_phi": 0.05,
        "step_sigma": 0.2,
        "step_P": 100.0
    }

    K = 2
    kwargs_for_params = {
        "num_regimes": K
    }

    pmmh = ParticleMarginalMetropolisHastings(bpf, kwargs_for_sampling=kwargs_for_sampling, kwargs_for_params=kwargs_for_params)

    logger.info(f"\nInitialized PMMH sampler")
    logger.info(f"K = {K}")
    logger.info(f"Sampling parameters: {kwargs_for_sampling}")

    logger.info("-" * 60)

    M = 10000
    C = 8
    burnin = 4000
    logger.info(f"Initializing sampling with parameters:")
    logger.info(f"- M = {M}")
    logger.info(f"- C = {C}")
    logger.info(f"- Burn-in = {burnin}")

    results_bpf = pmmh.run(y, n_iter=M, n_chain=C, burnin=burnin, name=name)

    logger.info(f"\nPMMH sampling completed.")

if __name__ == "__main__":
    main()