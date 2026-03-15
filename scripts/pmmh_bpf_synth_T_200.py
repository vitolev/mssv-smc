# General imports
import logging
import os
from pathlib import Path
from src.utils.log import setup_main_logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

# Experiment specific imports
from src.models.mssv import MSSVModelParams, MSSVModel
from src.filters.pmcmc.pmmh import ParticleMarginalMetropolisHastings
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.smc.resampling import systematic_resampling
from src.data_generation.simulate_data import simulate_data
from src.diagnostics.plotting_pmmh import plot_traceplots
from src.utils.mssv_utils import compute_transition_counts

def main(N, K, M, C, burnin):
    name = "pmmh_bpf_synth_T_200"

    logger = setup_main_logging(name)
    logger.info("=" * 60)
    logger.info("Particle Marginal Metropolis-Hastings (PMMH) algorithm with Bootstrap Particle Filter (BPF) on synthetic data with T=200")
    logger.info("=" * 60)

    logger.info("Project overview:")
    logs_dir = Path(os.environ['ROOT_DIR']) / 'experiments' / name / 'logs'
    results_dir = Path(os.environ['ROOT_DIR']) / 'experiments' / name / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(os.environ['ROOT_DIR']) / 'data'
    logger.info(f"- Logs dir: {logs_dir}")
    logger.info(f"- Results dir: {results_dir}")
    logger.info(f"- Data dir: {data_dir}")

    logger.info("=" * 60)

    # Random seed
    rng = np.random.default_rng(123)
    # Initialize model
    model = MSSVModel(rng=rng)

    # True parameters
    true_theta = MSSVModelParams(
        mu=[-1.0],
        phi=0.9,
        sigma_eta=0.1,
        P=[[1.0]]
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
    logger.info("-" * 60)

    # BPF initialization
    bpf = BootstrapParticleFilter(model, N, resampler=systematic_resampling)
    logger.info(f"Initialized Bootstrap Particle Filter")
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
    logger.info(f"Variance: {np.var(logmarliks_bpf)}\n")

    logger.info(f"BPF marginal likelihood")
    logger.info(f"Mean: {np.mean(np.exp(logmarliks_bpf))}")
    logger.info(f"Variance: {np.var(np.exp(logmarliks_bpf))}")

    logger.info("-" * 60)

    kwargs_for_sampling = {
        "step_mu": 0.2,
        "step_delta": 0.1,
        "step_phi": 0.05,
        "step_sigma": 0.2,
        "step_P": 500.0
    }

    kwargs_for_params = {
        "num_regimes": K
    }

    pmmh = ParticleMarginalMetropolisHastings(bpf, kwargs_for_sampling=kwargs_for_sampling, kwargs_for_params=kwargs_for_params)

    logger.info(f"Initialized PMMH sampler")
    logger.info(f"K = {K}")
    logger.info(
        "Sampling parameters:\n%s",
        "\n ".join(f"{k}: {v}" for k, v in kwargs_for_sampling.items())
    )

    logger.info("-" * 60)

    logger.info(f"Initializing sampling with parameters:")
    logger.info(f"- M = {M}")
    logger.info(f"- C = {C}")
    logger.info(f"- Burn-in = {burnin}")
    logger.info("-" * 60)

    results_bpf = pmmh.run(y, n_iter=M, n_chain=C, burnin=burnin, name=name)

    logger.info(f"PMMH sampling completed.")

    # Save results to pickle
    with open(results_dir / (name + "_results.pkl"), "wb") as f:
        pickle.dump(results_bpf, f)
    logger.info(f"Saved PMMH results to {results_dir / (name + '_results.pkl')}")

    logger.info("-" * 60)

    logger.info(f"PMMH chains diagnostics")
    
    # Log initial parameters theta for each chain
    for chain in range(C):
        samples, logmarlik, thetas, logalphas = results_bpf[chain]
        initial_theta = {key: values[0] for key, values in thetas.items()}
        logger.info(f"Chain {chain+1} initial theta: {initial_theta}")

    logger.info("-" * 60)
    logger.info("Plotting diagnostics ...")
    
    plot_traceplots(results_bpf, results_dir)

    # Now let's look at samples of trajectories
    plt.figure(figsize=(12, 8))
    for chain in range(len(results_bpf)):
        samples, _, _, _ = results_bpf[chain]
        # Compute mean trajectory post burn-in
        samples_h = np.array([sample.h_t for sample in samples])    # shape (T+1, N)
        mean_trajectory = np.mean(samples_h, axis=1)
        plt.plot(mean_trajectory, label=f"Chain {chain+1}", alpha=0.7)
    plt.plot(np.arange(1, len(h_true)+1), h_true, label="True Trajectory", color='black', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Log Volatility")
    plt.title("Mean Trajectory of Particles")
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / "mean_trajectory.png")
    plt.close()

    logger.info("Diagnostics plotting completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, required=True, help="Number of particles for the BPF")
    parser.add_argument("--K", type=int, required=True, help="Number of regimes in MSSV model")
    parser.add_argument("--M", type=int, required=True, help="Number of MCMC iterations")
    parser.add_argument("--C", type=int, required=True, help="Number of chains")
    parser.add_argument("--burnin", type=int, required=True, help="Number of burn-in iterations")

    args = parser.parse_args()

    main(args.N, args.K, args.M, args.C, args.burnin)