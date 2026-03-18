# General imports
import ast
import logging
import os
from pathlib import Path
from src.utils.log import setup_main_logging
from src.utils.config import Config
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# Experiment specific imports
from src.models.mssv import MSSVModelParams, MSSVModel
from src.filters.pmcmc.pmmh import ParticleMarginalMetropolisHastings
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.smc.resampling import systematic_resampling
from src.diagnostics.plotting_pmmh import plot_traceplots, plot_histograms
from src.utils.utils import ROOT_DIR

def main():
    # Get location of this script
    script_dir = Path(__file__).resolve().parent
    # Get config file path
    config_path = script_dir / "config.yaml"
    config = Config.from_yaml(config_path)
    # Extact values from config
    name = config.name
    T = config.T
    K = config.K
    N = config.N
    M = config.pmmh.M
    C = config.pmmh.C
    burnin = config.pmmh.burnin

    step_mu = config.proposal.step_mu
    step_delta = config.proposal.step_delta
    step_phi = config.proposal.step_phi
    step_sigma = config.proposal.step_sigma
    step_P = config.proposal.step_P

    logger = setup_main_logging(script_dir, name)
    logger.info("=" * 60)
    logger.info("Particle Marginal Metropolis-Hastings (PMMH) algorithm with Bootstrap Particle Filter (BPF)")
    logger.info("=" * 60)

    logger.info("Project overview:")
    logs_dir = script_dir / 'logs'
    results_dir = script_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT_DIR / 'data'
    logger.info(f"- Logs dir: {logs_dir}")
    logger.info(f"- Results dir: {results_dir}")
    logger.info(f"- Data dir: {data_dir}")

    logger.info("=" * 60)

    # Random seed
    rng = np.random.default_rng(123)
    # Initialize model
    model = MSSVModel(rng=rng)

    # Load data
    data_path = data_dir / "synthetic" / f"data_T_2000_{K}_regime.csv"
    data = pd.read_csv(data_path)
    y = data["y"].values
    h_true = data["h_true"].values
    s_true = data["s_true"].values.astype(int)
    y = y[:T]
    h_true = h_true[:T]
    s_true = s_true[:T]

    param_path = data_dir / "synthetic" / f"data_T_2000_{K}_regime_params.csv"
    params_df = pd.read_csv(param_path)
    P_rows = params_df["P"].apply(ast.literal_eval).tolist()
    P = np.array(P_rows)
    true_theta = MSSVModelParams(
        mu=params_df["mu"].values,
        phi=params_df["phi"].iloc[0],
        sigma_eta=params_df["sigma_eta"].iloc[0],
        P=P
    )

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
        "step_mu": step_mu,
        "step_delta": step_delta,
        "step_phi": step_phi,
        "step_sigma": step_sigma,
        "step_P": step_P
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

    logger.info("-" * 60)

    logger.info("Plotting diagnostics ...")
    
    plot_traceplots(results_bpf, results_dir)
    plot_histograms(results_bpf, results_dir)

    # Now let's look at samples of trajectories
    plt.figure(figsize=(12, 8))
    for chain in range(len(results_bpf)):
        samples, _, _, _ = results_bpf[chain]
        # Compute mean trajectory post burn-in
        samples_h = np.array([sample.h_t for sample in samples])    # shape (T+1, N)
        mean_trajectory = np.mean(samples_h, axis=1)
        plt.plot(mean_trajectory, label=f"Chain {chain+1}", alpha=0.7)
        # Plot random 10 trajectories
        for i in range(10):
            index = np.random.randint(0, samples_h.shape[1])
            plt.plot(samples_h[:, index], color='gray', alpha=0.1)
            
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