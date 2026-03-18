# General imports
import logging
import os
from pathlib import Path
from src.utils.log import setup_main_logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# Experiment specific imports
from src.models.mssv import MSSVModelParams, MSSVModel
from src.filters.pmcmc.pmmh import ParticleMarginalMetropolisHastings
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.smc.resampling import systematic_resampling
from src.diagnostics.plotting_pmmh import plot_traceplots

def main(N, K, M, C, burnin):
    name = "mssv_pmmh_bpf_synth_T_1000"

    logger = setup_main_logging(name)
    logger.info("=" * 60)
    logger.info("Particle Marginal Metropolis-Hastings (PMMH) algorithm with Bootstrap Particle Filter (BPF) on synthetic data with T=1000")
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

    T = 1000

    # Load data
    data_path = data_dir / "synthetic" / "data_T_2000_1_regime.csv"
    data = pd.read_csv(data_path)
    y = data["y"].values
    h_true = data["h_true"].values
    s_true = data["s_true"].values.astype(int)
    y = y[:T]
    h_true = h_true[:T]
    s_true = s_true[:T]

    param_path = data_dir / "synthetic" / "data_T_2000_1_regime_params.csv"
    params_df = pd.read_csv(param_path)
    true_theta = MSSVModelParams(
        mu=[params_df["mu"].iloc[0]],
        phi=params_df["phi"].iloc[0],
        sigma_eta=params_df["sigma_eta"].iloc[0],
        P=np.array([[1.0]])
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
        "step_mu": 0.2,
        "step_delta": 0.1,
        "step_phi": 0.1,
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

    logger.info("-" * 60)

    logger.info("Plotting diagnostics ...")
    
    plot_traceplots(results_bpf, results_dir)

    # Plot sigma_eta vs phi scatter plot
    plt.figure(figsize=(12, 8))
    for chain in range(len(results_bpf)):
        samples, logmarliks, thetas, logalphas = results_bpf[chain]
        phi_samples = thetas["phi"]
        sigma_eta_samples = thetas["sigma_eta"]
        plt.scatter(phi_samples, sigma_eta_samples, label=f"Chain {chain+1}", alpha=0.5)
    plt.xlabel("phi")
    plt.ylabel("sigma_eta")
    plt.title("Scatter Plot of phi vs sigma_eta")
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / "phi_sigma_eta_scatter.png")
    plt.close()

    # Likelihood diagnostics at fixed mu, sigma_eta
    phi_values = np.linspace(-0.99, 0.99, 100)
    logmarliks_phi_mean = []
    logmarliks_phi_var = []
    for phi in phi_values:
        logger.info(f"Evaluating log marginal likelihood for phi = {phi:.3f}")
        theta = MSSVModelParams(
            mu=true_theta.mu,
            phi=phi,
            sigma_eta=true_theta.sigma_eta,
            P=true_theta.P
        )
        temp = []
        for _ in range(100):
            history = bpf.run(y, theta)
            logmarlik = history[-1][3]
            temp.append(logmarlik)

        log_vals = np.array(temp) 
        m = np.max(log_vals)
        log_avg = m + np.log(np.mean(np.exp(log_vals - m)))
        logmarliks_phi_mean.append(log_avg)
        logmarliks_phi_var.append(np.var(log_vals))
    plt.figure(figsize=(12, 8))
    plt.plot(phi_values, logmarliks_phi_mean, label="Mean Log Marginal Likelihood")
    plt.fill_between(phi_values, 
                     np.array(logmarliks_phi_mean) - 2*np.sqrt(logmarliks_phi_var), 
                     np.array(logmarliks_phi_mean) + 2*np.sqrt(logmarliks_phi_var), 
                     color='gray', alpha=0.5, label="±2 Std Dev")
    plt.xlabel("phi")
    plt.ylabel("Log Marginal Likelihood")
    plt.title("Log Marginal Likelihood vs phi")
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / "logmarlik_vs_phi.png")
    plt.close()

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