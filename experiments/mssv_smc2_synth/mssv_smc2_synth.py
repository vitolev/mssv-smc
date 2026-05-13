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
from src.models.mssv import MSSVParams, MSSVModel
from src.filters.smc2.smc2 import SMC2
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.smc.resampling import systematic_resampling
from src.diagnostics.plotting_smc2 import plot_histograms
from src.utils.utils import ROOT_DIR

def main():
    # Get location of this script
    script_dir = Path(__file__).resolve().parent
    # Get config file path
    config_path = script_dir / "config.yaml"
    config = Config.from_yaml(config_path)
    # Extact values from config
    name: str = config.name
    T: int = config.T
    K: int = config.K
    N_x: int = config.N_x
    N_theta: int = config.N_theta
    gamma: float = config.gamma
    x_thin: int = config.x_thin

    # Prior parameters
    mu_mean: float = config.prior.mu_mean
    mu_sd: float = config.prior.mu_sd
    phi_a: float = config.prior.phi_a
    phi_b: float = config.prior.phi_b
    sigma_eta_a: float = config.prior.sigma_eta_a
    sigma_eta_b: float = config.prior.sigma_eta_b
    diff_mean: float = config.prior.diff_mean
    diff_sd: float = config.prior.diff_sd
    P_diag: float = config.prior.P_diag
    P_base: float = config.prior.P_base

    # Proposal params
    mode = config.proposal.mode
    step_mu: float = config.proposal.step_mu
    step_delta: float = config.proposal.step_delta
    step_phi: float = config.proposal.step_phi
    step_sigma: float = config.proposal.step_sigma
    step_P: int = config.proposal.step_P

    # Create subfolder with name of the experiment
    script_dir = script_dir / name
    script_dir.mkdir(parents=True)
    # Save a copy of config file in the experiment folder
    config.save_yaml(script_dir / "config.yaml")

    logger = setup_main_logging(script_dir, name)
    logger.info("=" * 60)
    logger.info("SMC^2 algorithm with Bootstrap Particle Filter (BPF)")
    logger.info("=" * 60)

    logger.info("Project overview:")
    logs_dir = script_dir / 'logs'
    results_dir = script_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = script_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
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
    true_theta = MSSVParams.from_mu(
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
    bpf = BootstrapParticleFilter(model, N_x, resampler=systematic_resampling)
    logger.info(f"Initialized Bootstrap Particle Filter")
    logger.info(f"N_x = {N_x}")

    # Test log marginal likelihood mean and variance for fixed parameters
    logmarliks_bpf = []
    for _ in range(200): 
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

    proposal_params = {
        "mode": mode,
        "step_mu": step_mu,
        "step_delta": step_delta,
        "step_phi": step_phi,
        "step_sigma": step_sigma,
        "step_P": step_P
    }

    kwargs_model = {
        "K": K,
    }

    kwargs_prior = {
        "mu_mean": mu_mean,
        "mu_sd": mu_sd,
        "phi_a": phi_a,
        "phi_b": phi_b,
        "sigma_eta_a": sigma_eta_a,
        "sigma_eta_b": sigma_eta_b,
        "diff_mean": diff_mean,
        "diff_sd": diff_sd,
        "P_diag": P_diag,
        "P_base": P_base
    }

    smc2 = SMC2(bpf, N_theta=N_theta, gamma=gamma, proposal_params=proposal_params, kwargs_prior=kwargs_prior, kwargs_model=kwargs_model)

    logger.info(f"Initialized SMC2 sampler")
    logger.info(f"N_theta = {N_theta}")
    logger.info(f"Gamma = {gamma}")
    logger.info("-" * 60)
    logger.info("Model parameters:")
    for k, v in kwargs_model.items():
       logger.info("- %s: %s", k, v)

    logger.info("-" * 60)

    logger.info("Proposal parameters:")
    for k, v in proposal_params.items():
        logger.info("- %s: %s", k, v)

    logger.info("-" * 60)

    logger.info("Prior parameters:")
    for k, v in kwargs_prior.items():
        logger.info("- %s: %s", k, v)

    logger.info("-" * 60)

    logger.info(f"Starting sampling:")

    history = smc2.run(y, logger=logger, thin=x_thin, output_dir=output_dir)

    logger.info(f"SMC2 sampling completed.")
    logger.info(f"Extracting results...")

    # Parameter estimates at last time step
    thetas, logweights, loglikelihoods, ess = history[-1]
    weights = np.exp(logweights - np.max(logweights))  # Normalize log-weights
    weights /= np.sum(weights)
    logger.info(f"Parameter weights statistics: mean={np.mean(weights)}, var={np.var(weights)}, sum={np.sum(weights)}, min={np.min(weights)}, max={np.max(weights)}")
    
    logger.info("Started plotting...")
    plot_histograms(thetas, weights, results_dir)
    logger.info("Plotting completed.")

    logger.info("-" * 60)

if __name__ == "__main__":
    main()