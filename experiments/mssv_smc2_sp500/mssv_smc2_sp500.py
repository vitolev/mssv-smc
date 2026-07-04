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
    R: int = config.R
    x_save_factor: int = config.x_save_factor

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

    # Create subfolder with name of the experiment
    script_dir = script_dir / name
    script_dir.mkdir(parents=True)
    logs_dir = script_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = script_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = script_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT_DIR / 'data'
    
    # Save a copy of config file in the experiment folder
    config.save_yaml(script_dir / "config.yaml")

    logger = setup_main_logging(logs_dir, name)
    logger.info("=" * 60)
    logger.info("SMC^2 algorithm for S&P 500 dataset")
    logger.info("=" * 60)

    logger.info("Project overview:")
    logger.info(f"- Logs dir: {logs_dir}")
    logger.info(f"- Results dir: {results_dir}")
    logger.info(f"- Data dir: {data_dir}")

    logger.info("=" * 60)

    # Random seed
    rng = np.random.default_rng(123)
    # Initialize model
    model = MSSVModel(rng=rng)

    # Load data
    data_path = data_dir / "real" / "sp500" / "sp500.csv"
    data = pd.read_csv(data_path)
    y = data["Close"].values
    y = y[-T:]       # Keep only the last T observations

    logger.info(f"Observations (returns) shape: {y.shape}")
    logger.info("-" * 60)

    # BPF initialization
    bpf = BootstrapParticleFilter(model, N_x, resampler=systematic_resampling)
    logger.info(f"Initialized Bootstrap Particle Filter")
    logger.info(f"N_x = {N_x}")

    logger.info("-" * 60)

    proposal_params = {
        "mode": mode,
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

    smc2 = SMC2(bpf, N_theta=N_theta, gamma=gamma, R=R, proposal_params=proposal_params, kwargs_prior=kwargs_prior, kwargs_model=kwargs_model)

    logger.info(f"Initialized SMC2 sampler")
    logger.info(f"N_theta = {N_theta}")
    logger.info(f"Gamma = {gamma}")
    logger.info(f"R = {R}")
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

    smc2.run(y, logger=logger, save_factor=x_save_factor, output_dir=output_dir)

    logger.info(f"SMC2 sampling completed.")

    logger.info("-" * 60)

if __name__ == "__main__":
    main()