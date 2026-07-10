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
from src.filters.pmcmc.pmmh import ParticleMarginalMetropolisHastings
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
    N: int = config.N
    M: int = config.pmmh.M
    C: int = config.pmmh.C
    burnin: int = config.pmmh.burnin

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
    logger.info("Particle Marginal Metropolis-Hastings (PMMH) algorithm for S&P 500 dataset")
    logger.info("=" * 60)

    logger.info("Project overview:")
    logger.info(f"- Logs dir: {logs_dir}")
    logger.info(f"- Results dir: {results_dir}")
    logger.info(f"- Output dir: {output_dir}")
    logger.info(f"- Data dir: {data_dir}")

    logger.info("=" * 60)

    # Random seed
    rng = np.random.default_rng(123)
    # Initialize model
    model = MSSVModel(rng=rng)

    # Load data
    data_path = data_dir / "real" / "sp500" / "sp500.csv"
    data = pd.read_csv(data_path)
    data['LogReturn'] = (data['Close'] / data['Close'].shift(1)).apply(lambda x: np.log(x))
    data = data.dropna()
    y = data["LogReturn"].values
    y = y[-T:]       # Keep only the last T observations

    logger.info(f"Observations (returns) shape: {y.shape}")
    logger.info("-" * 60)

    # BPF initialization
    bpf = BootstrapParticleFilter(model, N, resampler=systematic_resampling)
    logger.info(f"Initialized Bootstrap Particle Filter")
    logger.info(f"N = {N}")

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

    pmmh = ParticleMarginalMetropolisHastings(bpf, proposal_param=proposal_params, kwargs_prior=kwargs_prior, kwargs_model=kwargs_model)

    logger.info(f"Initialized PMMH sampler")
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

    logger.info(f"Starting sampling with parameters:")
    logger.info(f"- M = {M}")
    logger.info(f"- C = {C}")
    logger.info(f"- Burn-in = {burnin}")
    logger.info("-" * 60)

    pmmh.run(y, n_iter=M, n_chain=C, burnin=burnin, output_dir=output_dir, logs_dir=logs_dir)

    logger.info(f"PMMH sampling completed.")

if __name__ == "__main__":
    main()