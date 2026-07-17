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
    # Extact values from config
    name = "variance_estimation"
    T = 6663
    N_x = 8000

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

    logger = setup_main_logging(logs_dir, name)
    logger.info("=" * 60)
    logger.info("S&P 500 Particle Filter Variance Estimation")
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
    data['LogReturn'] = (data['Close'] / data['Close'].shift(1)).apply(lambda x: np.log(x))
    data = data.dropna()
    y = data["LogReturn"].values
    y = y[-T:]       # Keep only the last T observations

    logger.info(f"Observations (returns) shape: {y.shape}")
    logger.info("-" * 60)

    # BPF initialization
    bpf = BootstrapParticleFilter(model, N_x, resampler=systematic_resampling)
    logger.info(f"Initialized Bootstrap Particle Filter")
    logger.info(f"N_x = {N_x}")

    logger.info("-" * 60)

    log_liks = []

    P = [[0.99, 0.01], [0.13, 0.87]]
    sample_theta = MSSVParams(-9.8, np.array([1.8]), 0.975, 0.19, np.array(P))

    for i in range(100):
        # Run the particle filter
        history = bpf.run(y, sample_theta, only_last_step=True)
        log_likelihood = history[-1][3]  # Get the log-likelihood from the last step
        log_liks.append(log_likelihood)
        logger.info(f"Run {i+1}: Log-likelihood = {log_likelihood}")

    logger.info("-" * 60)

    logger.info(f"Mean of log-likelihood estimates: {np.mean(log_liks)}")
    logger.info(f"Variance of log-likelihood estimates: {np.var(log_liks)}")

    logger.info("-" * 60)

if __name__ == "__main__":
    main()