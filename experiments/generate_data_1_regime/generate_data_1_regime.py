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
from src.data_generation.simulate_data import simulate_data
from src.utils.utils import ROOT_DIR

def main(T, mu, phi, sigma_eta):
    name = "generate_data_1_regime"

    # Get location of this script
    script_dir = Path(__file__).resolve().parent

    logger = setup_main_logging(script_dir, name)
    logger.info("=" * 60)
    logger.info("Synthetic data generation for MSSV model with 1 regime")
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

    # True parameters
    true_theta = MSSVModelParams(
        mu=[mu],
        phi=phi,
        sigma_eta=sigma_eta,
        P=[[1.0]]
    )

    x_true, y = simulate_data(model, true_theta, T, rng)

    h_true = [x.h_t for x in x_true]
    h_true = np.array(h_true)
    s_true = [x.s_t for x in x_true]
    s_true = np.array(s_true)

    logger.info(f"True parameters: {true_theta}")
    logger.info(f"True log-volatility shape: {h_true.shape}")   # shape: (T, 1)
    logger.info(f"True regimes shape: {s_true.shape}")          # shape: (T, 1, 1) - one-hot encoded regimes
    logger.info(f"Observations (returns) shape: {y.shape}")     # shape: (T, 1)
    logger.info("-" * 60)

    # Reshape s_true by converting one-hot encoded regimes to integer labels
    s_true_labels = np.argmax(s_true, axis=-1)

    # Save the generated data to pandas DataFrame
    data_df = pd.DataFrame({
        'h_true': h_true.flatten(),  # Flatten to 1D array
        's_true': s_true_labels.flatten(),  # Flatten to 1D array
        'y': y.flatten()  # Flatten to 1D array
    })

    # Save the DataFrame to a CSV file
    data_file = data_dir / "synthetic" / f'data_T_{T}_1_regime.csv'
    data_df.to_csv(data_file, index=False)
    logger.info(f"Saved generated data to {data_file}")

    # Save the true parameters to a CSV file
    params_df = pd.DataFrame({
        'mu': true_theta.mu,
        'phi': true_theta.phi,
        'sigma_eta': true_theta.sigma_eta,
        'P': [true_theta.P[0].tolist()]
    }, index=[0])
    params_file = data_dir / "synthetic" / f'data_T_{T}_1_regime_params.csv'
    params_df.to_csv(params_file, index=False)
    logger.info(f"Saved true parameters to {params_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--T", type=int, required=True, help="Number of time steps to generate")
    parser.add_argument("--mu", type=float, default=-1.0, help="True mu parameter")
    parser.add_argument("--phi", type=float, default=0.9, help="True phi parameter. Must be in (-1, 1)")
    parser.add_argument("--sigma_eta", type=float, default=0.1, help="True sigma_eta parameter. Must be positive")

    args = parser.parse_args()

    main(args.T, args.mu, args.phi, args.sigma_eta)