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

def find_project_root(start: Path, marker=".gitignore"):
    for parent in [start] + list(start.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")

ROOT_DIR = find_project_root(Path(__file__).resolve())

def main(T, mu, phi, sigma_eta, P):
    name = "generate_data_2_regime"

    # Get location of this script
    script_dir = Path(__file__).resolve().parent

    logger = setup_main_logging(script_dir, name)
    logger.info("=" * 60)
    logger.info("Particle Marginal Metropolis-Hastings (PMMH) algorithm with Bootstrap Particle Filter (BPF) on synthetic data with T=200")
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
        mu=mu,
        phi=phi,
        sigma_eta=sigma_eta,
        P=np.array(P).reshape(2, 2)
    )

    x_true, y = simulate_data(model, true_theta, T, rng)

    h_true = [x.h_t for x in x_true]
    h_true = np.array(h_true)
    s_true = [x.s_t for x in x_true]
    s_true = np.array(s_true)

    logger.info(f"True parameters: {true_theta}")
    logger.info(f"True log-volatility shape: {h_true.shape}")   # shape: (T, 1)
    logger.info(f"True regimes shape: {s_true.shape}")          # shape: (T, 1, 2) - one-hot encoded regimes
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
    data_file = data_dir / "synthetic" / f'data_T_{T}_2_regime.csv'
    data_df.to_csv(data_file, index=False)
    logger.info(f"Saved generated data to {data_file}")

    # Save the true parameters to a CSV file. For phi and sigma_eta, duplicate entries for both regimes. For P, save the row for each regime.
    params_df = pd.DataFrame({
        'mu': true_theta.mu,
        'phi': [true_theta.phi] * 2,
        'sigma_eta': [true_theta.sigma_eta] * 2,
        'P': [true_theta.P[0].tolist(), true_theta.P[1].tolist()]
    })
    params_file = data_dir / "synthetic" / f'data_T_{T}_2_regime_params.csv'
    params_df.to_csv(params_file, index=False)
    logger.info(f"Saved true parameters to {params_file}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--T", type=int, required=True, help="Number of time steps to generate")
    parser.add_argument("--mu", nargs=2, type=float, default=[-1.0, 1.6], help="Two true mu parameters.")
    parser.add_argument("--phi", type=float, default=0.9, help="True phi parameter. Must be in (-1, 1)")
    parser.add_argument("--sigma_eta", type=float, default=0.1, help="True sigma_eta parameter. Must be positive")
    parser.add_argument("--P", nargs=4, type=float, default=[0.95, 0.05, 0.05, 0.95], help="Transition probabilities for the regime-switching model.")

    args = parser.parse_args()

    main(args.T, args.mu, args.phi, args.sigma_eta, args.P)