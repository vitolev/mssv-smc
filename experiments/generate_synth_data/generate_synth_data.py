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
from src.models.mssv import MSSVParams, MSSVModel
from src.data_generation.simulate_data import simulate_data
from src.utils.utils import ROOT_DIR

def main(name, T, mu, phi, sigma_eta, P):
    # Get location of this script
    script_dir = Path(__file__).resolve().parent

    logger = setup_main_logging(script_dir, name)
    logger.info("=" * 60)
    logger.info("Synthetic data generation for MSSV model")
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

    # Check that the provided parameters are valid
    assert len(mu)**2 == len(P), "The number of regimes implied by mu must match the size of the transition matrix P."
    logger.info("Provided parameters are:")
    logger.info(f"- mu: {mu}")
    logger.info(f"- phi: {phi}")
    logger.info(f"- sigma_eta: {sigma_eta}")
    logger.info(f"- P: {P}")

    # True parameters
    true_theta = MSSVParams.from_mu(
        mu=mu,
        phi=phi,
        sigma_eta=sigma_eta,
        P=np.array(P).reshape(len(mu), len(mu))
    )

    x_true, y = simulate_data(model, true_theta, T, rng)

    h_true = [x.h_t for x in x_true]
    h_true = np.array(h_true)
    s_true = [x.s_t for x in x_true]
    s_true = np.array(s_true)

    logger.info(f"True parameters: {true_theta}")
    logger.info(f"True log-volatility shape: {h_true.shape}")   # shape: (T, 1)
    logger.info(f"True regimes shape: {s_true.shape}")          # shape: (T, 1, len(mu)) - one-hot encoded regimes
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
    data_file = data_dir / "synthetic" / f'{name}.csv'
    data_df.to_csv(data_file, index=False)
    logger.info(f"Saved generated data to {data_file}")

    # Save the true parameters to a CSV file. For phi and sigma_eta, duplicate entries for both regimes. For P, save the row for each regime.
    params_df = pd.DataFrame({
        'mu': true_theta.mu,
        'phi': [true_theta.phi] * len(mu),
        'sigma_eta': [true_theta.sigma_eta] * len(mu),
        'P': [true_theta.P[i].tolist() for i in range(len(mu))]
    })
    params_file = data_dir / "synthetic" / f'{name}_params.csv'
    params_df.to_csv(params_file, index=False)
    logger.info(f"Saved true parameters to {params_file}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True, help="Name of the dataset/experiment. ")
    parser.add_argument("--T", type=int, required=True, help="Number of time steps to generate")
    parser.add_argument("--mu", nargs="+", type=float, default=[-1.0, 1.6], help="True mu parameters.")
    parser.add_argument("--phi", type=float, default=0.9, help="True phi parameter. Must be in (-1, 1)")
    parser.add_argument("--sigma_eta", type=float, default=0.1, help="True sigma_eta parameter. Must be positive")
    parser.add_argument("--P", nargs="+", type=float, default=[0.95, 0.05, 0.05, 0.95], help="Transition probabilities for the regime-switching model.")

    args = parser.parse_args()

    main(args.name, args.T, args.mu, args.phi, args.sigma_eta, args.P)