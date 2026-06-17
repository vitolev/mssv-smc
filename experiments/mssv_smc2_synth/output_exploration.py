# General imports
import ast
import logging
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

# Experiment specific imports
from src.models.mssv import MSSVParams, MSSVModel
from src.utils.utils import ROOT_DIR

def weighted_quantile(values, quantiles, sample_weight):
    """
    Compute weighted quantiles for 1D samples.

    Parameters
    ----------
    values : ndarray, shape (N,)
    quantiles : array-like in [0, 1]
    sample_weight : ndarray, shape (N,)

    Returns
    -------
    ndarray
        Quantile values.
    """

    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    sample_weight = np.asarray(sample_weight)

    sorter = np.argsort(values)

    values = values[sorter]
    weights = sample_weight[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    return np.interp(quantiles, cdf, values)

def weighted_credible_interval(values, weights, alpha=0.05):
    """
    Compute weighted credible intervals componentwise.

    Parameters
    ----------
    values : ndarray
        Shape (N, ...)
        where N = number of particles.

    weights : ndarray
        Shape (N,)

    alpha : float
        Credible level. alpha=0.05 -> 95% CI.

    Returns
    -------
    lower : ndarray
    upper : ndarray

    Both have shape values.shape[1:].
    """

    values = np.asarray(values)

    # -----------------------------------------
    # scalar parameter
    # -----------------------------------------
    if values.ndim == 1:
        q = weighted_quantile(
            values,
            [alpha / 2, 1 - alpha / 2],
            weights
        )
        return q[0], q[1]

    # -----------------------------------------
    # flatten parameter dimensions
    # -----------------------------------------
    original_shape = values.shape[1:]

    flat = values.reshape(values.shape[0], -1)

    lower = np.empty(flat.shape[1])
    upper = np.empty(flat.shape[1])

    for j in range(flat.shape[1]):

        q = weighted_quantile(
            flat[:, j],
            [alpha / 2, 1 - alpha / 2],
            weights
        )

        lower[j] = q[0]
        upper[j] = q[1]

    lower = lower.reshape(original_shape)
    upper = upper.reshape(original_shape)

    return lower, upper

def normalize_logweights(logw):
    """
    Stable normalization of log weights.
    """
    m = np.max(logw)
    w = np.exp(logw - m)
    return w / np.sum(w)

def generate_names(n_params):
    """
    Generate dynamically parameter names based on the number of parameters. 
    First 1 is "mu1", next K-1 are "delta_i", next one is "phi", next one is "sigma" and then K*K flatten parameters P_ij.
    In total there is 1 + (K-1) + 1 + 1 + K*K = K*K + K + 2 parameters.
    """
    K = int(-1+np.sqrt(n_params+1))

    names = []
    names.append("mu1")
    for i in range(K-1):
        names.append(f"delta_{i}")
    names.append("phi")
    names.append("sigma")
    for i in range(K):
        for j in range(K):
            names.append(f"P_{i}_{j}")

    return names


def main():
    K = 2
    script_dir = Path(__file__).resolve().parent
    script_dir = script_dir / "T_2000"
    output_dir = script_dir / "output"
    results_dir = script_dir / "results"

    data_dir = ROOT_DIR / 'data'

    # Load data
    data_path = data_dir / "synthetic" / f"data_T_2000_{K}_regime.csv"
    data = pd.read_csv(data_path)
    y = data["y"].values
    h_true = data["h_true"].values
    s_true = data["s_true"].values.astype(int)

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
    true_theta = true_theta.to_vector()


    with h5py.File(output_dir / "theta_history.h5", "r") as h5f:
        theta = h5f["theta"][:]
        logweights = h5f["logweights"][:]

    T_plus_1, N_theta, n_params = theta.shape

    h_true = h_true[:T_plus_1-1]  # Exclude t=0 since it's the prior
    s_true = s_true[:T_plus_1-1]  # Exclude t=0 since it's the prior

    parameter_names = generate_names(n_params)
    print("Parameter names:", parameter_names)

    # Storage
    means = np.zeros((T_plus_1, n_params))
    lower = np.zeros((T_plus_1, n_params))
    upper = np.zeros((T_plus_1, n_params))

    # ---------------------------------------------------
    # Compute weighted summaries
    # ---------------------------------------------------
    for t in range(T_plus_1):
        w = normalize_logweights(logweights[t])
        for j in range(n_params):
            x = theta[t, :, j]
            # weighted mean
            means[t, j] = np.sum(w * x)
            # weighted CI
            lo, hi = weighted_credible_interval(x, w, alpha=0.05)
            lower[t, j] = lo
            upper[t, j] = hi

    # ---------------------------------------------------
    # Plot theta parameters with time
    # ---------------------------------------------------
    time = np.arange(T_plus_1-1)  # Exclude t=0 since it's the prior

    for j in range(n_params):
        plt.figure(figsize=(10, 5))
        plt.plot(time,means[1:, j],label="Mean")                                        # Exclude t=0 since it's the prior
        plt.fill_between(time,lower[1:, j],upper[1:, j],alpha=0.3,label="95% CI")       # Exclude t=0 since it's the prior

        plt.axhline(true_theta[j], color="red", linestyle="--", label="True Value")

        plt.xlabel("Time")
        plt.ylabel(f"{parameter_names[j]}")
        plt.title(f"Posterior of Parameter {parameter_names[j]} over Time")

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(results_dir / f"{parameter_names[j]}.png")

    with h5py.File(output_dir / "state_history.h5", "r") as h5f:
        x_particles = h5f["x_particles"][:]
        trajectories = h5f["trajectories"][:]
        x_particles_pred = h5f["x_particles_pred"][:]

    # Plot smoothing distribution (mean and 95% CI) with time
    last_theta_weights = logweights[-1]
    last_theta_weights = normalize_logweights(last_theta_weights)   # shape (N_theta,)
    N_theta, Tp1, save_factor, state_dim = trajectories.shape

    trajectories = (
        np.swapaxes(trajectories, 1, 2)      # (N_theta, save_factor, T+1, state_dim)
        .reshape(N_theta * save_factor, Tp1, state_dim)
    )

    last_traj_weights = np.repeat(last_theta_weights, save_factor)   # shape (N_theta * save_factor,)
    last_traj_weights = last_traj_weights / np.sum(last_traj_weights)

    h_values = trajectories[:, :, 0]     # shape (N_theta * save_factor, T+1)
    means = np.sum(last_traj_weights[:, None] * h_values, axis=0)
    lower, higher = weighted_credible_interval(h_values, last_traj_weights, alpha=0.05)

    plt.figure(figsize=(12, 8))
    plt.plot(means, label="Mean", color='blue')
    plt.fill_between(np.arange(len(means)), lower, higher, alpha=0.3, label="95% CI", color='blue')
    plt.plot(np.arange(1, len(h_true)+1), h_true, label="True value", color='black', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Log Volatility")
    plt.title("Smoothing Distribution of Log Volatility over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(results_dir / "smoothing_h.png")
    ax = plt.gca()
    ylims = ax.get_ylim()
    plt.close()

    # Plot filtering distribution (mean and 95% CI) with time
    h_values = x_particles[:, :, 0]     # Particles are equally weighted, so we can just take the mean and quantiles without weights
    means = np.mean(h_values, axis=1)
    lower, higher = np.percentile(h_values, [2.5, 97.5], axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(means, label="Mean", color='blue')
    plt.fill_between(np.arange(len(means)), lower, higher, alpha=0.3, label="95% CI", color='blue')
    plt.plot(np.arange(1, len(h_true)+1), h_true, label="True value", color='black', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Log Volatility")
    plt.title("Filtering Distribution of Log Volatility over Time")
    plt.grid(True)
    plt.legend()
    plt.ylim(ylims)  # Use the same y-limits as the smoothing plot for better comparison
    plt.savefig(results_dir / "filtering_h.png")
    plt.close()

    # Plot filtering predicative distribution (mean and 95% CI) with time
    h_values = x_particles_pred[:, :, 0]     # Particles are equally weighted, so we can just take the mean and quantiles without weights
    means = np.mean(h_values, axis=1)
    lower, higher = np.percentile(h_values, [2.5, 97.5], axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(means)+1), means, label="Mean", color='blue')     # Start at 1, because we predict p(h_{t+1} | y_{1:t})
    plt.fill_between(np.arange(1, len(means)+1), lower, higher, alpha=0.3, label="95% CI", color='blue')    # Same here
    plt.plot(np.arange(1, len(h_true)+1), h_true, label="True value", color='black', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Log Volatility")
    plt.title("Predictive Distribution of Log Volatility over Time")
    plt.grid(True)
    plt.ylim(ylims)  # Use the same y-limits as the smoothing plot for better comparison
    plt.legend()
    plt.savefig(results_dir / "predictive_h.png")
    plt.close()


if __name__ == "__main__":
    main()