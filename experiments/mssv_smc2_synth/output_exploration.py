# General imports
import ast
import logging
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

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
    script_dir = Path(__file__).resolve().parent
    script_dir = script_dir / "T_2000_test_5"
    output_dir = script_dir / "output"
    results_dir = script_dir / "results"
    with h5py.File(output_dir / "history.h5", "r") as h5f:
        theta = h5f["theta"][:]
        logweights = h5f["logweights"][:]

    T_plus_1, N_theta, n_params = theta.shape

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
            lo, hi = weighted_credible_interval(
                x,
                w,
                alpha=0.05
            )
            lower[t, j] = lo
            upper[t, j] = hi

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    time = np.arange(T_plus_1-1)  # Exclude t=0 since it's the prior

    for j in range(n_params):
        plt.figure(figsize=(10, 5))
        plt.plot(time,means[1:, j],label="Mean")                                        # Exclude t=0 since it's the prior
        plt.fill_between(time,lower[1:, j],upper[1:, j],alpha=0.3,label="95% CI")       # Exclude t=0 since it's the prior

        plt.xlabel("Time")
        plt.ylabel(f"{parameter_names[j]}")
        plt.title(f"Posterior of Parameter {parameter_names[j]} over Time")

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(results_dir / f"{parameter_names[j]}.png")

if __name__ == "__main__":
    main()