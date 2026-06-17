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
    T = 1000
    script_dir = Path(__file__).resolve().parent
    script_dir = script_dir / f"{K}_regime_T_{T}"
    output_dir = script_dir / "output"
    results_dir = script_dir / "results"
    data_dir = ROOT_DIR / 'data'

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
    true_theta = true_theta.to_vector()

    # Get number of chain files from output_dir
    chain_files = list(output_dir.glob("chain_*.h5"))
    n_chains = len(chain_files)
    print(f"Found {n_chains} chain files.")
    print(output_dir)

    # Dictionary to store chain data
    chains_data = {}

    for i in range(n_chains):
        print(f"Processing chain {i+1}/{n_chains}...")
        with h5py.File(output_dir / f"chain_{i}.h5", "r") as h5f:
            acceptance_rate = h5f.attrs["acceptance_rate"]
            initial_theta = h5f.attrs["initial_parameters"]
            thetas = h5f["thetas"][:]
            logmarliks = h5f["logmarliks"][:]
            logalphas = h5f["logalphas"][:]
            trajectories = h5f["trajectories"][:]

            chains_data[i] = {
                "acceptance_rate": acceptance_rate,
                "initial_theta": initial_theta,
                "thetas": thetas,
                "logmarliks": logmarliks,
                "logalphas": logalphas,
                "trajectories": trajectories
            }

    n_params = chains_data[0]["thetas"].shape[1]

    parameter_names = generate_names(n_params)
    print("Parameter names:", parameter_names)

    ######################
    # Plotting traceplots
    ######################
    plt.figure(figsize=(12, 8))
    for i in range(n_chains):
        plt.plot(chains_data[i]["logmarliks"], label=f"Chain {i+1}", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Log Marginal Likelihood")
    plt.title("Log Marginal Likelihood Trace")
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / "logmarlik_trace.png")
    plt.close()

    for param_idx in range(n_params):
        plt.figure(figsize=(12, 8))
        for i in range(n_chains):
            plt.plot(chains_data[i]["thetas"][:, param_idx], label=f"Chain {i+1}", alpha=0.7)
        plt.xlabel("Iteration")
        plt.ylabel(parameter_names[param_idx])
        plt.title(f"Traceplot for {parameter_names[param_idx]}")
        plt.legend()
        plt.grid()
        plt.savefig(results_dir / f"{parameter_names[param_idx]}_trace.png")
        plt.close()

    ############################################
    # Plotting histograms of posterior samples
    ############################################
    for param_idx in range(n_params):
        plt.figure(figsize=(12, 8))
        samples = []
        for i in range(n_chains):
            samples.extend(chains_data[i]["thetas"][:, param_idx])
        mean = np.mean(samples)
        lower, higher = np.percentile(samples, [2.5, 97.5])
        plt.hist(samples, bins=int(np.sqrt(len(samples))), density=True)
        plt.axvline(true_theta[param_idx], color='green', linestyle='--', label='True value')
        plt.axvline(mean, color='red', linestyle='--', label='Mean')
        plt.axvline(lower, color='orange', linestyle='--', label='2.5%')
        plt.axvline(higher, color='orange', linestyle='--', label='97.5%')
        plt.xlabel(parameter_names[param_idx])
        plt.ylabel("Density")
        plt.title(f"Posterior distribution for {parameter_names[param_idx]}")
        plt.grid()
        plt.legend()
        plt.savefig(results_dir / f"{parameter_names[param_idx]}_hist.png")
        plt.close()

    ######################################
    # Plotting mean trajectories
    ######################################
    h_values_all_chains = []
    s_values_all_chains = []
    for i in range(n_chains):
        h_values = chains_data[i]["trajectories"][:, :, 0]    # shape (M, T+1)
        h_values_all_chains.append(h_values)
        # Construct back the one-hot encoded regime indicators
        s_values = chains_data[i]["trajectories"][:, :, 1:]   # shape (M, T+1, K)
        s_values_all_chains.append(s_values)

    # Compute mean trajectory across all chains
    h_values = np.concatenate(h_values_all_chains, axis=0)
    mean_h_values = np.mean(h_values, axis=0)    # shape (T+1,)
    lower, higher = np.percentile(h_values, [2.5, 97.5], axis=0)

    plt.figure(figsize=(12, 8))
    plt.plot(mean_h_values, label="Mean h_t", color='blue')
    plt.fill_between(np.arange(len(mean_h_values)), lower, higher, color='blue', alpha=0.3, label="95% CI")
    plt.plot(np.arange(1, len(h_true)+1), h_true, label="True h_t", color='black', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Log Volatility")
    plt.title("Mean Trajectory of h_t")
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / "mean_h_trajectory.png")
    plt.close()
    

if __name__ == "__main__":
    main()