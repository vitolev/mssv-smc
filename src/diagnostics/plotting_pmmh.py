import numpy as np
import matplotlib.pyplot as plt

def plot_traceplots(results, results_dir):
    """
    Diagnostics traceplotting for PMMH results.

    Parameters
    ----------
    results : list
        List of results from each chain. Each element is a tuple (samples, logmarliks, thetas, logalphas) for that chain. 
    results_dir : Path
        Directory where to save the plots.
    """
    # Plot log marginal likelihood traceplot
    plt.figure(figsize=(12, 8))
    for chain in range(len(results)):
        samples, logmarliks, thetas, alphas = results[chain]
        plt.plot(logmarliks, label=f"Chain {chain+1}", alpha=0.7)
        
    plt.xlabel("Iteration")
    plt.ylabel("Log Marginal Likelihood")
    plt.title("Log Marginal Likelihood Trace")
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / "logmarlik_trace.png")
    plt.close()

    post_samples = []

    for chain in range(len(results)):
        _, _, thetas, _ = results[chain]
        thetas_post = {key: np.array(values) 
                    for key, values in thetas.items()}
        post_samples.append(thetas_post)

    # Get parameter names
    param_keys = post_samples[0].keys()

    for key in param_keys:

        values = post_samples[0][key]

        # --------------------------------
        # Case 1: scalar parameter (N,)
        # --------------------------------
        if values.ndim == 1:
            plt.figure()
            for chain in range(len(results)):
                plt.plot(post_samples[chain][key],
                        label=f"Chain {chain+1}", alpha=0.7)
            plt.title(f'Traceplot: {key}')
            plt.xlabel('Iteration')
            plt.ylabel(key)
            plt.legend()
            plt.grid()
            plt.savefig(results_dir / f"{key}_trace.png")
            plt.close()

        # --------------------------------
        # Case 2: vector parameter (N, K)
        # --------------------------------
        elif values.ndim == 2:
            N, K = values.shape
            for k in range(K):
                plt.figure()
                for chain in range(len(results)):
                    plt.plot(post_samples[chain][key][:, k],
                            label=f"Chain {chain+1}", alpha=0.7)
                plt.title(f'Traceplot: {key}[{k}]')
                plt.xlabel('Iteration')
                plt.ylabel(f'{key}[{k}]')
                plt.legend()
                plt.grid()
                plt.savefig(results_dir / f"{key}_{k}_trace.png")
                plt.close()

        # --------------------------------------
        # Case 3: matrix parameter (N, K, K)
        # --------------------------------------
        elif values.ndim == 3:
            N, K, K2 = values.shape
            for i in range(K):
                for j in range(K2):
                    plt.figure()
                    for chain in range(len(results)):
                        plt.plot(post_samples[chain][key][:, i, j],
                                label=f"Chain {chain+1}", alpha=0.7)
                    plt.title(f'Traceplot: {key}[{i},{j}]')
                    plt.xlabel('Iteration')
                    plt.ylabel(f'{key}[{i},{j}]')
                    plt.legend()
                    plt.grid()
                    plt.savefig(results_dir / f"{key}_{i}_{j}_trace.png")
                    plt.close()

def plot_histograms(results, results_dir):
    """
    Diagnostics histogram plotting for PMMH results.

    Parameters
    ----------
    results : list
        List of results from each chain. Each element is a tuple (samples, logmarliks, thetas, logalphas) for that chain. 
    results_dir : Path
        Directory where to save the plots.
    """
    post_samples = []

    for chain in range(len(results)):
        _, _, thetas, _ = results[chain]
        thetas_post = {key: np.array(values) 
                    for key, values in thetas.items()}
        post_samples.append(thetas_post)

    # Get parameter names
    param_keys = post_samples[0].keys()

    for key in param_keys:

        values = post_samples[0][key]

        # --------------------------------
        # Case 1: scalar parameter (N,)
        # --------------------------------
        if values.ndim == 1:
            plt.figure()
            all_values = np.concatenate([post_samples[chain][key] for chain in range(len(results))])
            plt.hist(all_values, bins='auto', density=True, alpha=0.7)
            plt.title(f'Posterior Distribution: {key}')
            plt.xlabel(key)
            plt.ylabel('Density')
            plt.grid()
            plt.savefig(results_dir / f"{key}_hist.png")
            plt.close()

        # --------------------------------
        # Case 2: vector parameter (N, K)
        # --------------------------------
        elif values.ndim == 2:
            N, K = values.shape
            for k in range(K):
                plt.figure()
                all_values = np.concatenate([post_samples[chain][key][:, k] for chain in range(len(results))])
                plt.hist(all_values, bins='auto', density=True, alpha=0.7)
                plt.title(f'Posterior Distribution: {key}[{k}]')
                plt.xlabel(f'{key}[{k}]')
                plt.ylabel('Density')
                plt.grid()
                plt.savefig(results_dir / f"{key}_{k}_hist.png")
                plt.close()

        # --------------------------------------
        # Case 3: matrix parameter (N, K, K)
        # --------------------------------------
        elif values.ndim == 3:
            N, K, K2 = values.shape
            for i in range(K):
                for j in range(K2):
                    all_values = np.concatenate([post_samples[chain][key][:, i, j] for chain in range(len(results))])
                    plt.figure()
                    vmin = np.min(all_values)
                    vmax = np.max(all_values)
                    data_range = vmax - vmin

                    # enforce minimum range
                    if data_range < 1e-8:
                        center = np.mean(all_values)
                        half_width = 0.5 
                        vmin = center - half_width
                        vmax = center + half_width

                    plt.hist(all_values, bins=100, range=(vmin, vmax), density=True, alpha=0.7)
                    plt.title(f'Posterior Distribution: {key}[{i},{j}]')
                    plt.xlabel(f'{key}[{i},{j}]')
                    plt.ylabel('Density')
                    plt.grid()
                    plt.savefig(results_dir / f"{key}_{i}_{j}_hist.png")
                    plt.close()