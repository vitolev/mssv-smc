import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(thetas, weights,results_dir):
    """
    Diagnostics histogram plotting for SMC^2 results.

    Parameters
    ----------
    thetas : list
        List of theta values.
    weights : list
        List of corresponding weights for the theta particles.
    results_dir : Path
        Directory where to save the plots.
    """

    # Get parameter names from theta object
    param_keys = vars(thetas[0]).keys()

    for key in param_keys:
        # Collect parameter across particles
        values = np.array([
            getattr(t, key)
            for t in thetas
        ])
        # --------------------------------
        # Case 1: scalar parameter (N,)
        # --------------------------------
        if values.ndim == 1:
            n_bins = int(np.sqrt(len(values)))
            plt.figure()
            plt.hist(values, bins=n_bins, density=True, alpha=0.7, weights=weights)
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
                n_bins = int(np.sqrt(len(values[:, k])))
                plt.figure()
                plt.hist(values[:, k], bins=n_bins, density=True, alpha=0.7, weights=weights)
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
                    all_values = values[:, i, j]
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

                    plt.hist(all_values, bins=100, range=(vmin, vmax), density=True, alpha=0.7, weights=weights)
                    plt.title(f'Posterior Distribution: {key}[{i},{j}]')
                    plt.xlabel(f'{key}[{i},{j}]')
                    plt.ylabel('Density')
                    plt.grid()
                    plt.savefig(results_dir / f"{key}_{i}_{j}_hist.png")
                    plt.close()