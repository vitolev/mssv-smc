import numpy as np

def systematic_resampling(weights: np.ndarray, rng: np.random.Generator = None):
    """
    Systematic resampling for particle filters.

    Parameters
    ----------
        weights: np.ndarray
            Normalized weights (sum to 1), shape (N,)
        rng: np.random.Generator or None

    Returns
    -------
        indices:
            Indices of resampled particles, shape (N,)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    cdf = np.cumsum(weights)

    # Systematic points
    u0 = rng.uniform(0, 1.0 / N)
    u = u0 + np.arange(N) / N

    # Indices of particles to pick
    indices = np.searchsorted(cdf, u)

    return indices

