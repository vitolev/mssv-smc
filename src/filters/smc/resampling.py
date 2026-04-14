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

def multinomial_resampling(weights: np.ndarray, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    cdf = np.cumsum(weights)

    u = rng.uniform(0.0, 1.0, size=N)
    indices = np.searchsorted(cdf, u)

    return indices

def stratified_resampling(weights: np.ndarray, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    cdf = np.cumsum(weights)

    # Stratified uniforms
    u = (np.arange(N) + rng.uniform(size=N)) / N

    indices = np.searchsorted(cdf, u)

    return indices

def residual_resampling(weights: np.ndarray, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)

    # Deterministic part
    Ns = np.floor(N * weights).astype(int)
    R = N - np.sum(Ns)

    indices = np.repeat(np.arange(N), Ns)

    if R > 0:
        # Residual weights
        residual = weights - Ns / N
        residual = residual / residual.sum()

        cdf = np.cumsum(residual)
        u = rng.uniform(size=R)
        extra_indices = np.searchsorted(cdf, u)

        indices = np.concatenate([indices, extra_indices])

    return indices

