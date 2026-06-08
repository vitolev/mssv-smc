import numpy as np

def systematic_resampling(weights: np.ndarray, rng: np.random.Generator = None, N_out: int = None):
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
            Indices of resampled particles, shape (N_out,)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    if N_out is None:
        N_out = N
    cdf = np.cumsum(weights)

    # Systematic points
    u0 = rng.uniform(0, 1.0 / N_out)
    u = u0 + np.arange(N_out) / N_out

    # Indices of particles to pick
    indices = np.searchsorted(cdf, u)

    return indices

def multinomial_resampling(weights: np.ndarray, rng: np.random.Generator = None, N_out: int = None):
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    if N_out is None:
        N_out = N
    cdf = np.cumsum(weights)

    u = rng.uniform(0.0, 1.0, size=N_out)
    indices = np.searchsorted(cdf, u)

    return indices

def stratified_resampling(weights: np.ndarray, rng: np.random.Generator = None, N_out: int = None):
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    if N_out is None:
        N_out = N
    cdf = np.cumsum(weights)

    # Stratified uniforms
    u = (np.arange(N_out) + rng.uniform(size=N_out)) / N_out

    indices = np.searchsorted(cdf, u)

    return indices

def residual_resampling(weights: np.ndarray, rng: np.random.Generator = None, N_out: int = None):
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    if N_out is None:
        N_out = N

    # Deterministic offspring counts
    Ns = np.floor(N_out * weights).astype(int)
    R = int(N_out - Ns.sum())

    indices = np.repeat(np.arange(N), Ns)

    if R > 0:
        # Residual mass in count space (correct for any N_out)
        residual_counts = N_out * weights - Ns
        residual_probs = residual_counts / residual_counts.sum()  # sums to 1

        cdf = np.cumsum(residual_probs)
        u = rng.uniform(0.0, 1.0, size=R)
        extra_indices = np.searchsorted(cdf, u, side="right")

        indices = np.concatenate([indices, extra_indices])

    return indices

