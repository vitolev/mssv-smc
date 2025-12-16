import numpy as np

def systematic_resampling(particles, weights, rng=None):
    """
    Systematic resampling for particle filters.

    Args:
        particles (list): Current particle states
        weights (np.ndarray): Normalized weights (sum to 1)
        rng (np.random.Generator or None)

    Returns:
        resampled_particles (list): New list of equally weighted particles
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(particles)
    
    # Cumulative sum of weights
    cdf = np.cumsum(weights)
    
    # Start point: uniform in [0, 1/N)
    u0 = rng.uniform(0, 1.0 / N)
    # Equally spaced points
    u = u0 + np.arange(N) / N

    # Find which particle each u corresponds to
    indices = np.searchsorted(cdf, u)

    # Resample particles
    resampled_particles = [particles[i] for i in indices]
    
    return resampled_particles
