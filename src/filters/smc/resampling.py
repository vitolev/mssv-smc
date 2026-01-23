import numpy as np
from src.models.mssv import MSSVModelState
from src.models.lgm import LGModelState
from src.models.base import StateSpaceModelState

def systematic_resampling(particles: StateSpaceModelState, weights: np.ndarray, rng: np.random.Generator = None) -> StateSpaceModelState:
    """
    Systematic resampling for particle filters.

    Parameters
    ----------
        particles: StateSpaceModelState
            State of particles to be resampled. Only support LGModelState and MSSVModelState. 
        weights: np.ndarray
            Normalized weights (sum to 1)
        rng: np.random.Generator or None

    Returns
    -------
        resampled_particles: StateSpaceModelState
            New state of equally weighted particles
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

    # Resample particles (vectorized)
    if isinstance(particles, LGModelState):
        x_resampled = particles.x_t[indices].copy()
        return LGModelState(x_resampled)

    elif isinstance(particles, MSSVModelState):
        h_resampled = particles.h_t[indices].copy()
        s_resampled = particles.s_t[indices].copy()
        return MSSVModelState(h_resampled, s_resampled)

    else:
        raise ValueError("Unknown particle state type")
