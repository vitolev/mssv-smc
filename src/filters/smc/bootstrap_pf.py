import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams

class BootstrapParticleFilter:
    """
    Bootstrap Particle Filter implementation for generic state-space models.
    """
    def __init__(self, model : StateSpaceModel, n_particles : int, resampler):
        self.model = model
        self.N = n_particles
        self.resampler = resampler

    def run(self, y, theta: StateSpaceModelParams):
        """
        Run the particle filter on observation sequence y.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : StateSpaceModelParams
            Model parameters.

        Returns
        -------
        history : list of tuples of size T
            Each element is (particles, weights, indices) at each time step t.
            - particles: StateSpaceModelState with batched N particles.
            - weights: np.ndarray of shape (N,) with normalized weights of the particles.
            - indices: np.ndarray of shape (N,) with resampling indices used to get from step t-1 to t. At t=0, this is an empty array.
        """
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)  # Sample initial particles
        weights = self.model.likelihood(y[0], theta, particles)          # Initial weights
        weights /= weights.sum()                                         # Normalize weights

        history.append((particles, weights, np.array([], dtype=int)))    # Store history    

        indices = self.resampler(weights, self.model.rng)                # Resampling
        particles = particles[indices]                                   # Resample particles    

        # ----- Main loop -----
        for t in range(1, T):
            # Propagation
            particles = self.model.sample_next_state(theta, particles)

            # Weighting
            weights = self.model.likelihood(y[t], theta, particles)
            weights_sum = weights.sum()
            weights /= weights_sum  # Normalize weights

            # Store history
            history.append((particles, weights, indices)) # No need to copy particles as new object is created each time

            # Resampling
            indices = self.resampler(weights, self.model.rng)
            particles = particles[indices]

        return history
