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
        history : list of tuples
            Each element is (particles, weights) at each time step.
            - particles: StateSpaceModelState with batched N particles
            - weights: np.ndarray of shape (N,)
        """
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)  # Sample initial particles
        weights = self.model.likelihood(y[0], theta, particles)          # Initial weights
        weights /= weights.sum()                                         # Normalize weights

        history.append((particles, weights))
        particles = self.resampler(particles, weights, self.model.rng)   # Resample initial particles

        # ----- Main loop -----
        for t in range(1, T):
            # Propagation
            particles = self.model.sample_next_state(theta, particles)

            # Weighting
            weights = self.model.likelihood(y[t], theta, particles)
            weights_sum = weights.sum()
            weights /= weights_sum  # Normalize weights

            # Store history
            history.append((particles, weights)) # No need to copy particles as new object is created each time

            # Resampling
            particles = self.resampler(particles, weights, self.model.rng)

        return history
