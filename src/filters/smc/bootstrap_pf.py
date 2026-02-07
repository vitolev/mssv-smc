import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams
from src.filters.smc.base_pf import ParticleFilter

class BootstrapParticleFilter(ParticleFilter):
    """
    Bootstrap Particle Filter implementation for generic state-space models.
    """
    def __init__(self, model : StateSpaceModel, n_particles : int, resampler):
        super().__init__(model, n_particles, resampler)

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
        history : list of tuples of size T+1.
            Each element is (particles, weights, indices, logmarlik) at each time step t.
            - particles: StateSpaceModelState with batched N particles.
            - weights: np.ndarray of shape (N,) with normalized weights of the particles.
            - indices: np.ndarray of shape (N,) with resampling indices used to get from step t-1 to t. At t=0, this is an empty array.
            - logmarlik: float, the log marginal likelihood up to time t. At t=0, this is 0.
        """
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)  # Sample initial particles
        weights = np.ones(self.N) / self.N                               # Initialize weights uniformly

        history.append((particles, weights, np.array([], dtype=int), 0.0))    # Store history    
        indices = np.arange(self.N)                                      # Initial indices

        logmarlik = 0.0  # initialize log marginal likelihood

        # ----- Main loop -----
        for t in range(T):
            # Propagation
            particles = self.model.sample_next_state(theta, particles)

            # Weighting
            weights = self.model.likelihood(y[t], theta, particles)
            weights_sum = weights.sum()
            logmarlik += np.log(weights_sum / self.N)  # Update log marginal likelihood
            weights /= weights_sum  # Normalize weights

            # Store history
            history.append((particles, weights, indices, logmarlik)) # No need to copy particles as new object is created each time

            # Resampling
            indices = self.resampler(weights, self.model.rng)
            particles = particles[indices]

        return history