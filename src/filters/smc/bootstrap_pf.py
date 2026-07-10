import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams
from src.filters.smc.base_pf import ParticleFilter

class BootstrapParticleFilter(ParticleFilter):
    """
    Bootstrap Particle Filter implementation for generic state-space models.
    """
    def __init__(self, model : StateSpaceModel, n_particles : int, resampler):
        super().__init__(model, n_particles, resampler)

    def run(self, y, theta: StateSpaceModelParams, only_last_step=False):
        """
        Run the particle filter on observation sequence y.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : StateSpaceModelParams
            Model parameters.
        only_last_step : bool, optional
            If True, only store and return the last step's particles and weights. Default is False.

        Returns
        -------
        history : list of tuples of size T+1 (or 1 if only_last_step is True).
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

        if not only_last_step:
            history.append((particles, weights, np.array([], dtype=int), 0.0))    # Store history    
        indices = np.arange(self.N)                                      # Initial indices

        logmarlik = 0.0  # initialize log marginal likelihood

        # ----- Main loop -----
        for t in range(T):
            # Propagation
            particles = self.model.sample_next_state(theta, particles)

            # Weighting
            log_weights = self.model.log_likelihood(y[t], theta, particles)  # log p(y_t | x_t)
            max_log_w = np.max(log_weights)
            weights = np.exp(log_weights - max_log_w)  # subtract max to avoid underflow
            weights_sum = np.sum(weights)
            weights /= weights_sum
            logmarlik += max_log_w + np.log(weights_sum / self.N)

            # Store history
            if not only_last_step:
                history.append((particles, weights, indices, logmarlik)) # No need to copy particles as new object is created each time

            # Resampling
            indices = self.resampler(weights, self.model.rng)
            particles = particles[indices]

        if only_last_step:
            return [(particles, weights, indices, logmarlik)]
        else:
            return history

    def run_conditional(self, y, theta: StateSpaceModelParams, x_ref: list, only_last_step=False):
        """
        Run conditional bootstrap particle filter given reference trajectory x_ref.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : StateSpaceModelParams
            Model parameters.
        x_ref : list of length T+1
            Reference trajectory to condition on. Must have length T+1, where T is the length of the observation sequence y.
        only_last_step : bool, optional
            If True, only store and return the last step's particles and weights. Default is False.

        Returns
        -------
        history : same format as in run(), but with the first particle forced to follow x_ref at each time step.
        """
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)

        # Force particle 0 to equal reference initial state
        particles[0] = x_ref[0]

        weights = np.ones(self.N) / self.N
        if not only_last_step:
            history.append((particles, weights, np.array([], dtype=int), 0.0))

        logmarlik = 0.0
        indices = np.arange(self.N)

        # ----- Main loop -----
        for t in range(T):
            # Propagation
            particles = self.model.sample_next_state(theta, particles)

            # Force reference particle
            particles[0] = x_ref[t + 1]

            # Weighting
            log_weights = self.model.log_likelihood(y[t], theta, particles)

            max_log_w = np.max(log_weights)
            weights = np.exp(log_weights - max_log_w)
            weights_sum = np.sum(weights)
            weights /= weights_sum

            logmarlik += max_log_w + np.log(weights_sum / self.N)

            # Store history
            if not only_last_step:
                history.append((particles, weights, indices, logmarlik))
            else:
                history = [(particles, weights, indices, logmarlik)]  # Only keep the last step
            
            # Resampling
            indices = self.resampler(weights, self.model.rng)

            # Force reference lineage
            indices[0] = 0

            particles = particles[indices]

        return history