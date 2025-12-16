import numpy as np
from src.models.base import StateSpaceModel

class BootstrapParticleFilter:
    def __init__(self, model : StateSpaceModel, n_particles : int, resampler, rng=None):
        self.model = model
        self.N = n_particles
        self.resampler = resampler

    def run(self, y, theta):
        T = len(y)

        particles = [None] * self.N
        weights = np.full(self.N, 1.0 / self.N)

        history = []

        # ----- Initialization -----
        for i in range(self.N):
            particles[i] = self.model.sample_initial_state(theta)

        # ----- Main loop -----
        for t in range(T):
            # Propagation
            particles = [
                self.model.sample_transition(theta, p)
                for p in particles
            ]

            # Weighting
            weights = np.array([
                self.model.likelihood(y[t], theta, p)
                for p in particles
            ])

            weights_sum = np.sum(weights)

            # Normalization
            weights = weights / weights_sum

            # Store history
            history.append((particles.copy(), weights.copy()))

            # Resampling
            particles = self.resampler(particles, weights, self.model.rng)

        return history
