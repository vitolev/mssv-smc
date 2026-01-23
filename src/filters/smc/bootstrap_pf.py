import numpy as np
from src.models.base import StateSpaceModel

class BootstrapParticleFilter:
    def __init__(self, model : StateSpaceModel, n_particles : int, resampler):
        self.model = model
        self.N = n_particles
        self.resampler = resampler

    def run(self, y, theta):
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = [self.model.sample_initial_state(theta) for _ in range(self.N)]     # Sample initial particles
        weights = np.array([self.model.likelihood(y[0], theta, p) for p in particles])  # Initial weights
        weights /= weights.sum()                                                        # Normalize weights
        history.append((particles.copy(), weights.copy()))
        particles = self.resampler(particles, weights, self.model.rng)                  # Resample initial particles

        # ----- Main loop -----
        for t in range(1, T):
            # Propagation
            particles = [
                self.model.sample_next_state(theta, p)
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
