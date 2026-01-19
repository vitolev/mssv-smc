import numpy as np
from src.models.base import StateSpaceModel

class AuxiliaryParticleFilter:
    def __init__(self, model: StateSpaceModel, n_particles: int, resampler):
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

            # ===== Auxiliary weights (look-ahead) =====
            predicted_states = [
                self.model.expected_next_state(theta, p)
                for p in particles
            ]

            aux_weights = weights * np.array([
                self.model.likelihood(y[t], theta, x_hat)
                for x_hat in predicted_states
            ])

            aux_weights /= np.sum(aux_weights)

            # ===== Resample ancestors =====
            ancestor_indices = self.resampler(
                list(range(self.N)),
                aux_weights,
                self.model.rng
            )

            # Cache predicted states for chosen ancestors
            ancestor_predicted = [
                predicted_states[idx] for idx in ancestor_indices
            ]

            # ===== Propagation =====
            new_particles = []
            for idx in ancestor_indices:
                x_prev = particles[idx]
                x_new = self.model.sample_next_state(theta, x_prev)
                new_particles.append(x_new)

            particles = new_particles

            # ===== Weight correction =====
            weights = np.array([
                self.model.likelihood(y[t], theta, particles[i]) /
                self.model.likelihood(y[t], theta, ancestor_predicted[i])
                for i in range(self.N)
            ])

            weights /= np.sum(weights)

            history.append((particles.copy(), weights.copy()))

        return history
