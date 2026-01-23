import numpy as np
from src.models.base import StateSpaceModel
from src.models.lgm import LGModelState
from src.models.mssv import MSSVModelState

class AuxiliaryParticleFilter:
    def __init__(self, model: StateSpaceModel, n_particles: int, resampler):
        self.model = model
        self.N = n_particles
        self.resampler = resampler

    def run(self, y, theta):
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)  # Sample initial particles
        weights = self.model.likelihood(y[0], theta, particles)          # Initial weights
        weights /= weights.sum()                                         # Normalize weights

        history.append((particles, weights.copy()))

        # ----- Main loop -----
        for t in range(1, T):
            # ===== Auxiliary weights (look-ahead) =====
            predicted_states = self.model.expected_next_state(theta, particles)

            aux_weights = weights * self.model.likelihood(y[t], theta, predicted_states)
            aux_weights /= np.sum(aux_weights)

            # ===== Resample ancestors =====
            ancestor_indices = self.resampler(
                np.arange(self.N),
                aux_weights,
                self.model.rng
            )

            if isinstance(particles, LGModelState):
                x_prev = particles.x_t[ancestor_indices]
                ancestor_particles = LGModelState(x_prev)
                x_prev_pred = predicted_states.x_t[ancestor_indices]
                ancestor_predicted = LGModelState(x_prev_pred)
            elif isinstance(particles, MSSVModelState):
                h_prev = particles.h_t[ancestor_indices]
                s_prev = particles.s_t[ancestor_indices]
                ancestor_particles = MSSVModelState(h_prev, s_prev)
                h_prev_pred = predicted_states.h_t[ancestor_indices]
                s_prev_pred = predicted_states.s_t[ancestor_indices]
                ancestor_predicted = MSSVModelState(h_prev_pred, s_prev_pred)
            else:
                raise ValueError("Unknown particle state type")
            
            # ===== Propagation =====
            particles = self.model.sample_next_state(theta, ancestor_particles)
            
            # ===== Weight correction =====
            w_num = self.model.likelihood(y[t], theta, particles)
            w_den = self.model.likelihood(y[t], theta, ancestor_predicted)
            weights = w_num / w_den
            weights /= np.sum(weights)

            history.append((particles, weights.copy()))

        return history
