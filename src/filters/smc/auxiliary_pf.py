import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams
from src.filters.smc.smoothing import get_smoothing_trajectories

class AuxiliaryParticleFilter:
    """
    Auxiliary Particle Filter implementation for generic state-space models.
    """
    def __init__(self, model: StateSpaceModel, n_particles: int, resampler):
        self.model = model
        self.N = n_particles
        self.resampler = resampler

    def run(self, y, theta: StateSpaceModelParams):
        """
        Run the auxiliary particle filter on observation sequence y.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : StateSpaceModelParams
            Model parameters.

        Returns
        -------
        history : list of tuples of size T+1.
            Each element is (particles, weights, indices, loglik) at each time step t.
            - particles: StateSpaceModelState with batched N particles.
            - weights: np.ndarray of shape (N,) with normalized weights of the particles.
            - indices: np.ndarray of shape (N,) with resampling indices used to get from step t-1 to t. At t=0, this is an empty array.
            - loglik: float, the log marginal likelihood up to time t. At t=0, this is 0.
        """
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)         # Sample initial particles
        weights = np.ones(self.N) / self.N                                      # Initialize weights uniformly
        history.append((particles, weights.copy(), np.array([], dtype=int), 0.0))    # Store history

        loglik = 0.0  # initialize log marginal likelihood

        # ----- Main loop -----
        for t in range(T):
            # ===== Auxiliary weights (look-ahead) =====
            predicted_states = self.model.expected_next_state(theta, particles)

            aux_weights = weights * self.model.likelihood(y[t], theta, predicted_states)
            aux_weights /= np.sum(aux_weights)

            # ===== Resample ancestors =====
            ancestor_indices = self.resampler(aux_weights, self.model.rng)
            ancestor_particles = particles[ancestor_indices]
            ancestor_predicted = predicted_states[ancestor_indices]
            
            # ===== Propagation =====
            particles = self.model.sample_next_state(theta, ancestor_particles)
            
            # ===== Weight correction =====
            w_num = self.model.likelihood(y[t], theta, particles)
            w_den = self.model.likelihood(y[t], theta, ancestor_predicted)
            weights_unnormalized = w_num / w_den

            # --- Update log marginal likelihood ---
            loglik += np.log(np.mean(weights_unnormalized))

            # --- Normalize weights ---
            weights = weights_unnormalized / weights_unnormalized.sum()

            # --- Store history ---
            history.append((particles, weights.copy(), ancestor_indices, loglik))

        return history
    
    def smoothing_trajectories(self, history, n_traj=None):
        """
        Reconstruct full trajectories (smoothing samples) from particle filter history.

        Parameters
        ----------
        history : list of tuples
            Each element is (particles, weights, indices, loglik) at each time step.
            - particles: StateSpaceModelState with batched N particles
            - weights: np.ndarray of shape (N,)
            - indices: np.ndarray of shape (N,) mapping particles at t-1 -> particles at t
              (t=0 has empty indices)
            - loglik: float, the log marginal likelihood up to time t. At t=0, this is 0.
        n_traj : int or None
            Number of trajectories to sample. If None, returns all N trajectories.

        Returns
        -------
        trajectories : list of lists
            List of sampled trajectories. Each trajectory is a list of states over time. Trajectories are sampled according to the final weights, hence
            their contribution to the smoothing distribution is equally weighted.
            trajectories[i][t] is the state at time t of trajectory i.
        """
        return get_smoothing_trajectories(history, n_traj=n_traj, rng=self.model.rng)
