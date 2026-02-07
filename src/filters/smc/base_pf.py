from abc import ABC, abstractmethod
import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams

class ParticleFilter(ABC):
    """
    Base class for Particle Filters for generic state-space models.
    """
    def __init__(self, model : StateSpaceModel, n_particles : int, resampler):
        self.model = model
        self.N = n_particles
        self.resampler = resampler
        self.rng = model.rng

    @abstractmethod
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
        pass

    def smoothing_trajectories(self, history, n_traj=None):
        """
        Reconstruct full trajectories (smoothing samples) from particle filter history.

        Parameters
        ----------
        history : list of tuples
            Each element is (particles, weights, indices, logmarlik) at each time step.
            - particles: StateSpaceModelState with batched N particles
            - weights: np.ndarray of shape (N,)
            - indices: np.ndarray of shape (N,) mapping particles at t-1 -> particles at t
              (t=0 has empty indices)
            - logmarlik: float, the log marginal likelihood up to time t. At t=0, this is 0.
        n_traj : int or None
            Number of trajectories to sample. If None, returns all N trajectories.

        Returns
        -------
        trajectories : list of lists
            List of sampled trajectories. Each trajectory is a list of states over time. Trajectories are sampled according to the final weights, hence
            their contribution to the smoothing distribution is equally weighted.
            trajectories[i][t] is the state at time t of trajectory i.
        """
        T = len(history)
        N = len(history[0][1])

        if n_traj is None:
            n_traj = N

        # Sample final particles according to final weights
        final_weights = history[-1][1]
        final_indices = self.rng.choice(N, size=n_traj, p=final_weights)

        # Initialize storage
        trajectories = [ [None]*T for _ in range(n_traj) ]

        # Fill final time
        for i, idx in enumerate(final_indices):
            trajectories[i][T-1] = history[T-1][0][idx]

        # Trace backward
        for t in reversed(range(1, T)):
            indices = history[t][2]  # maps t-1 -> t
            for i in range(n_traj):
                parent_idx = indices[final_indices[i]]  # parent at t-1
                trajectories[i][t-1] = history[t-1][0][parent_idx]
                final_indices[i] = parent_idx  # update for next backward step

        return np.array(trajectories)