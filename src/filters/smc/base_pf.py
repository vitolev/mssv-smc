from abc import ABC, abstractmethod
import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from typing import List

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

    @abstractmethod
    def run_conditional(self, y, theta: StateSpaceModelParams, x_ref: List[StateSpaceModelState]):
        """
        Run the particle filter on observation sequence y, conditional on a fixed trajectory x_ref.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : StateSpaceModelParams
            Model parameters.
        x_ref : array-like, shape (T+1,)
            Reference trajectory to condition on. Must be of length T+1, where T is the length of y. Each element of trajectory is a StateSpaceModelState at that time.

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
        trajectories : list of StateSpaceModelState
            A list of length T+1, where each element is a StateSpaceModelState containing the sampled states for that time step across the n_traj trajectories.
        n_traj : int
            The number of trajectories returned (equal to n_traj if n_traj is not None, else equal to N).
        """
        T = len(history)    # history actually has length T+1, where T is the number of time steps in the original data sequence y. +1 comes from the initial step t=0 with particles from the prior. 
        N = len(history[0][1])

        if n_traj is None:
            n_traj = N

        # Sample final particles according to final weights
        final_weights = history[-1][1]
        final_indices = self.rng.choice(N, size=n_traj, p=final_weights)

        # Initialize storage
        trajectories = [None]*T

        # Fill final time
        trajectories[T-1] = history[T-1][0][final_indices]

        # Trace backward
        for t in reversed(range(1, T)):
            indices = history[t][2]  # maps t-1 -> t
            parent_idx = indices[final_indices]  # parent at t-1
            trajectories[t-1] = history[t-1][0][parent_idx]
            final_indices = parent_idx  # update for next backward step

        return trajectories, n_traj