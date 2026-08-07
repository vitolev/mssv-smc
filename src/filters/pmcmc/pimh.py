import h5py
import numpy as np
from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState

from typing import List, Tuple

class ParticleIndependentMetropolisHastings:
    """
    Particle Independent Metropolis-Hastings (PIMH) using a ParticleFilter.
    """

    def __init__(
        self,
        pf: ParticleFilter,
    ):
        """
        Parameters
        ----------
        pf : ParticleFilter
            A ParticleFilter instance to use for proposing trajectories and computing marginal likelihoods.
        """
        self.pf = pf
        self.rng = pf.model.rng

    def _run_pf_and_sample(self) -> Tuple[List[StateSpaceModelState], float]:
        """
        Run PF once and sample smoothing trajectory(ies).
        """
        history = self.pf.run(self.y, self.theta)

        # final log marginal likelihood
        logmarlik = history[-1][3]

        # sample trajectory from smoothing distribution
        trajectory, _ = self.pf.smoothing_trajectories(
            history,
            n_traj=1,
        )

        return trajectory, logmarlik

    def _initialize(self):
        """
        Initialize the chain with a PF run.
        """
        traj, logmarlik = self._run_pf_and_sample()
        self.current_trajectory = traj
        self.current_logmarlik = logmarlik

    def _step(self) -> float:
        """
        Perform one PIMH iteration.
        """
        traj_star, logmarlik_star = self._run_pf_and_sample()

        # MH acceptance probability
        log_alpha = logmarlik_star - self.current_logmarlik

        if np.log(self.rng.uniform()) < log_alpha:
            self.current_trajectory = traj_star
            self.current_logmarlik = logmarlik_star
            self.n_accepted += 1

        self.n_steps += 1

        return log_alpha

    def _init_hdf5_chain(self, output_dir, n_samples: int, state_dim: int, T: int) -> h5py.File:
        h5_path = output_dir / f"chain.h5"

        h5f = h5py.File(h5_path, "w")

        h5f.create_dataset(
            "trajectories",
            shape=(n_samples, T, state_dim),
            dtype="f8",
            compression="gzip",
            compression_opts=4,
        )

        h5f.create_dataset(
            "logmarliks",
            shape=(n_samples,),
            dtype="f8",
            compression="gzip",
            compression_opts=4,
        )

        h5f.create_dataset(
            "logalphas",
            shape=(n_samples,),
            dtype="f8",
            compression="gzip",
            compression_opts=4,
        )
    
        return h5f
    
    def _write_chain_step(self, h5f, idx: int, trajectory: List[StateSpaceModelState], logmarlik: float, log_alpha: float):
        h5f["trajectories"][idx] = np.array([state.to_numpy() for state in trajectory]).reshape(h5f["trajectories"].shape[1:])  # reshape to (T, state_dim)
        h5f["logmarliks"][idx] = logmarlik
        h5f["logalphas"][idx] = log_alpha


    def run(self, y, theta: StateSpaceModelParams, n_iter, output_dir, burnin=0, logger=None):
        """
        Run the PIMH chain on given data and model parameters.

        Parameters
        ----------
        y : array-like
            Observations
        theta : StateSpaceModelParams
            The parameters of the state space model.
        n_iter : int
            Number of iterations to perform.
        burnin : int, optional
            Number of initial iterations to discard as burn-in (default is 0).
        output_dir : str
            Directory to save intermediate results or logs.
        logger : logging.Logger, optional
            Logger for logging information. If None, no logging is performed.

        Returns
        -------
        None. The results are stored in the HDF5 file specified by output_dir.
        """
        if burnin >= n_iter:
            raise ValueError("Burn-in period must be less than the total number of iterations.")
        if logger is not None:
            logger.info(f"Initializing PIMH with {n_iter} iterations and {burnin} burn-in iterations.")

        self.current_trajectory = None
        self.current_logmarlik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y
        self.theta = theta

        self._initialize()

        h5f = self._init_hdf5_chain(output_dir, n_iter - burnin, self.current_trajectory[0].to_numpy().shape[1], len(self.current_trajectory))

        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"PIMH initialized. Starting burn-in...")

        # Burn-in phase
        for i in range(burnin):
            log_alpha = self._step()
            if logger is not None:
                logger.info(f"Burn-in iteration {i + 1}/{burnin}, log_alpha: {log_alpha:.4f}")

        # First sample after burn-in
        log_alpha = self._step()
        self._write_chain_step(h5f, 0, self.current_trajectory, self.current_logmarlik, log_alpha)  # Store the first sample after burn-in

        # Remaining iterations
        for i in range(burnin + 1, n_iter):
            log_alpha = self._step()
            self._write_chain_step(h5f, i - burnin, self.current_trajectory, self.current_logmarlik, log_alpha)
            
            if logger is not None:
                logger.info(f"Iteration {i-burnin}/{n_iter-burnin}, log_alpha: {log_alpha:.4f}")

        h5f.attrs["acceptance_rate"] = self.n_accepted / self.n_steps if self.n_steps > 0 else 0.0
 
        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"PIMH chain completed. Acceptance rate: {h5f.attrs['acceptance_rate']:.4f}")
            logger.info(f"Results saved to {output_dir / f'chain.h5'}")

        h5f.close()

    @property
    def acceptance_rate(self):
        if self.n_steps == 0:
            return 0.0
        return self.n_accepted / self.n_steps
