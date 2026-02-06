import numpy as np
from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams

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

    def _run_pf_and_sample(self):
        """
        Run PF once and sample smoothing trajectory(ies).
        """
        history = self.pf.run(self.y, self.theta)

        # final log marginal likelihood
        loglik = history[-1][3]

        # sample trajectory from smoothing distribution
        trajectories = self.pf.smoothing_trajectories(
            history,
            n_traj=1,
        )

        # for standard PIMH, we keep a single trajectory
        trajectory = trajectories[0]

        return trajectory, loglik

    def _initialize(self):
        """
        Initialize the chain with a PF run.
        """
        traj, loglik = self._run_pf_and_sample()
        self.current_trajectory = traj
        self.current_loglik = loglik

    def _step(self):
        """
        Perform one PIMH iteration.
        """
        traj_star, loglik_star = self._run_pf_and_sample()

        # MH acceptance probability
        log_alpha = loglik_star - self.current_loglik

        if np.log(self.rng.uniform()) < log_alpha:
            self.current_trajectory = traj_star
            self.current_loglik = loglik_star
            self.n_accepted += 1
            accepted = True
        else:
            accepted = False

        self.n_steps += 1
        return accepted

    def run(self, y, theta: StateSpaceModelParams, n_iter, burnin=0, verbose=False):
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
        verbose : bool, optional
            Whether to print progress (default is False).

        Returns
        -------
        samples : list
            List of smoothing trajectories.
        """
        self.current_trajectory = None
        self.current_loglik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y
        self.theta = theta

        self._initialize()

        samples = []
        logliks = []

        for i in range(n_iter):
            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{n_iter}, Acceptance Rate: {self.acceptance_rate:.3f}")

            self._step()

            if i >= burnin:
                samples.append(self.current_trajectory)
                logliks.append(self.current_loglik)

        return np.array(samples), np.array(logliks)

    @property
    def acceptance_rate(self):
        if self.n_steps == 0:
            return 0.0
        return self.n_accepted / self.n_steps
