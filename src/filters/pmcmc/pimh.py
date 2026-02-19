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
        logmarlik = history[-1][3]

        # sample trajectory from smoothing distribution
        trajectory = self.pf.smoothing_trajectories(
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

    def _step(self):
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
        samples : list of StateSpaceModelState, size T+1
            List of StateSpaceModelState instances representing trajectory values at each time step.
            Each StateSpaceModelState contains the (n_iter - burnin) array of sampled states at that time step across iterations.
        logmarliks : list
            List of corresponding log marginal likelihoods.
        """
        self.current_trajectory = None
        self.current_logmarlik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y
        self.theta = theta

        self._initialize()

        samples = self.current_trajectory       # array of size T+1 for the initial trajectory
        logmarliks = [self.current_logmarlik]

        for i in range(n_iter):
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{n_iter}, Acceptance Rate: {self.acceptance_rate:.3f}")

            self._step()

            if i >= burnin:
                samples = [state.add(element) for state, element in zip(samples, self.current_trajectory)]
                logmarliks.append(self.current_logmarlik)

        return samples, logmarliks

    @property
    def acceptance_rate(self):
        if self.n_steps == 0:
            return 0.0
        return self.n_accepted / self.n_steps
