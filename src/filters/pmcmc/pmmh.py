import numpy as np
from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams

class ParticleMarginalMetropolisHastings:
    """
    Particle Marginal Metropolis-Hastings (PMMH) using a ParticleFilter.
    """

    def __init__(
        self,
        pf: ParticleFilter,
        kwargs_for_params=None,
        kwargs_for_sampling=None,
    ):
        """
        Parameters
        ----------
        pf : ParticleFilter
            A ParticleFilter instance to use for proposing trajectories and computing marginal likelihoods.
        kwargs_for_params : dict, optional
            Additional keyword arguments to pass to the initialization of parameters. 
            For example, for MSSV model, num_regimes is needed to initialize the parameters.
        kwargs_for_sampling : dict, optional
            Additional keyword arguments to pass when proposing new parameters.
            For example, for MSSV model, step_mu, step_phi, step_sigma, step_P are needed to sample new parameters.
        """
        self.pf = pf
        self.rng = pf.model.rng
        self.kwargs_for_params = kwargs_for_params if kwargs_for_params is not None else {}
        self.kwargs_for_sampling = kwargs_for_sampling if kwargs_for_sampling is not None else {}

    def _run_pf_and_sample(self, y, theta: StateSpaceModelParams):
        """
        Run PF once and sample smoothing trajectory(ies).
        """
        history = self.pf.run(y, theta)

        # final log marginal likelihood
        logmarlik = history[-1][3]

        # sample trajectory from smoothing distribution
        trajectories = self.pf.smoothing_trajectories(
            history,
            n_traj=1,
        )

        # for standard PMMH, we keep a single trajectory
        trajectory = trajectories[0]

        return trajectory, logmarlik
    
    def _initialize(self):
        """
        Initialize the chain with a PF run by first sampling parameters from the prior and then running the PF to get an initial trajectory and marginal likelihood.
        """
        params_class = self.pf.model.params_type
        self.theta = params_class(self.rng, **self.kwargs_for_params) # Initialize parameters by prior sampling
        self.theta_vars = vars(self.theta)

        traj, logmarlik = self._run_pf_and_sample(self.y, self.theta)
        self.current_trajectory = traj
        self.current_logmarlik = logmarlik

    def _step(self):
        """
        Perform one PMMH iteration by proposing new parameters, running the PF to get a new trajectory and marginal likelihood, and then accepting or rejecting the proposal based on the MH acceptance probability.
        """
        theta_star = self.theta.sample_transition(self.rng, **self.kwargs_for_sampling)  # Propose new parameters
        traj_star, logmarlik_star = self._run_pf_and_sample(self.y, theta_star)     # Run PF with proposed parameters

        # MH acceptance probability
        log_alpha = logmarlik_star - self.current_logmarlik + theta_star.log_prior_density() - self.theta.log_prior_density() + theta_star.log_transition_density(self.theta, **self.kwargs_for_sampling) - self.theta.log_transition_density(theta_star, **self.kwargs_for_sampling)

        if np.log(self.rng.uniform()) < log_alpha:
            self.theta = theta_star
            self.current_trajectory = traj_star
            self.current_logmarlik = logmarlik_star
            self.n_accepted += 1

        self.n_steps += 1
        return log_alpha

    def run(self, y, n_iter: int, burnin=0, verbose=False):
        """
        Run the PMMH algorithm.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        n_iter : int
            Number of PMMH iterations.
        burnin : int, optional
            Number of initial iterations to discard as burn-in. Default is 0.
        verbose : bool, optional
            If True, print progress. Default is False.

        """
        self.current_trajectory = None
        self.current_logmarlik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y

        self._initialize()

        samples = []
        logmarliks = []
        thetas = {key: [] for key in self.theta_vars.keys()}  # Store parameter values separately
        alphas = []

        for i in range(n_iter):
            if verbose and (i+1) % 10000 == 0:
                print(f"Iteration {i+1}/{n_iter} - Acceptance Rate: {self.n_accepted/self.n_steps:.3f}")

            log_alpha = self._step()

            if i >= burnin:
                samples.append(self.current_trajectory)
                logmarliks.append(self.current_logmarlik)
                for key in thetas.keys():
                    thetas[key].append(getattr(self.theta, key))
                alphas.append(np.exp(log_alpha))

        return np.array(samples), np.array(logmarliks), thetas, np.array(alphas)
    