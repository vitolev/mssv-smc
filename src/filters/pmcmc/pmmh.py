import numpy as np
import arviz as az
from concurrent.futures import ProcessPoolExecutor

from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams

class PMMH_Chain:
    """
    A single PMMH chain that maintains the current parameters, trajectory, and marginal likelihood, and can perform PMMH iterations.
    """
    def __init__(
        self,
        pf: ParticleFilter,
        kwargs_model=None,
        kwargs_prior=None,
        proposal_param=None,
    ):
        """
        Parameters
        ----------
        pf : ParticleFilter
            A ParticleFilter instance to use for proposing trajectories and computing marginal likelihoods.
        kwargs_model : dict, optional
            Additional keyword arguments to pass about the model.
        proposal_param : dict, optional
            Additional keyword arguments to pass when proposing new parameters.
            For example, for MSSV model, step_mu, step_phi, step_sigma, step_P are needed to sample new parameters.
        kwargs_prior : dict, optional
            Additional keyword arguments to pass when sampling parameters from the prior.
        """
        self.pf = pf
        self.rng = pf.model.rng
        self.kwargs_model = kwargs_model if kwargs_model is not None else {}
        self.proposal_param = proposal_param if proposal_param is not None else {}
        self.kwargs_prior = kwargs_prior if kwargs_prior is not None else {}

        prior_cls = pf.model.prior_type
        self.prior = prior_cls(**self.kwargs_prior)
        proposal_cls = pf.model.proposal_type
        self.proposal = proposal_cls(self.proposal_param)

    def _run_pf_and_sample(self, y, theta: StateSpaceModelParams):
        """
        Run PF once and sample smoothing trajectory(ies).
        """
        history = self.pf.run(y, theta)

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
        Initialize the chain with a PF run by first sampling parameters from the prior and then running the PF to get an initial trajectory and marginal likelihood.
        """
        self.theta = self.prior.sample(self.rng, **self.kwargs_model)  # Sample initial parameters from the prior
        self.initial_params = self.theta.copy()                        # Store the initial parameters for later reference
        self.theta_vars = vars(self.theta)

        traj, logmarlik = self._run_pf_and_sample(self.y, self.theta)
        self.current_trajectory = traj
        self.current_logmarlik = logmarlik

    def _step(self):
        """
        Perform one PMMH iteration by proposing new parameters, running the PF to get a new trajectory and marginal likelihood, and then accepting or rejecting the proposal based on the MH acceptance probability.
        """
        theta_star = self.proposal.sample(self.rng, self.theta)  # Propose new parameters
        traj_star, logmarlik_star = self._run_pf_and_sample(self.y, theta_star)     # Run PF with proposed parameters

        # MH acceptance probability
        log_alpha = logmarlik_star - self.current_logmarlik + self.prior.logpdf(theta_star) - self.prior.logpdf(self.theta) + self.proposal.logpdf(self.theta, theta_star) - self.proposal.logpdf(theta_star, self.theta)

        if np.log(self.rng.uniform()) < log_alpha:
            self.theta = theta_star
            self.current_trajectory = traj_star
            self.current_logmarlik = logmarlik_star
            self.n_accepted += 1

        self.n_steps += 1
        return log_alpha

    def run(self, y, n_iter: int, burnin=0):
        """
        Run the PMMH algorithm.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        n_iter : int
            Number of PMMH iterations.
        burnin : int, optional
            Number of initial iterations to discard as burn-in. Must be less than n_iter. Default is 0.

        Returns
        -------
        samples : list 
            List of sampled trajectories after burn-in. 
        logmarliks : list 
            List of log marginal likelihoods corresponding to the sampled trajectories. 
        thetas : dict of lists 
            Dictionary where each key is a parameter name and the value is a list of sampled parameter values after burn-in. 
        logalphas : list 
            List of log acceptance probabilities for each iteration.
        """
        if burnin >= n_iter:
            raise ValueError("Burn-in must be less than the total number of iterations.")

        self.current_trajectory = None
        self.current_logmarlik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y

        self._initialize()

        for i in range(burnin):
            log_alpha = self._step()

        # The boundary iteration when we first collect samples
        logmarliks = []     # Initialize list to store log marginal likelihoods for each sampled trajectory
        thetas = {key: [] for key in self.theta_vars.keys()}  # Initialize dictionary to store sampled parameter values
        logalphas = []     # Initialize list to store log acceptance probabilities

        log_alpha = self._step()    # First step

        samples = self.current_trajectory
        logmarliks.append(self.current_logmarlik)
        for key in thetas.keys():
            thetas[key].append(getattr(self.theta, key))
        logalphas.append(log_alpha)

        # The second loop to run the remaining iterations and store samples after burn-in
        for i in range(burnin+1, n_iter):
            log_alpha = self._step()

            samples = [state.add(element) for state, element in zip(samples, self.current_trajectory)]
            logmarliks.append(self.current_logmarlik)
            for key in thetas.keys():
                thetas[key].append(getattr(self.theta, key))
            logalphas.append(log_alpha)

        return samples, logmarliks, thetas, logalphas
    
class ParticleMarginalMetropolisHastings:
    """
    Particle Marginal Metropolis-Hastings (PMMH) using a ParticleFilter.
    """

    def __init__(
        self,
        pf: ParticleFilter,
        kwargs_model=None,
        proposal_param=None,
        kwargs_prior=None,
    ):
        """
        Parameters
        ----------
        pf : ParticleFilter
            A ParticleFilter instance to use for proposing trajectories and computing marginal likelihoods.
        kwargs_model : dict, optional
            Additional keyword arguments to pass to the initialization of model parameters.
            For example, for MSSV model, num_regimes is needed to initialize the parameters.
        proposal_param : dict, optional
            Additional keyword arguments to pass when proposing new parameters.
            For example, for MSSV model, step_mu, step_phi, step_sigma, step_P are needed to sample new parameters.
        kwargs_prior : dict, optional
            Additional keyword arguments to pass to the initialization of prior.
        """
        self.pf = pf
        self.rng = pf.model.rng
        self.kwargs_model = kwargs_model if kwargs_model is not None else {}
        self.proposal_param = proposal_param if proposal_param is not None else {}
        self.kwargs_prior = kwargs_prior if kwargs_prior is not None else {}

    def _run_single_chain(self, seed, y, pf: ParticleFilter, kwargs_model, proposal_param, kwargs_prior, n_iter, burnin, chain_id):
        """
        Worker function for a single PMMH chain.
        """
        # Independent RNG for this chain
        rng = np.random.default_rng(seed)

        # Rebuild model with new RNG
        model_cls = pf.model.__class__
        model = model_cls(rng=rng)

        # Rebuild PF
        pf_chain = pf.__class__(
            model=model,
            n_particles=pf.N,
            resampler=pf.resampler
        )

        chain = PMMH_Chain(
            pf_chain,
            kwargs_model=kwargs_model,
            proposal_param=proposal_param,
            kwargs_prior=kwargs_prior
        )

        result = chain.run(y, n_iter=n_iter, burnin=burnin)
        acceptance_rate = chain.n_accepted / chain.n_steps if chain.n_steps > 0 else 0.0
        initial_parameters = chain.initial_params

        return result, acceptance_rate, initial_parameters, chain_id

    def run(self, y, n_iter: int, n_chain: int, burnin=0):
        """
        Run multiple PMMH chains in parallel and return the results.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        n_iter : int
            Number of PMMH iterations per chain.
        n_chain : int
            Number of parallel PMMH chains to run.
        burnin : int, optional
            Number of initial iterations to discard as burn-in. Default is 0.

        Returns
        -------
        all_results : list
            List of results from each chain. Each element is a tuple (samples, logmarliks, thetas, logalphas) for that chain. 
        acceptance_rates : list
            List of acceptance rates for each chain.
        """

        if n_chain == 1:
            # Run single chain without multiprocessing
            result, acceptance_rate, initial_parameters, _ = self._run_single_chain(
                seed=self.rng.integers(0, 1_000_000),
                y=y,
                pf=self.pf,
                kwargs_model=self.kwargs_model,
                proposal_param=self.proposal_param,
                kwargs_prior=self.kwargs_prior,
                n_iter=n_iter,
                burnin=burnin,
                chain_id=0,
            )
            return [result], [acceptance_rate], [initial_parameters]

        seeds = self.rng.integers(0, 1_000_000, size=n_chain)  # Generate random seeds for each chain

        with ProcessPoolExecutor(max_workers=n_chain) as executor:
            results = list(
                executor.map(
                    self._run_single_chain,
                    seeds,
                    [y] * n_chain,
                    [self.pf] * n_chain,
                    [self.kwargs_model] * n_chain,
                    [self.proposal_param] * n_chain,
                    [self.kwargs_prior] * n_chain,
                    [n_iter] * n_chain,
                    [burnin] * n_chain,
                    list(range(n_chain))
                )
            )

        all_results = [r[0] for r in results]          # PMMH samples etc.
        acceptance_rates = [r[1] for r in results]    # acceptance rates
        initial_parameters = [r[2] for r in results]  # initial parameters
        chain_ids = [r[3] for r in results]          # chain IDs

        return all_results, acceptance_rates, initial_parameters

    def to_inference_data(self, results):
        """
        Convert PMMH results to an ArviZ InferenceData object for analysis and visualization.

        Parameters
        ----------
        results : list
            List of results from each chain. Each element is a tuple (samples, logmarliks, thetas, logalphas) for that chain.

        Returns
        -------
        inference_data : arviz.InferenceData
            An ArviZ InferenceData object containing the log marginal likelihoods and parameter samples from all chains.
        """
        data = {}
        for chain_result in results:
            if "logmarliks" not in data:
                data["logmarliks"] = []
            _, logmarliks, thetas, _ = chain_result
            data["logmarliks"].append(logmarliks)
            for param_name, param_values in thetas.items():
                if param_name not in data:
                    data[param_name] = []
                data[param_name].append(param_values)
        
        for key in data:
            data[key] = np.array(data[key])
            
        return az.from_dict(data)