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
        trajectory = self.pf.smoothing_trajectories(
            history,
            n_traj=1,
        )

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
            Number of initial iterations to discard as burn-in. Default is 0.

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
        self.current_trajectory = None
        self.current_logmarlik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y

        self._initialize()

        samples = self.current_trajectory       # array of size T+1 for the initial trajectory
        logmarliks = [self.current_logmarlik]
        thetas = {key: [] for key in self.theta_vars.keys()}  # Store parameter values separately
        for key in self.theta_vars.keys():
            thetas[key].append(getattr(self.theta, key))
        logalphas = [0]     # log_alpha is arbitrary for the initial state

        for i in range(n_iter):
            if (i+1) % 1000 == 0:
                print(f"Chain progress: {i+1}/{n_iter} iterations completed.")
            log_alpha = self._step()

            if i >= burnin:
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

    def _run_single_chain(self, seed, y, pf: ParticleFilter, kwargs_for_params, kwargs_for_sampling, n_iter, burnin, chain_id):
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
            kwargs_for_params=kwargs_for_params,
            kwargs_for_sampling=kwargs_for_sampling,
        )

        result = chain.run(y, n_iter=n_iter, burnin=burnin)
        acceptance_rate = chain.n_accepted / chain.n_steps if chain.n_steps > 0 else 0.0

        return result, acceptance_rate, chain_id

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
        """

        if n_chain == 1:
            # Run single chain without multiprocessing
            result, acceptance_rate, _ = self._run_single_chain(
                seed=self.rng.integers(0, 1_000_000),
                y=y,
                pf=self.pf,
                kwargs_for_params=self.kwargs_for_params,
                kwargs_for_sampling=self.kwargs_for_sampling,
                n_iter=n_iter,
                burnin=burnin,
                chain_id=0
            )
            print(f"Chain 1 acceptance rate: {acceptance_rate:.3f}")
            return [result]

        seeds = self.rng.integers(0, 1_000_000, size=n_chain)  # Generate random seeds for each chain

        with ProcessPoolExecutor(max_workers=n_chain) as executor:
            results = list(
                executor.map(
                    self._run_single_chain,
                    seeds,
                    [y] * n_chain,
                    [self.pf] * n_chain,
                    [self.kwargs_for_params] * n_chain,
                    [self.kwargs_for_sampling] * n_chain,
                    [n_iter] * n_chain,
                    [burnin] * n_chain,
                    list(range(n_chain))  # chain IDs for logging
                )
            )

        all_results = [r[0] for r in results]          # PMMH samples etc.
        acceptance_rates = [r[1] for r in results]    # acceptance rates
        chain_ids = [r[2] for r in results]          # chain IDs

        # Print all acceptance rates at the end
        for i, rate in enumerate(acceptance_rates):
            print(f"Chain {chain_ids[i]+1} acceptance rate: {rate:.3f}")

        return all_results
    
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