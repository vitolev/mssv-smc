import numpy as np
import arviz as az
from concurrent.futures import ProcessPoolExecutor
from src.models.base import StateSpaceModel, StateSpaceModelParams
from src.filters.smc.base_pf import ParticleFilter

class PGS_Chain:
    """
    A single PGS chain.
    """

    def __init__(self, pf: ParticleFilter, kwargs_prior=None, kwargs_model=None, kwargs_proposal=None):
        """
        Parameters
        ----------
        pf: ParticleFilter
            Particle filter class to use for the conditional SMC step. Must have a .run_conditional() method implemented.
        kwargs_model: dict, optional
            Additional keyword arguments to pass to the initialization of the model. For example, for MSSV model, num_regimes is needed to initialize the model.
        kwargs_prior: dict, optional
            Additional keyword arguments to pass to the initialization of the prior distribution for parameters.
        kwargs_proposal: dict, optional
            Additional keyword arguments to pass to the proposal distribution.
        """
        self.pf = pf
        self.rng = pf.rng
        self.kwargs_prior = kwargs_prior if kwargs_prior is not None else {}
        self.kwargs_model = kwargs_model if kwargs_model is not None else {}
        self.kwargs_proposal = kwargs_proposal if kwargs_proposal is not None else {}

        prior_cls = pf.model.prior_type
        self.prior = prior_cls(**self.kwargs_prior)
        proposal_cls = pf.model.proposal_type
        self.proposal = proposal_cls(**self.kwargs_proposal)

    def _run_pf_and_sample(self, y, theta: StateSpaceModelParams, x_current):
        """
        Run conditional PF once and sample smoothing trajectory(ies).
        """
        history = self.pf.run_conditional(y, theta, x_current)

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
        Initialize the chain with a conditional PF run.
        """
        self.theta = self.prior.sample(self.rng, **self.kwargs_model)
        self.initial_params = self.theta.copy()
        self.theta_vars = vars(self.theta)

        # Generate initial trajectory by running one iteration of PF without conditioning,
        # since we don't have a reference trajectory at this point. This will be used as the initial trajectory for the first iteration of PGS.
        history = self.pf.run(self.y, self.theta)
        logmarlik = history[-1][3]
        trajectory, _ = self.pf.smoothing_trajectories(history, n_traj=1)

        self.current_trajectory = trajectory
        self.current_logmarlik = logmarlik
        

    def run(self, y, n_iter: int, burnin=0):
        """
        Run the Particle Gibbs sampler.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observation sequence.
        n_iter : int
            Number of PGS iterations.
        burnin : int, optional
            Number of burn-in iterations to discard. Must be less than n_iter. Default is 0.

        Returns
        -------
        samples : list of arrays, size n_iter
            Sampled trajectories from each iteration. Element samples[i] is the i-th trajectory with size T+1
        logmarliks : array, size n_iter
            Log marginal likelihoods from each iteration.
        thetas : list of StateSpaceModelParams, size n_iter
            Sampled parameters from each iteration.
        """
        if burnin >= n_iter:
            raise ValueError("Burn-in must be less than the total number of iterations.")
        
        self.current_trajectory = None
        self.current_logmarlik = None

        self.y = y

        self._initialize()

        # Run burn-in iterations
        for i in range(burnin):
            self._step()

        # Boundary iteration when we first collect samples
        logmarliks = []
        thetas = {key: [] for key in self.theta_vars.keys()}

        self._step()  # First step

        samples = self.current_trajectory
        logmarliks.append(self.current_logmarlik)
        for key in self.theta_vars.keys():
            thetas[key].append(getattr(self.theta, key))

        # Run remaining iterations
        for i in range(burnin + 1, n_iter):
            self._step()
            samples = [state.add(element) for state, element in zip(samples, self.current_trajectory)]
            logmarliks.append(self.current_logmarlik)
            for key in self.theta_vars.keys():
                thetas[key].append(getattr(self.theta, key))

        return samples, logmarliks, thetas
    
class ParticleGibbsSampler:
    """
    Particle Gibbs Sampler for state-space models.
    """

    def __init__(self, pf: ParticleFilter, kwargs_for_params=None):
        """
        Parameters
        ----------
        pf: ParticleFilter
            Particle filter class to use for the conditional SMC step. Must have a .run_conditional() method implemented.
        kwargs_for_params: dict, optional
            Additional keyword arguments to pass to the initialization of parameters. 
            For example, for MSSV model, num_regimes is needed to initialize the parameters.
        """
        self.pf = pf
        self.rng = pf.model.rng
        self.model = pf.model
        self.kwargs_for_params = kwargs_for_params if kwargs_for_params is not None else {}

    def _run_single_chain(self, seed, y, pf: ParticleFilter, kwargs_for_params, n_iter, x_init, chain_id):
        """
        Run a single PGS chain with a given random seed.
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

        chain = PGS_Chain(pf_chain, kwargs_for_params)

        results = chain.run(y, n_iter, x_init)

        return results, chain_id
    
    def run(self, y, n_iter: int, n_chain: int, x_init=None):
        """
        Run multiple PGS chains in parallel and return their results.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observation sequence.
        n_iter : int
            Number of PGS iterations per chain.
        n_chains : int
            Number of parallel PGS chains to run.
        x_init : list of StateSpaceModelState, optional
            Initial trajectory for all chains. If None, sampled from model.

        Returns
        -------
        all_results : list
            List of results from each chain, where each result is a tuple (samples, logmarliks, thetas).
        """

        if n_chain == 1:
            # Run single chain without multiprocessing
            results, _ = self._run_single_chain(
                seed=self.rng.integers(0, 1_000_000),
                y=y, 
                pf=self.pf, 
                kwargs_for_params=self.kwargs_for_params, 
                n_iter=n_iter, 
                x_init=x_init, 
                chain_id=0
            )
            return [results]
        
        seeds = self.rng.integers(0, 1_000_000, size=n_chain)

        with ProcessPoolExecutor(max_workers=n_chain) as executor:
            results = list(
                executor.map(
                    self._run_single_chain,
                    seeds,
                    [y] * n_chain,
                    [self.pf] * n_chain,
                    [self.kwargs_for_params] * n_chain,
                    [n_iter] * n_chain,
                    [x_init] * n_chain,
                    list(range(n_chain))
                )
            )
        return [r[0] for r in results]
    
    def to_inference_data(self, results):
        """
        Convert results from multiple chains into an ArviZ InferenceData object for analysis.

        Parameters
        ----------
        results : list
            List of results from each chain, where each result is a tuple (samples, logmarliks, thetas).

        Returns
        -------
        inference_data : arviz.InferenceData
            ArviZ InferenceData object containing the parameters and log marginal likelihoods from all chains.
        """
        data = {}
        for chain_result in results:
            if "logmarliks" not in data:
                data["logmarliks"] = []
            _, logmarliks, thetas = chain_result
            data["logmarliks"].append(logmarliks)
            for param_name, param_values in thetas.items():
                if param_name not in data:
                    data[param_name] = []
                data[param_name].append(param_values)

        for key in data:
            data[key] = np.array(data[key])

        return az.from_dict(data)