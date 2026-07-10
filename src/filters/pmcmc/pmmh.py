import h5py
import numpy as np
import arviz as az
from concurrent.futures import ProcessPoolExecutor
from typing import List

from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from src.utils.log import setup_chain_logging

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
    
    def _init_hdf5_chain(self, output_dir, chain_id: int, n_samples: int, theta_dim: int, state_dim: int, T: int):
        h5_path = output_dir / f"chain_{chain_id}.h5"

        h5f = h5py.File(h5_path, "w")

        h5f.create_dataset(
            "thetas",
            shape=(n_samples, theta_dim),
            dtype="f8",
            compression="gzip",
            compression_opts=4,
        )

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
    
    def _write_chain_step(self, h5f, idx: int, theta: StateSpaceModelParams, trajectory: List[StateSpaceModelState], logmarlik: float, log_alpha: float):
        h5f["thetas"][idx] = theta.to_vector()
        h5f["trajectories"][idx] = np.array([state.to_numpy() for state in trajectory]).reshape(h5f["trajectories"].shape[1:])  # reshape to (T, state_dim)
        h5f["logmarliks"][idx] = logmarlik
        h5f["logalphas"][idx] = log_alpha

    def run(self, y, n_iter: int, output_dir, burnin=0, chain_id=0, logger=None):
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
        output_dir : str
            Directory to save intermediate results or logs.
        logger : logging.Logger, optional
            Logger for logging information. If None, no logging is performed.
        chain_id : int, optional
            Identifier for the chain. Default is 0.

        Returns
        -------
        None. The results are stored in the HDF5 file specified by output_dir.
        """
        if burnin >= n_iter:
            raise ValueError("Burn-in must be less than the total number of iterations.")

        if logger is not None:
            logger.info(f"Initializing PMMH chain {chain_id} with {n_iter} iterations and burn-in of {burnin}.")

        self.current_trajectory = None
        self.current_logmarlik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y

        self._initialize()

        h5f = self._init_hdf5_chain(output_dir, chain_id, n_iter - burnin, len(self.theta.to_vector()), self.current_trajectory[0].to_numpy().shape[1], len(self.current_trajectory))

        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"PMMH chain {chain_id} initialized. Starting burn-in...")

        for i in range(burnin):
            log_alpha = self._step()
            if logger is not None:
                logger.info(f"Chain {chain_id} - Burn-in step {i+1}/{burnin}, log_alpha: {log_alpha:.4f}")

        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"Burn-in completed for chain {chain_id}. Starting sampling...")

        log_alpha = self._step()    # First step
        self._write_chain_step(h5f, 0, self.theta, self.current_trajectory, self.current_logmarlik, log_alpha)  # Store the first sample after burn-in

        # The second loop to run the remaining iterations and store samples after burn-in
        for i in range(burnin+1, n_iter):
            log_alpha = self._step()
            self._write_chain_step(h5f, i - burnin, self.theta, self.current_trajectory, self.current_logmarlik, log_alpha)
            if logger is not None:
                logger.info(f"Chain {chain_id} - Sampling step {i-burnin}/{n_iter-burnin}, log_alpha: {log_alpha:.4f}")

        h5f.attrs["acceptance_rate"] = self.n_accepted / self.n_steps if self.n_steps > 0 else 0.0
        h5f.attrs["initial_parameters"] = self.initial_params.to_vector()

        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"PMMH chain {chain_id} completed. Acceptance rate: {h5f.attrs['acceptance_rate']:.4f}")
            logger.info(f"Results saved to {output_dir / f'chain_{chain_id}.h5'}")

        h5f.close()

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

    def _run_single_chain(self, seed, y, pf: ParticleFilter, kwargs_model, proposal_param, kwargs_prior, n_iter, burnin, chain_id, output_dir, logs_dir=None):
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

        if logs_dir is not None:
            logger = setup_chain_logging(logs_dir, "PMMH", chain_id)
        else:
            logger = None

        chain.run(y, n_iter=n_iter, burnin=burnin, output_dir=output_dir, chain_id=chain_id, logger=logger)

    def run(self, y, n_iter: int, n_chain: int, output_dir, burnin=0, logs_dir=None):
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
        output_dir : str
            Directory to save intermediate results or logs. If None, returns error, as output_dir is required to save results.
        logs_dir : str, optional
            Directory to save logs. If None, no logs are saved.
            
        Returns
        -------
        None. The results are stored in the HDF5 file specified by output_dir.
        """

        if n_chain == 1:
            # Run single chain without multiprocessing
            self._run_single_chain(
                seed=self.rng.integers(0, 1_000_000),
                y=y,
                pf=self.pf,
                kwargs_model=self.kwargs_model,
                proposal_param=self.proposal_param,
                kwargs_prior=self.kwargs_prior,
                n_iter=n_iter,
                burnin=burnin,
                chain_id=0,
                output_dir=output_dir,
                logs_dir=logs_dir
            )
            return None

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
                    list(range(n_chain)),
                    [output_dir] * n_chain,
                    [logs_dir] * n_chain
                )
            )
        return None
