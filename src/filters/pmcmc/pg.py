import h5py
import numpy as np
import arviz as az
from concurrent.futures import ProcessPoolExecutor
from typing import List

from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from src.filters.smc.base_pf import ParticleFilter
from src.utils.log import setup_chain_logging

class PGS_Chain:
    """
    A single PGS chain.
    """

    def __init__(self, pf: ParticleFilter, kwargs_prior=None, kwargs_model=None, proposal_params=None):
        """
        Parameters
        ----------
        pf: ParticleFilter
            Particle filter class to use for the conditional SMC step. Must have a .run_conditional() method implemented.
        kwargs_model: dict, optional
            Additional keyword arguments to pass to the initialization of the model. For example, for MSSV model, num_regimes is needed to initialize the model.
        kwargs_prior: dict, optional
            Additional keyword arguments to pass to the initialization of the prior distribution for parameters.
        proposal_params: dict, optional
            Additional keyword arguments to pass to the proposal distribution.
        """
        self.pf = pf
        self.model = pf.model
        self.rng = pf.rng
        self.kwargs_prior = kwargs_prior if kwargs_prior is not None else {}
        self.kwargs_model = kwargs_model if kwargs_model is not None else {}
        self.proposal_params = proposal_params if proposal_params is not None else {}

        prior_cls = pf.model.prior_type
        self.prior = prior_cls(**self.kwargs_prior)
        proposal_cls = pf.model.proposal_type
        self.proposal = proposal_cls(self.proposal_params, self.prior)

    def _run_pf_and_sample(self, y, theta: StateSpaceModelParams, x_current: List[StateSpaceModelState]):
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
        
    def _step(self):
        """
        Perform one iteration of the PG algorithm: run conditional PF and sample new trajectory and parameters.
        """
        # Run conditional PF and sample new trajectory
        trajectory, logmarlik = self._run_pf_and_sample(self.y, self.theta, self.current_trajectory)

        # Sample new parameters given the new trajectory
        new_theta = self.proposal.sample(self.rng, self.theta, trajectory)

        # Update current state
        self.current_trajectory = trajectory
        self.current_logmarlik = logmarlik
        self.theta = new_theta
        self.n_steps += 1

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
    
        return h5f

    def _write_chain_step(self, h5f, idx: int, theta: StateSpaceModelParams, trajectory: List[StateSpaceModelState], logmarlik: float):
        h5f["thetas"][idx] = theta.to_vector()
        h5f["trajectories"][idx] = np.array([state.to_numpy() for state in trajectory]).reshape(h5f["trajectories"].shape[1:])  # reshape to (T, state_dim)
        h5f["logmarliks"][idx] = logmarlik

    def run(self, y, n_iter: int, output_dir, burnin=0, chain_id=0, logger=None):
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
        output_dir : str
            Directory to save the results of each chain.
        chain_id : int, optional
            Identifier for the chain. Default is 0.
        logger : logging.Logger, optional
            Logger for the chain. If None, no logging is performed. 

        Returns
        -------
        None. The results are stored in the HDF5 files in the output_dir.
        """
        if burnin >= n_iter:
            raise ValueError("Burn-in must be less than the total number of iterations.")

        if logger is not None:
            logger.info(f"Initializing PG chain {chain_id} with {n_iter} iterations and burn-in of {burnin}.")
        
        self.current_trajectory = None
        self.current_logmarlik = None

        self.n_accepted = 0
        self.n_steps = 0

        self.y = y

        self._initialize()
        
        h5f = self._init_hdf5_chain(output_dir, chain_id, n_iter - burnin, len(self.theta.to_vector()), self.current_trajectory[0].to_numpy().shape[1], len(self.current_trajectory))

        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"PG chain {chain_id} initialized. Starting burn-in...")

        for i in range(burnin):
            self._step()
            if logger is not None:
                logger.info(f"Chain {chain_id} - Burn-in step {i+1}/{burnin}")

        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"Burn-in completed for chain {chain_id}. Starting sampling...")

        self._step()  # First step
        self._write_chain_step(h5f, 0, self.theta, self.current_trajectory, self.current_logmarlik)

        # The second loop to run the remaining iterations and store samples after burn-in
        for i in range(burnin+1, n_iter):
            self._step()
            self._write_chain_step(h5f, i - burnin, self.theta, self.current_trajectory, self.current_logmarlik)
            if logger is not None:
                logger.info(f"Chain {chain_id} - Sampling step {i-burnin}/{n_iter-burnin}")

        h5f.attrs["acceptance_rate"] = self.n_accepted / self.n_steps if self.n_steps > 0 else 0.0
        h5f.attrs["initial_parameters"] = self.initial_params.to_vector()

        if logger is not None:
            logger.info("-" * 60)
            logger.info(f"PG chain {chain_id} completed. Acceptance rate: {h5f.attrs['acceptance_rate']:.4f}")
            logger.info(f"Results saved to {output_dir / f'chain_{chain_id}.h5'}")

        h5f.close()
    
class ParticleGibbsSampler:
    """
    Particle Gibbs Sampler for state-space models.
    """

    def __init__(self, pf: ParticleFilter, kwargs_model=None, kwargs_prior=None, proposal_params=None):
        """
        Parameters
        ----------
        pf: ParticleFilter
            Particle filter class to use for the conditional SMC step. Must have a .run_conditional() method implemented.
        kwargs_model: dict, optional
            Additional keyword arguments to pass to the initialization of the model. 
        kwargs_prior: dict, optional
            Additional keyword arguments to pass to the initialization of the prior distribution for parameters.
        proposal_params: dict, optional
            Additional keyword arguments to pass to the proposal distribution.
        """
        self.pf = pf
        self.rng = pf.model.rng
        self.kwargs_model = kwargs_model if kwargs_model is not None else {}
        self.kwargs_prior = kwargs_prior if kwargs_prior is not None else {}
        self.proposal_params = proposal_params if proposal_params is not None else {}

    def _run_single_chain(self, seed, y, pf: ParticleFilter, kwargs_model, kwargs_prior, proposal_params, n_iter, burnin, chain_id, output_dir, logs_dir=None):
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

        chain = PGS_Chain(pf_chain, kwargs_prior=kwargs_prior, kwargs_model=kwargs_model, proposal_params=proposal_params)

        if logs_dir is not None:
            logger = setup_chain_logging(logs_dir, "PG", chain_id)
        else:
            logger = None

        chain.run(y, n_iter=n_iter, burnin=burnin, output_dir=output_dir, chain_id=chain_id, logger=logger)

    
    def run(self, y, n_iter: int, n_chain: int, output_dir, burnin: int=0, logs_dir=None):
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
        burnin : int, optional
            Number of burn-in iterations to discard. Must be less than n_iter. Default is 0.
        output_dir : str
            Directory to save the results of each chain. If None, returns error, as output_dir is required to save results.
        logs_dir : str, optional
            Directory to save logs. If None, no logs are saved.
            
        Returns
        -------
        None. The results are stored in the HDF5 files in the output_dir. 
        """

        if n_chain == 1:
            # Run single chain without multiprocessing
            self._run_single_chain(
                seed=self.rng.integers(0, 1_000_000),
                y=y, 
                pf=self.pf, 
                kwargs_model=self.kwargs_model,
                kwargs_prior=self.kwargs_prior,
                proposal_params=self.proposal_params,
                n_iter=n_iter, 
                burnin=burnin,
                chain_id=0,
                output_dir=output_dir,
                logs_dir=logs_dir
            )
            return None
        
        seeds = self.rng.integers(0, 1_000_000, size=n_chain)

        with ProcessPoolExecutor(max_workers=n_chain) as executor:
            results = list(
                executor.map(
                    self._run_single_chain,
                    seeds,
                    [y] * n_chain,
                    [self.pf] * n_chain,
                    [self.kwargs_model] * n_chain,
                    [self.kwargs_prior] * n_chain,
                    [self.proposal_params] * n_chain,
                    [n_iter] * n_chain,
                    [burnin] * n_chain,
                    list(range(n_chain)),
                    [output_dir] * n_chain,
                    [logs_dir] * n_chain
                )
            )

        return None
