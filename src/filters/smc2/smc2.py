import os
import copy
import numpy as np
import arviz as az
import h5py
from concurrent.futures import ProcessPoolExecutor
import tempfile
from pathlib import Path

from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState, StateSpaceModelPrior, StateSpaceModelProposal
from src.filters.smc.resampling import systematic_resampling

from typing import List


class ThetaParticle:
    "Container for parameter particles in SMC2."
    def __init__(self, theta: StateSpaceModelParams, x_particles: StateSpaceModelState, theta_logweight: float):
        self.theta = theta                      # Model parameters in this particle
        self.x_particles = x_particles          # State particles associated with this parameter particle. 
        self.theta_logweight = theta_logweight        # Log weight of this parameter particle 
    
    def copy(self):
        return ThetaParticle(
            theta=self.theta.copy(),
            x_particles=self.x_particles.copy(),
            theta_logweight=self.theta_logweight
        )

class SMC2:
    def __init__(
        self,
        pf: ParticleFilter,
        N_theta: int,
        gamma: float = 1.0,
        R: int = 1,
        kwargs_model=None,
        proposal_params=None,
        kwargs_prior=None,
    ):
        self.pf = pf
        self.model = pf.model
        self.rng = pf.model.rng

        self.N_theta = N_theta
        self.N_x = pf.N
        self.gamma = gamma
        self.R = R

        self.kwargs_model = kwargs_model if kwargs_model is not None else {}
        self.proposal_params = proposal_params if proposal_params is not None else {}
        self.kwargs_prior = kwargs_prior if kwargs_prior is not None else {}

        prior_cls = self.model.prior_type
        self.prior = prior_cls(**self.kwargs_prior)

        proposal_cls = self.model.proposal_type
        self.proposal = proposal_cls(self.proposal_params)

        self.n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def _initialize_theta_particles(self) -> List[ThetaParticle]:
        theta_particles = []
        x_particles_pred = []

        for _ in range(self.N_theta):
            theta = self.prior.sample(self.rng, **self.kwargs_model)
            x_particles = self.model.sample_initial_state(theta, size=self.N_x)
            x_particles_pred.append(self.model.sample_next_state(theta, x_particles))

            theta_particles.append(ThetaParticle(theta, x_particles, 0.0))

        return theta_particles, x_particles_pred

    def _normalize_weights(self, theta_particles: List[ThetaParticle]):
        logweights = np.array([
            p.theta_logweight
            for p in theta_particles
        ])

        max_logw = np.max(logweights)
        w = np.exp(logweights - max_logw)
        w /= np.sum(w)

        return w

    def _resample_theta_particles(self, theta_particles: List[ThetaParticle]) -> List[ThetaParticle]:

        w = self._normalize_weights(theta_particles)

        idx = systematic_resampling(w, self.rng)

        new_particles = []

        for i in idx:
            p: ThetaParticle = theta_particles[i]
            p = p.copy()
            p.theta_logweight = 0.0
            new_particles.append(p)

        return new_particles, idx

    def _rejuvenate(self, theta_particles: List[ThetaParticle], y_history: List, executor: ProcessPoolExecutor) -> List[ThetaParticle]:
        args = [
            (
                particle,
                y_history,
                self.pf,
                self.prior,
                self.proposal,
                self.rng.integers(1e9)
            )
            for particle in theta_particles
        ]

        results = list(
            executor.map(_rejuvenate_particle, args)
        )
        theta_particles = [res[0] for res in results]
        diagnostics = [res[1] for res in results]

        return theta_particles, diagnostics

    def _compute_proposal_moments(
        self,
        theta_particles: List[ThetaParticle]
    ):
        """
        Compute weighted empirical mean/covariance
        in unconstrained parameter space.
        """

        # -----------------------------------------
        # normalized weights
        # -----------------------------------------
        w = self._normalize_weights(theta_particles)

        # -----------------------------------------
        # unconstrained particles
        # shape: (N_theta, d)
        # -----------------------------------------
        Z = np.array([
            p.theta.to_unconstrained()
            for p in theta_particles
        ])

        # -----------------------------------------
        # weighted mean
        # -----------------------------------------
        mu = np.sum(w[:, None] * Z, axis=0)

        # -----------------------------------------
        # weighted covariance
        # -----------------------------------------
        centered = Z - mu
        Sigma = centered.T @ (w[:, None] * centered)

        # -----------------------------------------
        # regularization for numerical stability
        # -----------------------------------------
        Sigma += 1e-8 * np.eye(Sigma.shape[0])

        return mu, Sigma

    def _vector_dim(self, theta_particles: List[ThetaParticle]):
        # Get dimension of vector representation of parameters
        theta_example = theta_particles[0].theta
        vec = theta_example.to_vector()
        return len(vec)

    def _init_theta_hdf5_history(self, output_dir, theta_particles: List[ThetaParticle], T):
        h5_path = output_dir / "theta_history.h5"
        theta_dim = self._vector_dim(theta_particles)

        h5f = h5py.File(h5_path, "w")
        h5f.create_dataset(
            "theta",
            shape=(T + 1, self.N_theta, theta_dim),
            dtype="f8",
            chunks=(1, self.N_theta, theta_dim),
            compression="gzip",
            compression_opts=4,
        )

        h5f.create_dataset(
            "logweights",
            shape=(T + 1, self.N_theta),
            dtype="f8",
            chunks=(1, self.N_theta),
            compression="gzip",
            compression_opts=4,
        )

        h5f.create_dataset(
            "ess",
            shape=(T + 1,),
            dtype="f8",
            compression="gzip",
            compression_opts=4,
        )
        h5f.create_dataset(
            "resampled_times",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            compression="gzip",
            compression_opts=4,
        )
        return h5f

    def _write_theta_step(self, h5f, t, theta_particles: List[ThetaParticle], ess):
        theta_vecs = np.stack([p.theta.to_vector() for p in theta_particles], axis=0)
        logweights = np.array([p.theta_logweight for p in theta_particles], dtype=np.float64)

        # Save normalized logweights
        max_logw = np.max(logweights)
        w = np.exp(logweights - max_logw)
        w /= np.sum(w)
        logweights = np.log(w)

        h5f["theta"][t] = theta_vecs
        h5f["logweights"][t] = logweights
        h5f["ess"][t] = ess

    def _init_state_hdf5_history(self, output_dir, theta_particles: List[ThetaParticle], T, save_factor):
        h5_path = output_dir / "state_history.h5"
        state_dim = theta_particles[0].x_particles.to_numpy().shape[1]  # The shape is (N, state_dim)

        h5f = h5py.File(h5_path, "w")
        # Dataset for state particles at each time step (T+1) for all theta particles.
        h5f.create_dataset(
            "x_particles",
            shape=(T+1, self.N_x * save_factor, state_dim),  # Store save_factor times the number of x_particles across all theta particles for better diversity visualization
            dtype="f8",
            chunks=(1, self.N_x * save_factor, state_dim),
            compression="gzip",
            compression_opts=4,
        )
        # Dataset for final trajectories for each theta particle. This will be used for smoothing.
        h5f.create_dataset(
            "trajectories",
            shape=(self.N_theta, T+1, save_factor, state_dim),  # shape (N_theta, save_factor, T+1, state_dim)
            dtype="f8",
            chunks=(1, T+1, save_factor, state_dim),
            compression="gzip",
            compression_opts=4,
        )
        # Dataset for predicted state particles at each time step for all theta particles. This will be used for predictive checks.
        h5f.create_dataset(
            "x_particles_pred",
            shape=(T+1, self.N_x * save_factor, state_dim),  # Store save_factor times the number of x_particles across all theta particles for better diversity visualization
            dtype="f8",
            chunks=(1, self.N_x * save_factor, state_dim),
            compression="gzip",
            compression_opts=4,
        )
        return h5f
    
    def _write_state_step(self, h5f, t, theta_particles: List[ThetaParticle], x_particles_pred: List[StateSpaceModelState], save_factor: int):
        x_particles = np.array([p.x_particles.to_numpy() for p in theta_particles])
        theta_logweights = np.array([p.theta_logweight for p in theta_particles], dtype=np.float64)
        x_particles_pred = np.array([x.to_numpy() for x in x_particles_pred])

        # Change theta_logweights so that each element appears N_x times (for each state particle) and normalize
        w = np.exp(theta_logweights - np.max(theta_logweights))
        w /= np.sum(w)
        w_expanded = np.repeat(w, self.N_x)  # shape (N_theta * N_x,)
        w_expanded /= np.sum(w_expanded)  # Normalize the expanded weights
        x_particles = x_particles.reshape(-1, x_particles.shape[-1])
        x_particles_pred = x_particles_pred.reshape(-1, x_particles_pred.shape[-1])

        idx = systematic_resampling(w_expanded, self.rng, N_out=self.N_x * save_factor)
        x_particles = x_particles[idx]
        x_particles_pred = x_particles_pred[idx]

        # Save the resampled state particles at this time step
        h5f["x_particles"][t] = x_particles
        h5f["x_particles_pred"][t] = x_particles_pred

    def _write_trajectories_step(self, h5f, theta_id, trajectories: np.ndarray):
        # trajectories shape: (1, T+1, save_factor, state_dim)
        h5f["trajectories"][theta_id] = trajectories

    def _append_resampled_time(self, h5f, time_index):
        ds = h5f["resampled_times"]
        ds.resize((ds.shape[0] + 1,))
        ds[-1] = time_index

    def run(self, y, output_dir, logger=None, save_factor=1):
        """
        Run the SMC2 algorithm on the given data.

        Parameters
        ----------
        y: list or array-like, shape (T,)
            Observations over time.
        logger: logging.Logger, optional
            Logger for debug/info messages. If None, no logging is done.
        save_factor: int, optional, default=1
            Factor to determine the number of state particles to save in the history at each time step. The number of state particles saved will be N_x * save_factor.
        output_dir: pathlib.Path
            Directory to save all the history.

        Returns
        -------
        None. The history of particles, weights, ESS, and resampled times are saved in an HDF5 file in the output directory.
        """
        if logger is not None:
            logger.info("Initializing theta particles...")
            logger.info(f"Number of cores for parallel processing: {self.n_workers}")
        theta_particles, x_particles_pred = self._initialize_theta_particles()
        
        T = len(y)
        resampled_times = []
        if logger is not None:
            logger.info("Opening HDF5 history file...")
        h5f_theta = self._init_theta_hdf5_history(output_dir, theta_particles, T)
        h5f_state = self._init_state_hdf5_history(output_dir, theta_particles, T, save_factor)
        self._write_theta_step(h5f_theta, 0, theta_particles, np.nan)
        self._write_state_step(h5f_state, 0, theta_particles, x_particles_pred, save_factor)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            for t in range(T):
                if logger is not None:
                    logger.info(f"Time step {t+1}/{T}")

                y_t = y[t]

                args = [
                    (
                        particle,
                        y_t,
                        self.model,
                        self.N_x,
                        self.rng.integers(1e9)
                    )
                    for particle in theta_particles]

                results = list(executor.map(_update_theta_particle, args))

                theta_particles = [particle for particle, _ in results]
                x_particles_pred = [x_particles for _, x_particles in results]

                # ESS
                w = self._normalize_weights(theta_particles)
                ess = 1.0 / np.sum(w**2)
                if logger is not None:
                    logger.info(f"- ESS: {ess:.2f} / {self.N_theta}")

                if ess < self.gamma * self.N_theta:
                    if logger is not None:
                        logger.info(f"- ESS below threshold. Resampling and rejuvenating...")
                    # Resample theta particles
                    theta_particles, idx = self._resample_theta_particles(theta_particles)

                    # Update proposal moments
                    mu, Sigma = self._compute_proposal_moments(theta_particles)
                    new_params = {
                        "mean": mu,
                        "covariance": Sigma
                    }
                    self.proposal.update_params(new_params)
                    if logger is not None:
                        logger.debug(f"Mean vector: {mu}")
                        logger.debug(f"Covariance matrix: {Sigma}")

                    resampled_times.append(t+1)
                    self._append_resampled_time(h5f_theta, t+1)

                    for r in range(self.R):
                        if logger is not None:
                            logger.info(f"Rejuvenation step {r+1}/{self.R}...")

                        # MCMC rejuvenation
                        theta_particles, diagnostics = self._rejuvenate(theta_particles, y[:t+1], executor)

                
                self._write_theta_step(h5f_theta, t + 1, theta_particles, ess)
                self._write_state_step(h5f_state, t + 1, theta_particles, x_particles_pred, save_factor)

        h5f_theta.attrs["T"] = T
        h5f_theta.attrs["N_theta"] = self.N_theta
        h5f_theta.attrs["N_x"] = self.N_x
        h5f_theta.attrs["gamma"] = self.gamma
        h5f_theta.close()

        if logger is not None:
            logger.info(f"SMC2 completed. Moving to smoothing problem")

        # Create new exceutor for smoothing trajectories with fewer workers to avoid memory issues, as smoothing is much more memory intensive.
        with ProcessPoolExecutor(max_workers=max(1, self.n_workers // 8)) as executor:
            args = [
                (
                    i,
                    particle,
                    y,
                    self.pf,
                    save_factor,
                    output_dir,
                )
                for i, particle in enumerate(theta_particles)
            ]

            for i, tmp_file in executor.map(_compute_smoothing_trajectories, args):
                if logger is not None:
                    logger.info(f"Writing smoothing trajectories for theta particle {i+1}/{self.N_theta}")

                trajectories = np.load(tmp_file)
                self._write_trajectories_step(h5f_state, i, trajectories)
                os.remove(tmp_file)
            
            h5f_state.attrs["T"] = T
            h5f_state.attrs["N_theta"] = self.N_theta
            h5f_state.attrs["N_x"] = self.N_x
            h5f_state.attrs["save_factor"] = save_factor
            h5f_state.close()

def _compute_smoothing_trajectories(args: tuple[int, ThetaParticle, np.ndarray, ParticleFilter, int, str]) -> tuple[int, str]:
    i, particle, y, pf, save_factor, output_dir = args

    theta = particle.theta

    history = pf.run(y, theta)
    trajectories, _ = pf.smoothing_trajectories(
        history,
        n_traj=save_factor,
    )

    trajectories = np.array(
        [traj.to_numpy() for traj in trajectories]
    )

    tmp_file = Path(output_dir) / f"traj_{i}.npy"
    np.save(tmp_file, trajectories)

    return i, str(tmp_file)

def _update_theta_particle(args: tuple[ThetaParticle, float, StateSpaceModel, int, int]) -> tuple[ThetaParticle, StateSpaceModelState]:
    particle, y_t, model, N_x, seed = args
    rng = np.random.default_rng(seed)
    
    # Propagation
    x_particles = model.sample_next_state(particle.theta, particle.x_particles)

    # Weighting
    log_weights = model.log_likelihood(y_t, particle.theta, x_particles)

    max_log_w = np.max(log_weights)
    weights = np.exp(log_weights - max_log_w)
    weights_sum = np.sum(weights)
    weights /= weights_sum

    # Incremental likelihood
    loglik_increment = max_log_w + np.log(weights_sum / N_x)

    # Resampling
    indices = systematic_resampling(weights, rng)
    x_particles_new = x_particles[indices]      # All particles are now equally weighted (1/N_x), so we dont have to carry on weights.

    particle.x_particles = x_particles_new
    particle.theta_logweight += loglik_increment

    return particle, x_particles

def _rejuvenate_particle(args: tuple[ThetaParticle, np.ndarray, ParticleFilter, StateSpaceModelPrior, StateSpaceModelProposal, int]):
    particle, y_history, pf, prior, proposal, seed = args
    rng = np.random.default_rng(seed)

    theta_current = particle.theta
    current_history = pf.run(y_history, theta_current, only_last_step=True)
    loglik_current = current_history[-1][3]

    theta_prop = proposal.sample(rng, theta_current)
    proposed_history = pf.run(y_history, theta_prop, only_last_step=True)

    x_particles = proposed_history[-1][0]
    loglik_prop = proposed_history[-1][3]

    prior_prop = prior.logpdf(theta_prop)
    prior_current = prior.logpdf(theta_current)
    proposal_forward = proposal.logpdf(theta_prop, theta_current)
    proposal_backward = proposal.logpdf(theta_current, theta_prop)

    log_alpha = (
        loglik_prop
        + prior_prop
        + proposal_backward
        - loglik_current
        - prior_current
        - proposal_forward
    )

    if np.log(rng.uniform()) < log_alpha:
        return ThetaParticle(theta_prop, x_particles, 0.0), (log_alpha, loglik_prop, loglik_current, prior_prop, prior_current, proposal_backward, proposal_forward)

    return particle, (log_alpha, loglik_prop, loglik_current, prior_prop, prior_current, proposal_backward, proposal_forward)