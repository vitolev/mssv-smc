import copy
import numpy as np
import arviz as az
import h5py
from concurrent.futures import ProcessPoolExecutor

from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from src.filters.smc.resampling import systematic_resampling

from typing import List


class ThetaParticle:
    "Container for parameter particles in SMC2."
    def __init__(self, theta: StateSpaceModelParams, x_particles: StateSpaceModelState, logweight: float):
        self.theta = theta                      # Model parameters in this particle
        self.x_particles = x_particles          # State particles associated with this parameter particle. 
        self.logweight = logweight              # Log weight of this parameter particle 
    
    def copy(self):
        return ThetaParticle(
            theta=self.theta.copy(),
            x_particles=self.x_particles.copy(),
            logweight=self.logweight
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

    def _initialize_theta_particles(self) -> List[ThetaParticle]:

        theta_particles = []

        for _ in range(self.N_theta):
            theta = self.prior.sample(self.rng, **self.kwargs_model)
            x_particles = self.model.sample_initial_state(theta, size=self.N_x)

            theta_particles.append(ThetaParticle(theta, x_particles, 0.0))

        return theta_particles

    def _normalize_weights(self, theta_particles: List[ThetaParticle]):
        logweights = np.array([
            p.logweight
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
            p.logweight = 0.0
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

    def _init_hdf5_history(self, output_dir, theta_particles, T):
        h5_path = output_dir / "history.h5"
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
            chunks=(256,),
            compression="gzip",
            compression_opts=4,
        )
        h5f.create_dataset(
            "resampled_times",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(256,),
            compression="gzip",
            compression_opts=4,
        )
        return h5f

    def _write_history_step(self, h5f, t, theta_particles, ess):
        theta_vecs = np.stack([p.theta.to_vector() for p in theta_particles], axis=0)
        logweights = np.array([p.logweight for p in theta_particles], dtype=np.float64)

        h5f["theta"][t] = theta_vecs
        h5f["logweights"][t] = logweights
        h5f["ess"][t] = ess

    def _append_resampled_time(self, h5f, time_index):
        ds = h5f["resampled_times"]
        ds.resize((ds.shape[0] + 1,))
        ds[-1] = time_index

    def run(self, y, logger=None, thin=1, output_dir=None):
        """
        Run the SMC2 algorithm on the given data.

        Parameters
        ----------
        y: list or array-like, shape (T,)
            Observations over time.
        logger: logging.Logger, optional
            Logger for debug/info messages. If None, no logging is done.
        thin: int, optional
            Thinning interval for storing particles in history. (default is 1, meaning store every step)
        output_dir: pathlib.Path, optional
            Directory to save all the history.

        Returns
        -------
        None. The history of particles, weights, ESS, and resampled times are saved in an HDF5 file in the output directory if provided.
        """
        if logger is not None:
            logger.info("Initializing theta particles...")
        theta_particles = self._initialize_theta_particles()
        
        T = len(y)
        resampled_times = []
        h5f = None
        if output_dir is not None:
            if logger is not None:
                logger.info("Opening HDF5 history file...")
            h5f = self._init_hdf5_history(output_dir, theta_particles, T)
            self._write_history_step(h5f, 0, theta_particles, np.nan)

        with ProcessPoolExecutor(max_workers=8) as executor:
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

                theta_particles = list(executor.map(_update_theta_particle, args))

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
                    if h5f is not None:
                        self._append_resampled_time(h5f, t+1)

                    for r in range(self.R):
                        if logger is not None:
                            logger.info(f"Rejuvenation step {r+1}/{self.R}...")

                        # MCMC rejuvenation
                        theta_particles, diagnostics = self._rejuvenate(theta_particles, y[:t+1], executor)

                if h5f is not None:
                    self._write_history_step(h5f, t + 1, theta_particles, ess)

        if h5f is not None:
            h5f.attrs["T"] = T
            h5f.attrs["N_theta"] = self.N_theta
            h5f.attrs["N_x"] = self.N_x
            h5f.attrs["gamma"] = self.gamma
            h5f.close()


def _update_theta_particle(args):
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
    x_particles = x_particles[indices]      # All particles are now equally weighted (1/N_x), so we dont have to carry on weights.

    particle.x_particles = x_particles
    particle.logweight += loglik_increment

    return particle

def _rejuvenate_particle(args):
    particle, y_history, pf, prior, proposal, seed = args
    rng = np.random.default_rng(seed)

    theta_current = particle.theta
    current_history = pf.run(y_history, theta_current)
    loglik_current = current_history[-1][3]

    theta_prop = proposal.sample(rng, theta_current)
    proposed_history = pf.run(y_history, theta_prop)

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