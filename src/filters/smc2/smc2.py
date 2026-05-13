import copy
import numpy as np
import arviz as az
import pickle
from concurrent.futures import ProcessPoolExecutor

from src.filters.smc.base_pf import ParticleFilter
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from src.filters.smc.resampling import systematic_resampling

from typing import List


class ThetaParticle:
    "Container for parameter particles in SMC2."
    def __init__(self, theta: StateSpaceModelParams, x_particles: StateSpaceModelState, loglikelihood: float, logweight: float):
        self.theta = theta                      # Model parameters in this particle
        self.x_particles = x_particles          # State particles associated with this parameter particle. 
        self.loglikelihood = loglikelihood      # Log marginal likelihood estimate associated with this parameter particle
        self.logweight = logweight              # Log weight of this parameter particle 
    
    def copy(self):
        return ThetaParticle(
            theta=self.theta.copy(),
            x_particles=self.x_particles.copy(),
            loglikelihood=self.loglikelihood,
            logweight=self.logweight
        )

class SMC2:
    def __init__(
        self,
        pf: ParticleFilter,
        N_theta: int,
        gamma: float = 1.0,
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

            theta_particles.append(ThetaParticle(theta, x_particles, 0.0, 0.0))

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

        theta_particles = list(
            executor.map(_rejuvenate_particle, args)
        )

        return theta_particles


    def run(self, y, logger=None, thin=1, output_dir=None):
        if logger is not None:
            logger.info("Initializing theta particles...")
        theta_particles = self._initialize_theta_particles()
        
        T = len(y)

        history = []

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

                    # MCMC rejuvenation
                    theta_particles = self._rejuvenate(theta_particles, y[:t+1], executor)

                # Save theta particles in memory
                thetas = np.array([p.theta for p in theta_particles])
                logweights = np.array([p.logweight for p in theta_particles])
                logliks = np.array([p.loglikelihood for p in theta_particles])
                history.append((thetas, logweights, logliks, ess))

                # # Save x_particles on disk                # For now this is commented out because storing requires too much space. Maybe will do it somehow more efficient later.
                # if output_dir is not None:
                #     if logger is not None:
                #         logger.info(f"- Saving x_particles to disk...")
                #     with open(output_dir / f"x_particles_t{t}.pkl", "wb") as f:
                #         pickle.dump([p.x_particles[::thin] for p in theta_particles], f)

        # Save history on disk
        if output_dir is not None:
            if logger is not None:
                logger.info(f"- Saving history to disk...")
            with open(output_dir / "history.pkl", "wb") as f:
                pickle.dump(history, f)

        return history


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
    particle.loglikelihood += loglik_increment
    particle.logweight += loglik_increment

    return particle

def _rejuvenate_particle(args):
    particle, y_history, pf, prior, proposal, seed = args
    rng = np.random.default_rng(seed)

    theta_current = particle.theta
    loglik_current = particle.loglikelihood

    theta_prop = proposal.sample(rng, theta_current)

    pf_history = pf.run(y_history, theta_prop)

    x_particles = pf_history[-1][0]
    loglik_prop = pf_history[-1][3]

    log_alpha = (
        loglik_prop
        + prior.logpdf(theta_prop)
        + proposal.logpdf(theta_current, theta_prop)
        - loglik_current
        - prior.logpdf(theta_current)
        - proposal.logpdf(theta_prop, theta_current)
    )

    if np.log(rng.uniform()) < log_alpha:
        return ThetaParticle(
            theta_prop,
            x_particles,
            
            loglik_prop,
            0.0
        )

    return particle