import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams
from src.filters.smc.base_pf import ParticleFilter

class AuxiliaryParticleFilter(ParticleFilter):
    """
    Auxiliary Particle Filter implementation for generic state-space models.
    """
    def __init__(self, model: StateSpaceModel, n_particles: int, resampler):
        super().__init__(model, n_particles, resampler)

    def run(self, y, theta: StateSpaceModelParams):
        """
        Run the auxiliary particle filter on observation sequence y.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : StateSpaceModelParams
            Model parameters.

        Returns
        -------
        history : list of tuples of size T+1.
            Each element is (particles, weights, indices, logmarlik) at each time step t.
            - particles: StateSpaceModelState with batched N particles.
            - weights: np.ndarray of shape (N,) with normalized weights of the particles.
            - indices: np.ndarray of shape (N,) with resampling indices used to get from step t-1 to t. At t=0, this is an empty array.
            - logmarlik: float, the log marginal likelihood up to time t. At t=0, this is 0.
        """
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)         # Sample initial particles
        weights = np.ones(self.N) / self.N                                      # Initialize weights uniformly
        history.append((particles, weights.copy(), np.array([], dtype=int), 0.0))    # Store history

        logmarlik = 0.0  # initialize log marginal likelihood

        # ----- Main loop -----
        for t in range(T):
            # ===== Auxiliary weights (look-ahead) =====
            predicted_states = self.model.expected_next_state(theta, particles)

            log_aux_weights = np.log(np.maximum(weights, 1e-300)) + self.model.log_likelihood(
                y[t], theta, predicted_states
            )

            # ---- First-stage normalizing constant ----
            max_log_aux = np.max(log_aux_weights)
            aux_sum = np.sum(np.exp(log_aux_weights - max_log_aux))
            log_first_stage = max_log_aux + np.log(aux_sum)

            aux_weights = np.exp(log_aux_weights - max_log_aux)
            aux_weights /= aux_sum

            # ===== Resample ancestors =====
            ancestor_indices = self.resampler(aux_weights, self.model.rng)
            ancestor_particles = particles[ancestor_indices]
            ancestor_predicted = predicted_states[ancestor_indices]
            
            # ===== Propagation =====
            particles = self.model.sample_next_state(theta, ancestor_particles)
            
            # ===== Weight correction =====
            log_w_num = self.model.log_likelihood(y[t], theta, particles)
            log_w_den = self.model.log_likelihood(y[t], theta, ancestor_predicted)
            log_weights = log_w_num - log_w_den

            # ---- Second-stage normalizing constant ----
            max_log_w = np.max(log_weights)
            weights_unnorm = np.exp(log_weights - max_log_w)
            weights_sum = np.sum(weights_unnorm)
            
            log_second_stage = max_log_w + np.log(weights_sum / self.N)

            # ---- Update marginal likelihood ----
            logmarlik += log_first_stage + log_second_stage

            # Normalize weights for filtering
            weights = weights_unnorm / weights_sum

            # --- Store history ---
            history.append((particles, weights.copy(), ancestor_indices, logmarlik))

        return history
    
    def run_conditional(self, y, theta: StateSpaceModelParams, x_ref):
        """
        Run conditional auxiliary particle filter given reference trajectory x_ref.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : StateSpaceModelParams
            Model parameters.
        x_ref : array-like, shape (T+1,)
            Reference trajectory to condition on. Must have length T+1, where T is the length of the observation sequence y.

        Returns
        -------
        history : same format as in run(), but with the first particle forced to follow x_ref at each time step.
        """
        T = len(y)
        history = []

        # ----- Initialization -----
        particles = self.model.sample_initial_state(theta, size=self.N)

        # Force reference initial state
        particles[0] = x_ref[0]

        weights = np.ones(self.N) / self.N
        history.append((particles, weights.copy(), np.array([], dtype=int), 0.0))

        logmarlik = 0.0

        # Track reference particle index
        ref_index = 0

        # ----- Main loop -----
        for t in range(T):

            # ===== Auxiliary weights =====
            predicted_states = self.model.expected_next_state(theta, particles)

            log_aux_weights = np.log(np.maximum(weights, 1e-300)) + self.model.log_likelihood(y[t], theta, predicted_states)

            # ---- First-stage normalizing constant ----
            max_log_aux = np.max(log_aux_weights)
            aux_sum = np.sum(np.exp(log_aux_weights - max_log_aux))
            log_first_stage = max_log_aux + np.log(aux_sum)

            aux_weights = np.exp(log_aux_weights - max_log_aux)
            aux_weights /= aux_sum

            # ===== Resample ancestors =====
            ancestor_indices = self.resampler(aux_weights, self.model.rng)

            # Force reference lineage
            ancestor_indices[0] = ref_index

            ancestor_particles = particles[ancestor_indices]
            ancestor_predicted = predicted_states[ancestor_indices]

            # ===== Propagation =====
            particles = self.model.sample_next_state(theta, ancestor_particles)

            # Force reference particle
            particles[0] = x_ref[t + 1]

            # ===== Second-stage weights =====
            log_w_num = self.model.log_likelihood(y[t], theta, particles)
            log_w_den = self.model.log_likelihood(y[t], theta, ancestor_predicted)
            log_weights = log_w_num - log_w_den

            # ---- Second-stage normalizing constant ----
            max_log_w = np.max(log_weights)
            weights_unnorm = np.exp(log_weights - max_log_w)
            weights_sum = np.sum(weights_unnorm)

            log_second_stage = max_log_w + np.log(weights_sum / self.N)

            # ---- Update marginal likelihood ----
            logmarlik += log_first_stage + log_second_stage

            # Normalize weights for filtering
            weights = weights_unnorm / weights_sum

            # --- Store history ---
            history.append((particles, weights.copy(), ancestor_indices, logmarlik))

        return history
    
    