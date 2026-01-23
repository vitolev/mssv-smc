import numpy as np
from scipy.stats import norm
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from typing import List, Tuple

class MSSVModelParams(StateSpaceModelParams):
    """
    Container for MSSV model parameters. If initialized without parameters, sampling from prior is done.
    """
    def __init__(self, rng: np.random.Generator = None,
                 num_regimes: int = None,
                 mu: List[float] = None, 
                 phi: List[float] = None, 
                 sigma_eta: List[float] = None, 
                 P: List[List[float]] = None):
        super().__init__()

        # Sample from prior if any parameter is None
        if any(param is None for param in [mu, phi, sigma_eta, P]):
            if rng is None or num_regimes is None:
                raise ValueError("RNG and num_regimes must be provided when sampling from prior.")
            mu = []
            phi = []
            sigma_eta = []
            P = []

            for _ in range(num_regimes):
                mu.append(rng.normal(0, 10))
                phi.append(rng.uniform(0, 1))
                sigma_eta.append(rng.uniform(0.1, 2.0))

            P_matrix = rng.dirichlet(np.ones(num_regimes), size=num_regimes)
            P = P_matrix.tolist()

            self.mu = np.array(mu)
            self.phi = np.array(phi)
            self.sigma_eta = np.array(sigma_eta)
            self.P = np.array(P)
        
        # Set provided parameters
        else:
            self.mu = np.array(mu)
            self.phi = np.array(phi)
            self.sigma_eta = np.array(sigma_eta)
            self.P = np.array(P)

            if not (len(mu) == len(phi) == len(sigma_eta) == len(P)):
                raise ValueError("Parameters mu, phi, sigma_eta, and P must have the same length.")
            
            if any(sigma <= 0 for sigma in sigma_eta):
                raise ValueError("Standard deviations sigma_eta must be positive.")
            
            if any(p < 0 for row in P for p in row):
                raise ValueError("Transition probabilities must be non-negative.")
            
            if any(abs(sum(row) - 1.0) > 1e-8 for row in P):
                raise ValueError("Each row of transition matrix P must sum to 1.")

class MSSVModelState(StateSpaceModelState):
    """
    Container for MSSV model state: (h_t, s_t)
        h_t: continuous latent log-volatility vector
        s_t: one-hot encoded regime vector 
    """
    def __init__(self, h_t: np.ndarray, s_t: np.ndarray):
        self.h_t = h_t  # Log-volatility
        self.s_t = s_t  # Regime (one-hot encoded)

class MSSVModel(StateSpaceModel):
    """
    Markov-Switching Stochastic Volatility Model

    A model used for capturing regime-switching behavior in financial time series, by modeling
    the log-volatility as a latent variable that switches between different regimes.

    Model definition:
        h_t -- log-volatility (continuous latent state)
        s_t -- regime (categorical latent state vector)
        y_t -- observation (observable returns)

        s_0 ~ Uniform{1, ..., K}
        h_0 | s_0 ~ N(mu_{s_0}, sigma_eta_{s_0}^2)
        s_t | s_{t-1} ~ Categorical(P_{s_{t-1}, :))
        h_t | h_{t-1}, s_t ~ N(mu_{s_t} + phi_{s_t} * (h_{t-1} - mu_{s_t}), sigma_eta_{s_t}^2)
        y_t | h_t ~ N(0, exp(h_t))
    """
    def __init__(self, rng=None):
        super().__init__(rng)

    def sample_observation(self, theta : MSSVModelParams, state: MSSVModelState) -> np.ndarray:
        """
        Sample an observation y_t given state x_t = (h_t, s_t) and parameters theta.

        y_t ~ p(y_t | x_t, theta)

        Parameters
        ----------
            theta: MSSVModelParams
                Model parameters.
            state: MSSVModelState
                Current state of size N. 
        Returns
        -------
            y_t: np.ndarray
                Sampled observation with shape (N,).
        """
        h_t = state.h_t
        return self.rng.normal(0.0, np.exp(0.5 * h_t))

    def sample_initial_state(self, theta : MSSVModelParams, size: int = 1) -> MSSVModelState:
        """
        Sample the initial state x_0 = (h_0, s_0) given initial parameters theta.

        x_0 ~ p(x_0 | theta)

        Parameters
        ----------
            theta: MSSVModelParams
                Model parameters.
            size: int
                Number of initial states to sample. This influences the shape of returned arrays in MSSVModelState. (default = 1)
        Returns
        -------
            state: MSSVModelState
                Sampled initial state with the shapes:
                    h_0: (size,)
                    s_0: (size, K)
        """
        K = len(theta.mu)
        s0 = np.zeros((size, K))   # Initialize regime array as one-hot encoding

        # Uniformly sample initial regimes
        regimes = self.rng.integers(0, K, size=size)
        s0[np.arange(size), regimes] = 1

        # Sample initial log-volatilities based on regimes
        h0 = self.rng.normal(theta.mu[regimes], theta.sigma_eta[regimes])   # np.random.normal uses stddev as second parameter

        return MSSVModelState(h0, s0)

    def sample_next_state(self, theta : MSSVModelParams, state: MSSVModelState) -> MSSVModelState:
        """
        Sample the next state x_t = (h_t, s_t) given previous state x_t-1 and new parameters theta.

        x_t ~ p(x_t | x_{t-1}, theta)

        Parameters
        ----------
            theta: MSSVModelParams
                Model parameters.
            state: MSSVModelState
                Previous state of size N.
        Returns
        -------
            state: MSSVModelState
                Sampled next state of size N.
        """
        h_prev, s_prev = state.h_t, state.s_t
        N, K = s_prev.shape

        # Regime transition
        probs = s_prev @ theta.P  # (N, K)
        u = self.rng.random(probs.shape[0])     # (N,): random uniform values [0,1)
        indices = np.sum(np.cumsum(probs, axis=1) < u[:, None], axis=1)     # (N,): sampled regime indices by CDF inversion
        s_t = np.zeros_like(probs)
        s_t[np.arange(probs.shape[0]), indices] = 1

        # Volatility transition
        mu = theta.mu[indices]
        phi = theta.phi[indices]
        sigma = theta.sigma_eta[indices]

        h_t = mu + phi * (h_prev - mu) + sigma * self.rng.normal(size=N)

        return MSSVModelState(h_t, s_t)
    
    def expected_next_state(self, theta : MSSVModelParams, state: MSSVModelState) -> MSSVModelState:
        """
        Compute the expected next state given current state and parameters theta.

        E[x_t | x_{t-1}, theta]

        Parameters
        ----------
            theta: MSSVModelParams
                Model parameters.
            state: MSSVModelState
                Current state of size N.
        Returns
        -------
            state: MSSVModelState
                Expected next state of size N.
        """
        h_prev, s_prev = state.h_t, state.s_t

        # Expected regime distribution
        s_exp = s_prev @ theta.P

        # Regime-specific parameters
        mu = theta.mu                                # (K,)
        phi = theta.phi                              # (K,)

        # Expected log-volatility
        # shape tricks:
        # h_prev[:, None]  -> (N, 1)
        # mu[None, :]      -> (1, K)
        h_exp = np.sum(
            s_exp * (mu + phi * (h_prev[:, None] - mu)),
            axis=1
        )                                            # (N,)

        return MSSVModelState(h_exp, s_exp)
    
    def likelihood(self, y_t, theta : MSSVModelParams, state: MSSVModelState) -> np.ndarray:
        """
        Compute the likelihoods of observation y_t given current states with shape (N,).

        p(y_t | x_t, theta) ~ N(0, exp(h_t))

        Parameters
        ----------
            y_t: float
                Observation at time t.
            theta: MSSVModelParams
                Model parameters.
            state: MSSVModelState
                Current state of size N.
        Returns
        -------
            likelihood: np.ndarray
                Likelihood values with shape (N,).
        """
        h_t = state.h_t
        return norm.pdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))  # scale parameter is standard deviation hence 0.5

    def log_likelihood(self, y_t, theta : MSSVModelParams, state: MSSVModelState) -> np.ndarray:
        """
        Compute the log-likelihood of observation y_t given current state.

        Parameters
        ----------
            y_t: float
                Observation at time t.
            theta: MSSVModelParams
                Model parameters.
            state: MSSVModelState
                Current state of size N.
        Returns
        -------
            log_likelihood: np.ndarray
                Log-likelihood values with shape (N,).
        """
        h_t = state.h_t
        return norm.logpdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))
    
    def state_transition(self, theta : MSSVModelParams, state_prev: MSSVModelState, state_next: MSSVModelState) -> np.ndarray:
        """
        Compute the state transition probability p(x_t | x_{t-1}, theta).

        p(x_t | x_{t-1}, theta) = p(s_t | s_{t-1}, theta) * p(h_t | h_{t-1}, s_t, theta)

        Parameters
        ----------
            theta: MSSVModelParams
                Model parameters.
            state_prev: MSSVModelState
                Previous state of size N.
            state_next: MSSVModelState
                Next state of size N.
        Returns
        -------
            transition_prob: np.ndarray
                Transition probabilities with shape (N,).
        """
        h_prev, s_prev = state_prev.h_t, state_prev.s_t
        h_next, s_next = state_next.h_t, state_next.s_t

        # Regime indices per particle
        idx_prev = np.argmax(s_prev, axis=1)    # (N,)
        idx_next = np.argmax(s_next, axis=1)    # (N,)

        # Regime transition probabilities
        p_s = theta.P[idx_prev, idx_next]       # (N,)

        # Volatility transition
        mu = theta.mu[idx_next]
        phi = theta.phi[idx_next]
        sigma = theta.sigma_eta[idx_next]

        mean_h = mu + phi * (h_prev - mu)       # (N,)
        p_h = norm.pdf(h_next, loc=mean_h, scale=sigma)

        return p_s * p_h                        # (N,)
    
    def log_state_transition(self, theta : MSSVModelParams, state_prev: MSSVModelState, state_next: MSSVModelState) -> np.ndarray:
        """
        Compute the log of the state transition probability log p(x_t | x_{t-1}, theta).

        log p(x_t | x_{t-1}, theta) = log p(s_t | s_{t-1}, theta) + log p(h_t | h_{t-1}, s_t, theta)

        Parameters
        ----------
            theta: MSSVModelParams
                Model parameters.
            state_prev: MSSVModelState
                Previous state of size N.
            state_next: MSSVModelState
                Next state of size N.
        Returns
        -------
            log_transition_prob: np.ndarray
                Log transition probabilities with shape (N,).
        """
        h_prev, s_prev = state_prev.h_t, state_prev.s_t
        h_next, s_next = state_next.h_t, state_next.s_t

        # Regime transition log-probability
        index_prev = np.argmax(s_prev, axis=1)
        index_next = np.argmax(s_next, axis=1)
        log_p_s = np.log(theta.P[index_prev, index_next])

        # Volatility transition log-probability
        mu = theta.mu[index_next]
        phi = theta.phi[index_next]
        sigma = theta.sigma_eta[index_next]

        mean_h = mu + phi * (h_prev - mu)
        log_p_h = norm.logpdf(h_next, loc=mean_h, scale=sigma)

        return log_p_s + log_p_h


