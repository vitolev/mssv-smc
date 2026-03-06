import numpy as np
from scipy.stats import norm, uniform, expon, dirichlet
from scipy.special import logit, expit, logsumexp
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from typing import List, Tuple

EPS = 1e-10

class MSSVModelParams(StateSpaceModelParams):
    """
    Container for MSSV model parameters. If initialized without parameters, sampling from prior is done.
    """
    def __init__(self, rng: np.random.Generator = None,
                 num_regimes: int = None,
                 mu: List[float] = None, 
                 phi: float = None, 
                 sigma_eta: float = None, 
                 P: List[List[float]] = None):

        # Sample from prior if any parameter is None
        if any(param is None for param in [mu, phi, sigma_eta, P]):
            if rng is None or num_regimes is None:
                raise ValueError("RNG and num_regimes must be provided when sampling from prior.")
            self.sample_prior(rng, num_regimes)
        
        # Set provided parameters
        else:
            # mu must be ordered
            mu = np.array(mu)
            if not all(mu[k] < mu[k+1] for k in range(len(mu)-1)):
                raise ValueError("mu parameters must be ordered: mu[0] < mu[1] < ... < mu[K-1]")
            self.mu1 = mu[0]
            self.delta = np.log(np.diff(mu))
            self.mu = mu
            self.phi = phi
            self.sigma_eta = sigma_eta
            self.P = np.array(P)

            if not (len(mu) == len(P)):
                raise ValueError("Parameters mu and P must have the same length.")
            
            if sigma_eta <= 0:
                raise ValueError("Standard deviation sigma_eta must be positive.")
            
            if any(p < 0 for row in P for p in row):
                raise ValueError("Transition probabilities must be non-negative.")
            
            if any(abs(sum(row) - 1.0) > 1e-8 for row in P):
                raise ValueError("Each row of transition matrix P must sum to 1.")
            
        # Ensure P is a valid transition matrix
        self.P = np.clip(self.P, EPS, None)
        self.P /= self.P.sum(axis=1, keepdims=True)

    def _delta_to_mu(self, mu1, delta):
        """
        Convert (mu1, delta) -> ordered mu vector.
        """
        K = len(delta) + 1
        mu = np.zeros(K)
        mu[0] = mu1

        for k in range(1, K):
            mu[k] = mu[k-1] + np.exp(delta[k-1])

        return mu


    def _mu_to_delta(self, mu):
        """
        Convert ordered mu -> (mu1, delta)
        """
        mu1 = mu[0]
        delta = np.log(np.diff(mu))
        return mu1, delta

    def sample_prior(self, rng: np.random.Generator, num_regimes: int):
        """
        Sample model parameters from a prior distribution.

        Parameters
        ----------
            rng: np.random.Generator
                Random number generator for reproducibility.
            num_regimes: int
                Number of regimes (K) in the MSSV model.
        """
        self.mu1 = rng.normal(0,1)
        self.delta = rng.normal(0,1,size=num_regimes-1)
        self.mu = self._delta_to_mu(self.mu1, self.delta)

        self.phi = rng.uniform(0, 1)       # Prior for phi_k in (0,1)
        self.sigma_eta = rng.exponential(1.0)  # Prior for sigma_eta_k > 0

        # Prior for transition matrix P: Dirichlet distribution for each row
        alpha = np.ones(num_regimes)  # Symmetric Dirichlet prior
        self.P = np.array([rng.dirichlet(alpha) for _ in range(num_regimes)])

    def log_prior_density(self) -> float:
        """
        Compute log p(theta) for the MSSV model parameters.
        """
        logp = 0.0

        # ---- mu_k ~ N(0, 1^2) ----
        logp += norm.logpdf(self.mu1, loc=0, scale=1)   # mu1 prior
        logp += np.sum(norm.logpdf(self.delta, loc=0, scale=1))  # delta priors

        # ---- phi_k ~ Uniform(0, 1) ----
        # Explicit check to avoid -inf surprises
        if self.phi <= 0.0 or self.phi >= 1.0:
            return -np.inf
        logp += uniform.logpdf(self.phi, loc=0.0, scale=1.0)

        # ---- sigma_eta_k ~ Exponential(1) ----
        if self.sigma_eta <= 0.0:
            return -np.inf
        logp += expon.logpdf(self.sigma_eta, scale=1.0)

        # ---- Transition matrix rows ~ Dirichlet(1,...,1) ----
        for row in self.P:
            # Dirichlet already enforces positivity and sum-to-1
            logp += dirichlet.logpdf(row, alpha=np.ones(len(row)))

        return logp

    def _normalize_rows_strict(P, eps=EPS):
        P = np.clip(P, eps, None)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def sample_transition(
        self, 
        rng: np.random.Generator,
        step_mu=0.1,  
        step_delta=0.1,  
        step_phi=0.1,
        step_sigma=0.1,
        step_P=20.0
    ) -> 'MSSVModelParams':
        """
        Given the current parameters, sample a new set of parameters by perturbing the current ones.

        Parameters
        ----------
            rng: np.random.Generator
                Random number generator for reproducibility.
            step_mu, step_delta, step_phi, step_sigma, step_P: float
                Step sizes for the proposal distribution for each parameter type. These control how much the new parameters can deviate from the current ones.

        Returns
        -------
            new_params: MSSVModelParams
                New set of parameters sampled from a proposal distribution.
        """

        K = len(self.mu)

        # ---- mu_k : Gaussian RW ----
        mu1 = self.mu1 + rng.normal(0, step_mu)
        delta = self.delta + rng.normal(0, step_delta, size=len(self.delta))
        mu = self._delta_to_mu(mu1, delta)

        # ---- phi_k : reflected RW ----
        L = 0.0
        U = 1.0
        width = U - L  # 1.0
        phi = self.phi + rng.normal(0.0, step_phi)
        # Infinite reflection via modulo folding
        phi = (phi - L) % (2.0 * width)
        if phi > width:
            phi = 2.0 * width - phi
        phi = phi + L

        # ---- sigma_eta_k : log RW ----
        log_sigma = np.log(self.sigma_eta)
        log_sigma_new = log_sigma + rng.normal(0.0, step_sigma)
        sigma_eta = np.exp(log_sigma_new)

        # ---- transition matrix rows : Dirichlet RW ----
        P = np.zeros_like(self.P)
        for k in range(K):
            alpha = step_P * np.clip(self.P[k], EPS, None)
            row = rng.dirichlet(alpha)
            row = np.clip(row, EPS, None)
            row /= row.sum()

            P[k] = row

        return MSSVModelParams(mu=mu, phi=phi, sigma_eta=sigma_eta, P=P)

    def log_transition_density(
        self,
        other: "MSSVModelParams",
        step_mu=0.1,
        step_delta=0.1,
        step_phi=0.1,
        step_sigma=0.1,
        step_P=20.0
    ) -> float:
        """
        Compute log q(other | self) where q is the proposal distribution used in sample_transition.

        Parameters
        ----------
            other: MSSVModelParams
                The parameter set for which we want to compute the log transition density from self.
            step_mu, step_delta, step_phi, step_sigma, step_P: float
                The step sizes used in the proposal distribution for each parameter type.

        Returns
        -------
            logq: float
                The log of the proposal density q(other | self).
        """
        logq = 0.0
        K = len(self.mu)

        # ---- mu_k ----
        logq += norm.logpdf(other.mu1, loc=self.mu1, scale=step_mu)

        logq += np.sum(
            norm.logpdf(other.delta, loc=self.delta, scale=step_delta)
        )

        # ---- phi_k ----
        period = 2.0  # 2 * width where width = 1
        K_images = 2  # truncate infinite sum

        terms = []
        for k in range(-K_images, K_images + 1):
            shift = period * k
            terms.append(
                norm.logpdf(
                    other.phi,
                    loc=self.phi - shift,
                    scale=step_phi,
                )
            )
        logq += logsumexp(terms)

        # ---- sigma_eta_k (log space) ----
        log_self = np.log(self.sigma_eta)
        log_other = np.log(other.sigma_eta)

        logq += norm.logpdf(
            log_other, loc=log_self, scale=step_sigma
        )

        # Jacobian
        logq -= log_other

        # ---- transition matrix ----
        for k in range(K):
            alpha = step_P * np.clip(self.P[k], EPS, None)
            row = np.clip(other.P[k], EPS, None)
            row /= row.sum()

            logq += dirichlet.logpdf(row, alpha=alpha)

        return logq

    def sample_from_data(self, x_traj: list["MSSVModelState"], y: np.ndarray) -> "MSSVModelParams":
        """
        Sample new parameters from the conditional distribution p(theta | x_traj, y).
        For simplicity, we will use a Metropolis-Hastings step here, using the current parameters as the proposal mean.

        Parameters
        ----------
            x_traj: list of MSSVModelState
                The latent trajectory (h_t, s_t) over time.
            y: np.ndarray
                The observed data over time.

        Returns
        -------
            new_params: MSSVModelParams
                New set of parameters sampled from p(theta | x_traj, y).
        """
        #TODO: Implement
        raise NotImplementedError("Parameter sampling from data is not implemented yet.")
        
class MSSVModelState(StateSpaceModelState):
    """
    Container for MSSV model state: (h_t, s_t)
        h_t: continuous latent log-volatility vector
        s_t: one-hot encoded regime vector 
    """
    def __init__(self, h_t: np.ndarray, s_t: np.ndarray):
        self.h_t = h_t  # Log-volatility
        self.s_t = s_t  # Regime (one-hot encoded)

    def __getitem__(self, idx):
        return MSSVModelState(
            h_t=np.array(self.h_t[idx], copy=True),
            s_t=np.array(self.s_t[idx], copy=True)
        )

    def __len__(self):
        return self.h_t.shape[0]
    
    def __repr__(self):
        return f"MSSVModelState(h_t={self.h_t}, s_t={self.s_t})"
    
    def add(self, other: "MSSVModelState") -> "MSSVModelState":
        """
        Add another MSSVModelState to this one. This is a simple element-wise addition of the h_t and s_t components.

        Parameters
        ----------
            other: MSSVModelState
                Another state to add to this one.

        Returns
        -------
            new_state: MSSVModelState
                A new MSSVModelState where h_t and s_t are the sums of the corresponding components of self and other.
        """
        if not isinstance(other, MSSVModelState):
            raise TypeError(f"Other must be an instance of MSSVModelState, got {type(other)}")
        
        new_h_t = np.concatenate((self.h_t, other.h_t), axis=0)
        new_s_t = np.concatenate((self.s_t, other.s_t), axis=0)

        return MSSVModelState(h_t=new_h_t, s_t=new_s_t)
    
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
        h_t | h_{t-1}, s_t ~ N(mu_{s_t} + phi * (h_{t-1} - mu_{s_t}), sigma_eta^2)
        y_t | h_t ~ N(0, exp(h_t))
    """
    params_type = MSSVModelParams
    state_type = MSSVModelState

    def _check_params(self, theta):
        if not isinstance(theta, self.params_type):
            raise TypeError(
                f"Expected params of type {self.params_type.__name__}, "
                f"got {type(theta).__name__}"
            )

    def _check_state(self, state):
        if not isinstance(state, self.state_type):
            raise TypeError(
                f"Expected state of type {self.state_type.__name__}, "
                f"got {type(state).__name__}"
            )

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
        self._check_params(theta)
        self._check_state(state)

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
        self._check_params(theta)

        K = len(theta.mu)
        s0 = np.zeros((size, K))   # Initialize regime array as one-hot encoding

        # Uniformly sample initial regimes
        regimes = self.rng.integers(0, K, size=size)
        s0[np.arange(size), regimes] = 1

        # Sample initial log-volatilities based on regimes
        h0 = self.rng.normal(theta.mu[regimes], theta.sigma_eta)   # np.random.normal uses stddev as second parameter

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
        self._check_params(theta)
        self._check_state(state)
        
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

        h_t = mu + theta.phi * (h_prev - mu) + self.rng.normal(size=N, scale=theta.sigma_eta)
    
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
        self._check_params(theta)
        self._check_state(state)

        h_prev, s_prev = state.h_t, state.s_t

        # Expected regime distribution
        s_exp = s_prev @ theta.P                                                        

        # Expected log-volatility
        # shape tricks:
        # h_prev[:, None]  -> (N, 1)
        # mu[None, :]      -> (1, K)
        h_exp = np.sum(
            s_exp * (theta.mu + theta.phi * (h_prev[:, None] - theta.mu)),
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
        self._check_params(theta)
        self._check_state(state)

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
        self._check_params(theta)
        self._check_state(state)

        h_t = state.h_t
        return norm.logpdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))
    
    def transition_density(self, theta : MSSVModelParams, state_prev: MSSVModelState, state_next: MSSVModelState) -> np.ndarray:
        """
        Compute the state transition density p(x_t | x_{t-1}, theta).

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
        self._check_params(theta)
        self._check_state(state_prev)
        self._check_state(state_next)

        h_prev, s_prev = state_prev.h_t, state_prev.s_t
        h_next, s_next = state_next.h_t, state_next.s_t

        # Regime indices per particle
        idx_prev = np.argmax(s_prev, axis=1)    # (N,)
        idx_next = np.argmax(s_next, axis=1)    # (N,)

        # Regime transition probabilities
        p_s = theta.P[idx_prev, idx_next]       # (N,)

        # Volatility transition
        mu = theta.mu[idx_next]

        mean_h = mu + theta.phi * (h_prev - mu)       # (N,)
        p_h = norm.pdf(h_next, loc=mean_h, scale=theta.sigma_eta)

        return p_s * p_h                        # (N,)
    
    def log_transition_density(self, theta : MSSVModelParams, state_prev: MSSVModelState, state_next: MSSVModelState) -> np.ndarray:
        """
        Compute the log of the state transition density log p(x_t | x_{t-1}, theta).

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
        self._check_params(theta)
        self._check_state(state_prev)
        self._check_state(state_next)
        
        h_prev, s_prev = state_prev.h_t, state_prev.s_t
        h_next, s_next = state_next.h_t, state_next.s_t

        # Regime transition log-probability
        index_prev = np.argmax(s_prev, axis=1)
        index_next = np.argmax(s_next, axis=1)
        log_p_s = np.log(theta.P[index_prev, index_next])

        # Volatility transition log-probability
        mu = theta.mu[index_next]

        mean_h = mu + theta.phi * (h_prev - mu)
        log_p_h = norm.logpdf(h_next, loc=mean_h, scale=theta.sigma_eta)

        return log_p_s + log_p_h


