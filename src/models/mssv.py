import numpy as np
from scipy.stats import norm, dirichlet, beta, gamma, truncnorm
from scipy.special import logit, expit
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState, StateSpaceModelPrior, StateSpaceModelProposal
from typing import List, Tuple
from dataclasses import dataclass

EPS = 1e-10

# =========================
# PARAMETER CONTAINER
# =========================
@dataclass(frozen=True)
class MSSVParams(StateSpaceModelParams):
    """
    Parameters for the MSSV model.
    """
    mu1: float
    delta: np.ndarray
    phi: float
    sigma_eta: float
    P: np.ndarray

    def __post_init__(self):
        self._validate()

    @classmethod
    def from_mu(
        cls,
        mu: np.ndarray,
        phi: float,
        sigma_eta: float,
        P: np.ndarray,
    ) -> "MSSVParams":
        """
        Alternative constructor that takes vector of regime means mu (must be strictly increasing) instead of mu1 and delta. 
        """
        mu = np.asarray(mu, dtype=float)
        P = np.asarray(P, dtype=float)

        if mu.ndim != 1:
            raise ValueError("mu must be a 1D array-like")
        if len(mu) < 1:
            raise ValueError("mu must have at least one element")

        mu1 = mu[0]

        if len(mu) == 1:
            delta = np.array([])
        else:
            diff = np.diff(mu)
            if np.any(diff <= 0):
                raise ValueError("mu must be strictly increasing")

            delta = np.log(diff)

        return cls(mu1, delta, phi, sigma_eta, P)

    @property
    def mu(self) -> np.ndarray:
        increments = np.exp(self.delta)
        return np.concatenate(([self.mu1], self.mu1 + np.cumsum(increments)))

    @property
    def K(self) -> int:
        return len(self.delta) + 1

    def _validate(self):
        if self.sigma_eta <= 0:
            raise ValueError("sigma_eta must be > 0")

        if not (-1 < self.phi < 1):
            raise ValueError("phi must be in (-1,1)")

        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError("P must be square")

        if self.P.shape[0] != self.K:
            raise ValueError("P dimension must match number of regimes")

        if np.any(self.P < 0):
            raise ValueError("P must be non-negative")

        row_sums = self.P.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Rows of P must sum to 1")

    def copy(self):
        """
        Create a copy of the MSSVParams instance. 
        """
        return MSSVParams(
            mu1=self.mu1,
            delta=np.array(self.delta, copy=True),
            phi=self.phi,
            sigma_eta=self.sigma_eta,
            P=np.array(self.P, copy=True)
        )


# =========================
# PRIOR
# =========================
class MSSVPrior(StateSpaceModelPrior):
    """
    Prior distribution for the MSSV model parameters.
    """
    def __init__(
        self,
        mu_mean=0.0,
        mu_sd=1.0,
        diff_mean=0.0,
        diff_sd=2.0,
        phi_a=20.0,
        phi_b=2.0,
        sigma_eta_a=2.0,
        sigma_eta_b=5.0,
        P_diag=2.5,
        P_base=1.5,
    ):
        self.mu_mean = mu_mean
        self.mu_sd = mu_sd
        self.diff_mean = diff_mean
        self.diff_sigma = diff_sd
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.sigma_eta_a = sigma_eta_a
        self.sigma_eta_b = sigma_eta_b
        self.P_diag = P_diag
        self.P_base = P_base

    def sample(self, rng: np.random.Generator, K: int) -> MSSVParams:
        mu1 = rng.normal(self.mu_mean, self.mu_sd)
        a = (0 - self.diff_mean) / self.diff_sigma   # lower bound
        b = np.inf             # upper bound
        diff = truncnorm.rvs(a, b, loc=self.diff_mean, scale=self.diff_sigma, random_state=rng, size=K - 1)
        mu = np.concatenate(([mu1], mu1 + np.cumsum(diff)))

        u = rng.beta(self.phi_a, self.phi_b)
        phi = 2 * u - 1

        sigma_eta = rng.gamma(self.sigma_eta_a, scale=1.0 / self.sigma_eta_b)

        P = []
        for i in range(K):
            alpha = self.P_base * np.ones(K)
            alpha[i] += self.P_diag
            P.append(rng.dirichlet(alpha))
        P = np.array(P)

        return MSSVParams.from_mu(mu, phi, sigma_eta, P)

    def logpdf(self, params: MSSVParams) -> float:
        logp = 0.0

        logp += norm.logpdf(params.mu1, self.mu_mean, self.mu_sd)

        delta = params.delta
        diff = np.exp(delta)

        a = (0 - self.diff_mean) / self.diff_sigma
        b = np.inf

        logp += np.sum(
            truncnorm.logpdf(diff, a, b, loc=self.diff_mean, scale=self.diff_sigma)
            + delta   # Jacobian term
        )

        u = (params.phi + 1) / 2
        logp += beta.logpdf(u, self.phi_a, self.phi_b) - np.log(2)

        logp += gamma.logpdf(
            params.sigma_eta,
            a=self.sigma_eta_a,
            scale=1.0 / self.sigma_eta_b,
        )

        for i, row in enumerate(params.P):
            alpha = self.P_base * np.ones(params.K)
            alpha[i] += self.P_diag
            row_safe = np.clip(row, EPS, 1.0)
            row_safe = row_safe / row_safe.sum()
            logp += dirichlet.logpdf(row_safe, alpha)

        return logp


# =========================
# PROPOSAL (MCMC)
# =========================
class MSSVProposal(StateSpaceModelProposal):
    """
    Proposal distribution for MCMC sampling of MSSV model parameters.
    """
    def __init__(
        self,
        step_mu=0.1,
        step_delta=0.1,
        step_phi=0.1,
        step_sigma=0.1,
        step_P=20.0,
    ):
        self.step_mu = step_mu
        self.step_delta = step_delta
        self.step_phi = step_phi
        self.step_sigma = step_sigma
        self.step_P = step_P

    def sample(self, rng: np.random.Generator, p: MSSVParams) -> MSSVParams:
        mu1 = p.mu1 + rng.normal(0, self.step_mu)
        delta = p.delta + rng.normal(0, self.step_delta, size=len(p.delta))

        # phi (logit transform)
        z = logit((p.phi + 1) / 2)
        z_new = z + rng.normal(0, self.step_phi)
        phi = 2 * expit(z_new) - 1

        # sigma (log space)
        log_sigma = np.log(p.sigma_eta)
        sigma_eta = np.exp(log_sigma + rng.normal(0, self.step_sigma))

        # transition matrix
        P = np.empty_like(p.P)
        for k in range(p.K):
            alpha = self.step_P * np.clip(p.P[k], EPS, None)
            P[k] = rng.dirichlet(alpha)

        return MSSVParams(mu1, delta, phi, sigma_eta, P)

    def logpdf(self, from_p: MSSVParams, to_p: MSSVParams) -> float:
        logq = 0.0

        logq += norm.logpdf(to_p.mu1, from_p.mu1, self.step_mu)
        logq += np.sum(
            norm.logpdf(to_p.delta, from_p.delta, self.step_delta)
        )

        # phi
        z_from = logit((from_p.phi + 1) / 2)
        z_to = logit((to_p.phi + 1) / 2)
        logq += norm.logpdf(z_to, z_from, self.step_phi)
        logq += np.log(2) - np.log(1 - to_p.phi**2)

        # sigma
        log_from = np.log(from_p.sigma_eta)
        log_to = np.log(to_p.sigma_eta)
        logq += norm.logpdf(log_to, log_from, self.step_sigma)
        logq -= log_to

        # P
        for k in range(from_p.K):
            alpha = self.step_P * np.clip(from_p.P[k], EPS, None)

            row = np.clip(to_p.P[k], EPS, 1.0)
            row = row / row.sum()

            logq += dirichlet.logpdf(row, alpha)

        return logq
       
# =========================
# STATE
# =========================
class MSSVState(StateSpaceModelState):
    """
    Container for MSSV model state: (h_t, s_t)
        h_t: continuous latent log-volatility vector
        s_t: one-hot encoded regime vector 
    """
    def __init__(self, h_t: np.ndarray, s_t: np.ndarray):
        self.h_t = h_t  # Log-volatility
        self.s_t = s_t  # Regime (one-hot encoded)

    def __getitem__(self, idx):
        return MSSVState(
            h_t=np.array(self.h_t[idx], copy=True),
            s_t=np.array(self.s_t[idx], copy=True)
        )

    def __len__(self):
        return self.h_t.shape[0]
    
    def __repr__(self):
        return f"MSSVState(h_t={self.h_t}, s_t={self.s_t})"
    
    def add(self, other: "MSSVState") -> "MSSVState":
        """
        Add another MSSVState to this one. This is a simple element-wise addition of the h_t and s_t components.

        Parameters
        ----------
            other: MSSVState
                Another state to add to this one.

        Returns
        -------
            new_state: MSSVState
                A new MSSVState where h_t and s_t are the sums of the corresponding components of self and other.
        """
        if not isinstance(other, MSSVState):
            raise TypeError(f"Other must be an instance of MSSVState, got {type(other)}")
        
        new_h_t = np.concatenate((self.h_t, other.h_t), axis=0)
        new_s_t = np.concatenate((self.s_t, other.s_t), axis=0)

        return MSSVState(h_t=new_h_t, s_t=new_s_t)
    
# =========================
# MODEL
# =========================
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
    params_type = MSSVParams
    state_type = MSSVState
    prior_type = MSSVPrior
    proposal_type = MSSVProposal

    def __init__(self, rng=None):
        super().__init__(rng)

    def sample_observation(self, theta : MSSVParams, state: MSSVState) -> np.ndarray:
        """
        Sample an observation y_t given state x_t = (h_t, s_t) and parameters theta.

        y_t ~ p(y_t | x_t, theta)

        Parameters
        ----------
            theta: MSSVParams
                Model parameters.
            state: MSSVState
                Current state of size N. 
        Returns
        -------
            y_t: np.ndarray
                Sampled observation with shape (N,).
        """
        h_t = state.h_t
        return self.rng.normal(0.0, np.exp(0.5 * h_t))

    def sample_initial_state(self, theta : MSSVParams, size: int = 1) -> MSSVState:
        """
        Sample the initial state x_0 = (h_0, s_0) given initial parameters theta.

        x_0 ~ p(x_0 | theta)

        Parameters
        ----------
            theta: MSSVParams
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
        var = theta.sigma_eta ** 2 / (1 - theta.phi ** 2)  # Stationary variance of AR(1) process
        h0 = self.rng.normal(theta.mu[regimes], np.sqrt(var))   # np.random.normal uses stddev as second parameter

        return MSSVState(h0, s0)

    def sample_next_state(self, theta : MSSVParams, state: MSSVState) -> MSSVState:
        """
        Sample the next state x_t = (h_t, s_t) given previous state x_t-1 and new parameters theta.

        x_t ~ p(x_t | x_{t-1}, theta)

        Parameters
        ----------
            theta: MSSVParams
                Model parameters.
            state: MSSVState
                Previous state of size N.
        Returns
        -------
            state: MSSVState
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

        h_t = mu + theta.phi * (h_prev - mu) + self.rng.normal(size=N, scale=theta.sigma_eta)
    
        return MSSVState(h_t, s_t)
    
    def expected_next_state(self, theta : MSSVParams, state: MSSVState) -> MSSVState:
        """
        Compute the expected next state given current state and parameters theta.

        E[x_t | x_{t-1}, theta]

        Parameters
        ----------
            theta: MSSVParams
                Model parameters.
            state: MSSVState
                Current state of size N.
        Returns
        -------
            state: MSSVState
                Expected next state of size N.
        """
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

        return MSSVState(h_exp, s_exp)
    
    def likelihood(self, y_t, theta : MSSVParams, state: MSSVState) -> np.ndarray:
        """
        Compute the likelihoods of observation y_t given current states with shape (N,).

        p(y_t | x_t, theta) ~ N(0, exp(h_t))

        Parameters
        ----------
            y_t: float
                Observation at time t.
            theta: MSSVParams
                Model parameters.
            state: MSSVState
                Current state of size N.
        Returns
        -------
            likelihood: np.ndarray
                Likelihood values with shape (N,).
        """
        h_t = state.h_t
        return norm.pdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))  # scale parameter is standard deviation hence 0.5

    def log_likelihood(self, y_t, theta : MSSVParams, state: MSSVState) -> np.ndarray:
        """
        Compute the log-likelihood of observation y_t given current state.

        Parameters
        ----------
            y_t: float
                Observation at time t.
            theta: MSSVParams
                Model parameters.
            state: MSSVState
                Current state of size N.
        Returns
        -------
            log_likelihood: np.ndarray
                Log-likelihood values with shape (N,).
        """
        h_t = state.h_t
        return norm.logpdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))
    
    def transition_density(self, theta : MSSVParams, state_prev: MSSVState, state_next: MSSVState) -> np.ndarray:
        """
        Compute the state transition density p(x_t | x_{t-1}, theta).

        p(x_t | x_{t-1}, theta) = p(s_t | s_{t-1}, theta) * p(h_t | h_{t-1}, s_t, theta)

        Parameters
        ----------
            theta: MSSVParams
                Model parameters.
            state_prev: MSSVState
                Previous state of size N.
            state_next: MSSVState
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

        mean_h = mu + theta.phi * (h_prev - mu)       # (N,)
        p_h = norm.pdf(h_next, loc=mean_h, scale=theta.sigma_eta)

        return p_s * p_h                        # (N,)
    
    def log_transition_density(self, theta : MSSVParams, state_prev: MSSVState, state_next: MSSVState) -> np.ndarray:
        """
        Compute the log of the state transition density log p(x_t | x_{t-1}, theta).

        log p(x_t | x_{t-1}, theta) = log p(s_t | s_{t-1}, theta) + log p(h_t | h_{t-1}, s_t, theta)

        Parameters
        ----------
            theta: MSSVParams
                Model parameters.
            state_prev: MSSVState
                Previous state of size N.
            state_next: MSSVState
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

        mean_h = mu + theta.phi * (h_prev - mu)
        log_p_h = norm.logpdf(h_next, loc=mean_h, scale=theta.sigma_eta)

        return log_p_s + log_p_h


