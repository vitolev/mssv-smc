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
# STATE
# =========================
class MSSVState(StateSpaceModelState):
    """
    Container for MSSV model state: (h_t, s_t)
        h_t: continuous latent log-volatility vector
        s_t: one-hot encoded regime vector 
    """
    def __init__(self, h_t: np.ndarray, s_t: np.ndarray):
        if len(h_t) != len(s_t):
            raise ValueError(f"Length mismatch: h_t has length {len(h_t)}, s_t has length {len(s_t)}")
        self.h_t = h_t  # Log-volatility
        self.s_t = s_t  # Regime (one-hot encoded)

    def __getitem__(self, idx):
        return MSSVState(
            h_t=np.atleast_1d(self.h_t[idx]),
            s_t=np.atleast_2d(self.s_t[idx])
        )
    
    def __setitem__(self, idx, value: "MSSVState"):
        if not isinstance(value, MSSVState):
            raise TypeError(f"Value must be an instance of MSSVState, got {type(value)}")
        
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            expected_len = len(range(start, stop, step))
        elif np.isscalar(idx):
            expected_len = 1
        else:
            # fancy indexing (list / array)
            expected_len = len(idx)

        # --- Check length ---
        if len(value) != expected_len:
            raise ValueError(
                f"Length mismatch: expected {expected_len}, got {len(value)}"
            )

        # Assign
        if expected_len == 1:
            self.h_t[idx] = value.h_t[0]
        else:
            self.h_t[idx] = value.h_t
        self.s_t[idx] = value.s_t

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
        params
    ):
        self.mode = params["mode"]  # "rw" or "informed"
        default_params = {
            "rw": {
                "step_mu": 0.1,
                "step_delta": 0.1,
                "step_phi": 0.1,
                "step_sigma": 0.1,
                "step_P": 20.0,
            },
            "informed": {
                "step_mu": 0.05,
                "step_delta": 0.05,
                "step_phi": 0.02,
                "step_sigma": 0.02,
                "step_P": 20.0,
                "lam": 0.2,
            },
        }

        # allow user overrides
        for k in params:
            if k in default_params:
                default_params[k].update(params[k])

        self.params = default_params

    def _sample_rw(self, rng: np.random.Generator, p: MSSVParams) -> MSSVParams:
        cfg = self.params["rw"]
        mu1 = p.mu1 + rng.normal(0, cfg["step_mu"])
        delta = p.delta + rng.normal(0, cfg["step_delta"], size=len(p.delta))

        # phi (logit transform)
        z = logit((p.phi + 1) / 2)
        z_new = z + rng.normal(0, cfg["step_phi"])
        phi = 2 * expit(z_new) - 1

        # sigma (log space)
        log_sigma = np.log(p.sigma_eta)
        sigma_eta = np.exp(log_sigma + rng.normal(0, cfg["step_sigma"]))

        # transition matrix
        P = np.empty_like(p.P)
        for k in range(p.K):
            alpha = cfg["step_P"] * np.clip(p.P[k], EPS, None)
            P[k] = rng.dirichlet(alpha)

        return MSSVParams(mu1, delta, phi, sigma_eta, P)

    def _logpdf_rw(self, from_p: MSSVParams, to_p: MSSVParams) -> float:
        cfg = self.params["rw"]
        logq = 0.0

        logq += norm.logpdf(to_p.mu1, from_p.mu1, cfg["step_mu"])
        logq += np.sum(
            norm.logpdf(to_p.delta, from_p.delta, cfg["step_delta"])
        )

        # phi
        z_from = logit((from_p.phi + 1) / 2)
        z_to = logit((to_p.phi + 1) / 2)
        logq += norm.logpdf(z_to, z_from, cfg["step_phi"])
        logq += np.log(2) - np.log(1 - to_p.phi**2)

        # sigma
        log_from = np.log(from_p.sigma_eta)
        log_to = np.log(to_p.sigma_eta)
        logq += norm.logpdf(log_to, log_from, cfg["step_sigma"])
        logq -= log_to

        # P
        for k in range(from_p.K):
            alpha = cfg["step_P"] * np.clip(from_p.P[k], EPS, None)

            row = np.clip(to_p.P[k], EPS, 1.0)
            row = row / row.sum()

            logq += dirichlet.logpdf(row, alpha)

        return logq
       
    def _estimate_P(self, z, K):
        counts = np.zeros((K, K))

        for t in range(1, len(z)):
            counts[z[t-1], z[t]] += 1

        # smoothing to avoid zeros
        alpha = counts + 1.0

        P_hat = alpha / alpha.sum(axis=1, keepdims=True)
        return P_hat, alpha

    def _estimate_theta_from_traj_regime(self, h, z, K):
        T = h.shape[0]

        mu_hat = np.zeros(K)

        # --- estimate mu_k ---
        for k in range(K):
            idx = (z == k)
            if np.sum(idx) < 5:
                # If we have too few points in this regime, just use the overall mean as a fallback to avoid extreme estimates
                mu_hat[k] = np.mean(h)
            else:
                mu_hat[k] = np.mean(h[idx])

        # Sort mu_hat to ensure identifiability (enforce ordering)
        mu_hat = np.sort(mu_hat)
        eps = 1e-4
        for k in range(1, K):
            if mu_hat[k] <= mu_hat[k-1]:
                mu_hat[k] = mu_hat[k-1] + eps

        # --- estimate phi ---
        x = []
        y = []

        for t in range(1, T):
            k = z[t]
            x.append(h[t-1] - mu_hat[k])
            y.append(h[t] - mu_hat[k])

        x = np.array(x)
        y = np.array(y)

        # Simple OLS regression to estimate phi
        num = np.sum(x * y)
        den = np.sum(x * x) + 1e-8
        phi_hat = num / den

        # --- estimate shared sigma ---
        resid = y - phi_hat * x
        sigma_hat = np.sqrt(np.mean(resid**2) + 1e-8)

        return mu_hat, phi_hat, sigma_hat

    def _sample_informed(self, rng: np.random.Generator, traj: List[MSSVState]) -> MSSVParams:
        cfg = self.params["informed"]
        h = np.array([state.h_t for state in traj])  # shape (T, 1)
        h = h.squeeze()  # shape (T,)
        s = np.array([state.s_t for state in traj])  # shape (T, 1, K)
        K = s.shape[2]
        z = np.argmax(s, axis=2)  # shape (T, 1)
        z = z.squeeze()  # shape (T,)

        mu_hat, phi_hat, sigma_hat = self._estimate_theta_from_traj_regime(h, z, K)
        P_hat, alpha = self._estimate_P(z, K)

        # --- mu ---
        mu1 = mu_hat[0] + rng.normal(0, cfg["step_mu"])
        delta_hat = np.log(np.diff(mu_hat))  
        delta = delta_hat + rng.normal(0, cfg["step_delta"], size=len(delta_hat))

        # --- phi ---
        z_phi = logit((phi_hat + 1) / 2)
        z_new = z_phi + rng.normal(0, cfg["step_phi"])
        phi = 2 * expit(z_new) - 1

        # --- sigma ---
        log_sigma = np.log(sigma_hat)
        sigma_eta = np.exp(log_sigma + rng.normal(0, cfg["step_sigma"]))

        # --- P ---
        P = np.zeros((K, K))
        for k in range(K):
            P[k] = rng.dirichlet(alpha[k])

        return MSSVParams(mu1, delta, phi, sigma_eta, P)

    def _logpdf_informed(self, p: MSSVParams, traj: List[MSSVState]) -> float:
        cfg = self.params["informed"]

        h = np.array([state.h_t for state in traj]).squeeze()
        s = np.array([state.s_t for state in traj])
        K = s.shape[2]
        z = np.argmax(s, axis=2).squeeze()

        mu_hat, phi_hat, sigma_hat = self._estimate_theta_from_traj_regime(h, z, K)
        _, alpha = self._estimate_P(z, K)

        logq = 0.0

        # mu1
        logq += norm.logpdf(p.mu1, mu_hat[0], cfg["step_mu"])

        # delta 
        delta_hat = np.log(np.diff(mu_hat))
        logq += np.sum(norm.logpdf(p.delta, delta_hat, cfg["step_delta"]))

        # phi
        z_hat = logit((phi_hat + 1) / 2)
        z = logit((p.phi + 1) / 2)

        logq += norm.logpdf(z, z_hat, cfg["step_phi"])
        logq += np.log(2) - np.log(1 - p.phi**2)

        # sigma
        log_sigma_hat = np.log(sigma_hat)
        log_sigma = np.log(p.sigma_eta)

        logq += norm.logpdf(log_sigma, log_sigma_hat, cfg["step_sigma"])
        logq -= log_sigma

        # P
        for k in range(K):
            row = np.clip(p.P[k], EPS, 1.0)
            row = row / row.sum()
            logq += dirichlet.logpdf(row, alpha[k])

        return logq
    
    def sample(self, rng: np.random.Generator, from_p: MSSVParams = None, traj: List[MSSVState] = None) -> MSSVParams:
        if self.mode == "rw":
            if from_p is None:
                raise ValueError("from_p must be provided for random walk proposal")
            return self._sample_rw(rng, from_p)
        elif self.mode == "informed":
            if traj is None:
                raise ValueError("traj must be provided for informed proposal")
            return self._sample_informed(rng, traj)
        else:
            raise ValueError(f"Unknown proposal mode: {self.mode}")
        
    def logpdf(self, from_p: MSSVParams, to_p: MSSVParams = None, traj: List[MSSVState] = None) -> float:
        if self.mode == "rw":
            if from_p is None:
                raise ValueError("from_p must be provided for random walk proposal")
            return self._logpdf_rw(from_p, to_p)
        elif self.mode == "informed":
            if traj is None:
                raise ValueError("traj must be provided for informed proposal")
            return self._logpdf_informed(to_p, traj)
        else:
            raise ValueError(f"Unknown proposal mode: {self.mode}")
    
    
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

    def initial_state_density(self, theta : MSSVParams, state: MSSVState) -> np.ndarray:
        """
        Compute the density of the initial state p(x_0 | theta).

        p(x_0 | theta) = p(s_0 | theta) * p(h_0 | s_0, theta)

        Parameters
        ----------
            theta: MSSVParams
                Model parameters.
            state: MSSVState
                Initial state of size N.
        Returns
        -------
            initial_density: np.ndarray
                Initial state densities with shape (N,).
        """
        h_0, s_0 = state.h_t, state.s_t

        # Regime indices per particle
        idx = np.argmax(s_0, axis=1)    # (N,)

        # Regime probabilities (uniform)
        p_s = 1.0 / theta.K

        # Volatility distribution
        var = theta.sigma_eta ** 2 / (1 - theta.phi ** 2)
        mu = theta.mu[idx]

        p_h = norm.pdf(h_0, loc=mu, scale=np.sqrt(var))

        return p_s * p_h                        # (N,)

    def log_initial_state_density(self, theta : MSSVParams, state: MSSVState) -> np.ndarray:
        """
        Compute the log of the density of the initial state log p(x_0 | theta).

        log p(x_0 | theta) = log p(s_0 | theta) + log p(h_0 | s_0, theta)

        Parameters
        ----------
            theta: MSSVParams
                Model parameters.
            state: MSSVState
                Initial state of size N.
        Returns
        -------
            log_initial_density: np.ndarray
                Log initial state densities with shape (N,).
        """
        h_0, s_0 = state.h_t, state.s_t

        # Regime indices per particle
        idx = np.argmax(s_0, axis=1)    # (N,)

        # Regime probabilities (uniform)
        log_p_s = -np.log(theta.K)

        # Volatility distribution
        var = theta.sigma_eta ** 2 / (1 - theta.phi ** 2)
        mu = theta.mu[idx]

        log_p_h = norm.logpdf(h_0, loc=mu, scale=np.sqrt(var))

        return log_p_s + log_p_h