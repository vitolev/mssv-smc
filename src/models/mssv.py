import numpy as np
from scipy.stats import norm, dirichlet, beta, invgamma, truncnorm, multivariate_normal
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
    eta2: float
    P: np.ndarray

    def __post_init__(self):
        self._validate()

    @classmethod
    def from_mu(
        cls,
        mu: np.ndarray,
        phi: float,
        eta2: float,
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

        return cls(mu1, delta, phi, eta2, P)

    @property
    def mu(self) -> np.ndarray:
        increments = np.exp(self.delta)
        return np.concatenate(([self.mu1], self.mu1 + np.cumsum(increments)))

    @property
    def K(self) -> int:
        return len(self.delta) + 1

    def _validate(self):
        if self.eta2 <= 0:
            raise ValueError("eta2 must be > 0")

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
            eta2=self.eta2,
            P=np.array(self.P, copy=True)
        )

    def to_unconstrained(self) -> np.ndarray:
        """
        Convert the constrained parameters to an unconstrained vector representation for MCMC proposals.

        The transformation is as follows:
            - mu1: unconstrained
            - delta: unconstrained
            - phi: unconstrained (arctanh transform)
            - eta2: unconstrained (log transform)
            - P: unconstrained (logits for each row)
        """
        z = []
        for i in range(self.K):
            row = np.clip(self.P[i], EPS, 1.0)
            row = row / row.sum()

            logits = np.log(row[:-1]) - np.log(row[-1])

            z.extend(logits)

        return np.array([
            self.mu1,
            *self.delta,
            np.arctanh(self.phi),
            np.log(self.eta2),
            *z
        ])

    def to_vector(self) -> np.ndarray:
        """
        Convert parameters to a vector representation. Used for storing samples in a consistent format. 
        """
        return np.concatenate((
            [self.mu1],
            self.delta,
            [self.phi],
            [self.eta2],
            self.P.flatten()
        ))

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

    def copy(self) -> "MSSVState":
        """
        Create a copy of this MSSVState instance.

        Returns
        -------
            new_state: MSSVState
                A new MSSVState with the same h_t and s_t values as this one.
        """
        return MSSVState(
            h_t=np.array(self.h_t, copy=True),
            s_t=np.array(self.s_t, copy=True)
        )
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert the MSSVState to a single numpy array for easier storage or processing. 
        It forms a matrix of shape (N, 1 + K) where the first column is h_t and the next K columns are the one-hot encoded s_t.
        """
        return np.hstack((self.h_t.reshape(-1, 1), self.s_t))

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
        eta2_a=2.001,
        eta2_b=1.0,
        P_diag=2.5,
        P_base=1.5,
    ):
        self.mu_mean = mu_mean
        self.mu_sd = mu_sd
        self.diff_mean = diff_mean
        self.diff_sigma = diff_sd
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.eta2_a = eta2_a
        self.eta2_b = eta2_b
        self.P_diag = P_diag
        self.P_base = P_base

    def sample(self, rng: np.random.Generator, K: int) -> MSSVParams:
        mu1 = rng.normal(self.mu_mean, self.mu_sd)
        lo = (0 - self.diff_mean) / self.diff_sigma
        hi = np.inf     # no need to transform as it is inf
        diff = truncnorm.rvs(lo, hi, loc=self.diff_mean, scale=self.diff_sigma, random_state=rng, size=K - 1)
        mu = np.concatenate(([mu1], mu1 + np.cumsum(diff)))

        u = rng.beta(self.phi_a, self.phi_b)
        phi = 2 * u - 1

        temp = rng.gamma(shape=self.eta2_a, scale=1.0 / self.eta2_b)
        eta2 = 1.0 / temp   # Inverse-gamma distribution

        P = []
        for i in range(K):
            alpha = self.P_base * np.ones(K)
            alpha[i] += self.P_diag
            P.append(rng.dirichlet(alpha))
        P = np.array(P)

        return MSSVParams.from_mu(mu, phi, eta2, P)

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

        logp += invgamma.logpdf(
            params.eta2,
            a=self.eta2_a,
            scale=self.eta2_b,
        )

        for i, row in enumerate(params.P):
            alpha = self.P_base * np.ones(params.K)
            alpha[i] += self.P_diag
            row_safe = np.clip(row, EPS, 1.0)
            row_safe = row_safe / row_safe.sum()
            logp += dirichlet.logpdf(row_safe, alpha)

        return logp

# =========================
# PROPOSAL
# =========================
class MSSVProposal(StateSpaceModelProposal):
    """
    Proposal distribution for MCMC sampling of MSSV model parameters.
    """
    def __init__(
        self,
        params,
        prior: MSSVPrior = None
    ):
        self.mode = params["mode"]  # "rw", "conditional" or "independent"
        if self.mode not in ["rw", "conditional", "independent"]:
            raise ValueError(f"Unknown proposal mode: {self.mode}")
        
        default_params = {
            "rw": {
                "step_mu": 0.1,
                "step_delta": 0.1,
                "step_phi": 0.1,
                "step_eta2": 0.1,
                "step_P": 20.0,
            },
            "conditional": {
                "K": 2,
            },
            "independent": {
                "mean": 1.0,
                "covariance": 1.0
            }
        }

        # allow user overrides
        for k in params:
            if k in default_params[self.mode]:
                default_params[self.mode][k] = params[k]

        self.params = default_params
        self.prior = prior  # Optional prior distribution used for conditional proposals

    def update_params(self, new_params):
        for k in new_params:
            if k in self.params[self.mode]:
                self.params[self.mode][k] = new_params[k]

    def _sample_rw(self, rng: np.random.Generator, p: MSSVParams) -> MSSVParams:
        cfg = self.params["rw"]
        mu1 = p.mu1 + rng.normal(0, cfg["step_mu"])
        delta = p.delta + rng.normal(0, cfg["step_delta"], size=len(p.delta))

        # phi (logit transform)
        z = logit((p.phi + 1) / 2)
        z_new = z + rng.normal(0, cfg["step_phi"])
        phi = 2 * expit(z_new) - 1

        # sigma2 (log space)
        log_eta2 = np.log(p.eta2)
        eta2 = np.exp(log_eta2 + rng.normal(0, cfg["step_eta2"]))

        # transition matrix
        P = np.empty_like(p.P)
        for k in range(p.K):
            alpha = cfg["step_P"] * np.clip(p.P[k], EPS, None)
            P[k] = rng.dirichlet(alpha)

        return MSSVParams(mu1, delta, phi, eta2, P)

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

        # eta2
        log_from = np.log(from_p.eta2)
        log_to = np.log(to_p.eta2)
        logq += norm.logpdf(log_to, log_from, cfg["step_eta2"])
        logq -= log_to

        # P
        for k in range(from_p.K):
            alpha = cfg["step_P"] * np.clip(from_p.P[k], EPS, None)

            row = np.clip(to_p.P[k], EPS, 1.0)
            row = row / row.sum()

            logq += dirichlet.logpdf(row, alpha)

        return logq
       
    def _sample_conditional(self, rng: np.random.Generator, p: MSSVParams, traj: List[MSSVState]) -> MSSVParams:
        cfg = self.params["conditional"]

        T_plus_1 = len(traj)
        K = len(traj[0].s_t[0])

        counts = np.zeros((K, K))               # Count matrix for transitions between regimes
        regime_sets = [[] for _ in range(K)]    # List of lists to store time indices for each regime
        for t in range(1, T_plus_1):
            s_prev = traj[t-1].s_t[0]
            s_curr = traj[t].s_t[0]

            prev_idx = np.argmax(s_prev)
            curr_idx = np.argmax(s_curr)

            counts[prev_idx, curr_idx] += 1
            regime_sets[curr_idx].append(t)

        # -----------------------
        # Conditional on P
        # -----------------------
        P = []
        for i in range(K):
            alpha = self.prior.P_base * np.ones(K)
            alpha[i] += self.prior.P_diag
            alpha += counts[i]
            P.append(rng.dirichlet(alpha))
        P = np.array(P)

        # -----------------------
        # Conditional on eta2
        # -----------------------
        e_list = [traj[t].h_t[0] - p.mu[traj[t].s_t[0].argmax()] - p.phi * (traj[t-1].h_t[0] - p.mu[traj[t].s_t[0].argmax()]) for t in range(1, T_plus_1)]
        e_list = np.array(e_list)
        Q = np.sum(np.square(e_list)) + (traj[0].h_t[0] - p.mu[traj[0].s_t[0].argmax()])**2 * (1 - p.phi**2)
        temp = rng.gamma(shape=self.prior.eta2_a + T_plus_1 / 2, scale=1.0 / (self.prior.eta2_b + Q / 2))
        eta2 = 1.0 / temp

        # -----------------------
        # Conditional on mu1
        # -----------------------
        y_list = [traj[t].h_t[0] - p.phi * traj[t-1].h_t[0] for t in range(1, T_plus_1)]
        y_list = np.array(y_list)
        if traj[0].s_t[0].argmax() == 0:
            # Initial state is in regime 0
            V = 1 / (1 / self.prior.mu_sd**2 + (1 - p.phi)**2 * len(regime_sets[0]) / eta2 + (1-p.phi**2) / eta2)
            m = V * (self.prior.mu_mean / self.prior.mu_sd**2 + (1 - p.phi) * np.sum([y_list[t-1] for t in regime_sets[0]]) / eta2 + (1 - p.phi**2) * traj[0].h_t[0] / eta2)
        else:
            V = 1 / (1 / self.prior.mu_sd**2 + (1 - p.phi)**2 * len(regime_sets[0]) / eta2)
            m = V * (self.prior.mu_mean / self.prior.mu_sd**2 + (1 - p.phi) * np.sum([y_list[t-1] for t in regime_sets[0]]) / eta2)
        mu1 = rng.normal(m, np.sqrt(V))

        # -----------------------
        # Conditional on diff
        # ----------------------
        diff = []
        for k in range(1, K):
            # Sample from conditional posterior
            if traj[0].s_t[0].argmax() == k:
                V_k = 1 / (1 / self.prior.diff_sigma**2 + (1 - p.phi)**2 * len(regime_sets[k]) / eta2 + (1-p.phi**2) / eta2)
                m_k = V_k * (self.prior.diff_mean / self.prior.diff_sigma**2 + (1 - p.phi) * np.sum([y_list[t-1] - (1-p.phi)*p.mu[k-1] for t in regime_sets[k]]) / eta2 + (1 - p.phi**2) * (traj[0].h_t[0] - p.mu[k-1]) / eta2)
            else:
                V_k = 1 / (1 / self.prior.diff_sigma**2 + (1 - p.phi)**2 * len(regime_sets[k]) / eta2)
                m_k = V_k * (self.prior.diff_mean / self.prior.diff_sigma**2 + (1 - p.phi) * np.sum([y_list[t-1] - (1-p.phi)*p.mu[k-1] for t in regime_sets[k]]) / eta2)
            lo = (0 - m_k) / np.sqrt(V_k)
            hi = np.inf
            diff_k = truncnorm.rvs(lo, hi, loc=m_k, scale=np.sqrt(V_k), random_state=rng)
            diff.append(diff_k)

        # ----------------------
        # Conditional on phi        (closed form does not exist, we use MH step)
        # ----------------------
        y_list = [traj[t].h_t[0] - p.mu[traj[t].s_t[0].argmax()] for t in range(1, T_plus_1)]
        x_list = [traj[t-1].h_t[0] - p.mu[traj[t].s_t[0].argmax()] for t in range(1, T_plus_1)]
        A = np.sum(np.square(x_list)) - (traj[0].h_t[0] - p.mu[traj[0].s_t[0].argmax()])**2
        B = np.sum(np.multiply(y_list, x_list))
        mu = B / A
        sd = np.sqrt(p.eta2 / A)

        # truncated normal bounds
        lo = (-1 - mu) / sd
        hi = (1 - mu) / sd
        phi_current = p.phi

        for i in range(10):  # Run MH step for 10 iterations
            phi_star = truncnorm.rvs(lo, hi, loc=mu, scale=sd)
            log_alpha = (self.prior.phi_a - 0.5)*(np.log(1-phi_star) - np.log(1-phi_current))+(self.prior.phi_b - 0.5)* (np.log(1+phi_star) - np.log(1+phi_current))

            if np.log(rng.uniform()) < min(0, log_alpha):
                phi_current = phi_star

        mu = np.concatenate(([mu1], mu1 + np.cumsum(diff)))
        return MSSVParams.from_mu(mu, phi_current, eta2, P)

    def _logpdf_conditional(self, p: MSSVParams, traj: List[MSSVState]) -> float:
        cfg = self.params["conditional"]

        raise NotImplementedError("Conditional proposal logpdf is not implemented yet.")

    def _sample_independent(self, rng: np.random.Generator) -> MSSVParams:
        cfg = self.params["independent"]

        mean = cfg["mean"]
        cov = cfg["covariance"]

        z = rng.multivariate_normal(mean=mean, cov=cov)

        idx = 0
        D = len(z)
        K_float = np.sqrt(D - 2)
        if not np.isclose(K_float, round(K_float)):
            raise ValueError(
                f"Invalid parameter vector dimension D={D}; "
                f"expected D = K^2 + 2"
            )
        K = int(round(K_float))

        mu1 = z[idx]
        idx += 1

        delta = z[idx: idx + (K - 1)]
        idx += (K - 1)

        phi_unconstrained = z[idx]
        phi = np.tanh(phi_unconstrained)
        idx += 1

        log_eta2 = z[idx]
        eta2 = np.exp(log_eta2)
        idx += 1

        P = np.zeros((K, K))
        for i in range(K):
            logits = z[idx: idx + (K - 1)]
            idx += (K - 1)

            exp_logits = np.exp(logits)

            denom = 1.0 + np.sum(exp_logits)

            row = np.empty(K)
            row[:-1] = exp_logits / denom
            row[-1] = 1.0 / denom

            P[i] = row

        return MSSVParams(
            mu1=mu1,
            delta=np.asarray(delta),
            phi=phi,
            eta2=eta2,
            P=P,
        )

    def _logpdf_independent(self, p: MSSVParams) -> float:
        cfg = self.params["independent"]

        mean = cfg["mean"]
        cov = cfg["covariance"]

        K = p.K
        z_parts = []

        z_parts.append(np.array([p.mu1]))   # mu1

        z_parts.append(np.asarray(p.delta)) # delta

        # -------------------------------------------------
        # phi in (-1,1)
        # x = atanh(phi)
        #
        # Jacobian:
        #   dphi/dx = 1 - phi^2
        # Therefore:
        #   log |dphi/dx|
        # -------------------------------------------------
        phi = np.clip(p.phi, -1 + EPS, 1 - EPS)
        phi_unconstrained = np.arctanh(phi)
        z_parts.append(np.array([phi_unconstrained]))
        log_jacobian = np.log(1.0 - phi**2)

        # -------------------------------------------------
        # eta2 > 0
        # x = log(eta2)
        #
        # Jacobian:
        #   deta2/dx = eta2
        # -------------------------------------------------
        eta2 = max(p.eta2, EPS)
        log_eta2 = np.log(eta2)
        z_parts.append(np.array([log_eta2]))
        log_jacobian += np.log(eta2)

        # -------------------------------------------------
        # Transition matrix
        #
        # Row transform:
        #   a_j = log(p_j / p_K)
        #
        # Jacobian determinant for inverse softmax:
        #   |J| = prod_j p_j
        #
        # Therefore:
        #   log|J| = sum log(p_j)
        # -------------------------------------------------
        for i in range(K):
            row = np.clip(p.P[i], EPS, 1.0)
            row = row / row.sum()

            logits = np.log(row[:-1]) - np.log(row[-1])

            z_parts.append(logits)

            log_jacobian += np.sum(np.log(row))

        # -------------------------------------------------
        # Build unconstrained vector
        # -------------------------------------------------
        z = np.concatenate(z_parts)

        # -------------------------------------------------
        # Gaussian log-density
        # -------------------------------------------------
        log_q_z = multivariate_normal.logpdf(
            z,
            mean=mean,
            cov=cov,
            allow_singular=False,
        )

        # -------------------------------------------------
        # Density on constrained space
        # -------------------------------------------------
        return log_q_z - log_jacobian

    def sample(self, rng: np.random.Generator, from_p: MSSVParams = None, traj: List[MSSVState] = None) -> MSSVParams:
        if self.mode == "rw":
            if from_p is None:
                raise ValueError("from_p must be provided for random walk proposal")
            return self._sample_rw(rng, from_p)
        elif self.mode == "conditional":
            if traj is None:
                raise ValueError("traj must be provided for conditional proposal")
            return self._sample_conditional(rng, from_p, traj)
        elif self.mode == "independent":
            return self._sample_independent(rng)
        else:
            raise ValueError(f"Unknown proposal mode: {self.mode}")
        
    def logpdf(self, to_p: MSSVParams, from_p: MSSVParams = None, traj: List[MSSVState] = None) -> float:
        if self.mode == "rw":
            if from_p is None:
                raise ValueError("from_p must be provided for random walk proposal")
            return self._logpdf_rw(from_p, to_p)
        elif self.mode == "conditional":
            if traj is None:
                raise ValueError("traj must be provided for conditional proposal")
            return self._logpdf_conditional(to_p, traj)
        elif self.mode == "independent":
            return self._logpdf_independent(to_p)
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
        var = theta.eta2 / (1 - theta.phi ** 2)  # Stationary variance of AR(1) process
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

        h_t = mu + theta.phi * (h_prev - mu) + self.rng.normal(size=N, scale=np.sqrt(theta.eta2))
    
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
        p_h = norm.pdf(h_next, loc=mean_h, scale=np.sqrt(theta.eta2))   # (N,)

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
        log_p_h = norm.logpdf(h_next, loc=mean_h, scale=np.sqrt(theta.eta2))

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
        var = theta.eta2 / (1 - theta.phi ** 2)
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
        var = theta.eta2 / (1 - theta.phi ** 2)
        mu = theta.mu[idx]

        log_p_h = norm.logpdf(h_0, loc=mu, scale=np.sqrt(var))

        return log_p_s + log_p_h