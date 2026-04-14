import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState, StateSpaceModelPrior, StateSpaceModelProposal
from scipy.stats import norm, uniform, expon, beta
from scipy.special import logsumexp
from dataclasses import dataclass

# =========================
# PARAMETER CONTAINER
# =========================
@dataclass(frozen=True)
class LGModelParams(StateSpaceModelParams):
    a: float
    b: float
    sigma_x: float
    sigma_y: float

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.sigma_x <= 0:
            raise ValueError(f"sigma_x must be positive, got {self.sigma_x}")
        if self.sigma_y <= 0:
            raise ValueError(f"sigma_y must be positive, got {self.sigma_y}")
        if self.a < -1.0 or self.a > 1.0:
            raise ValueError(f"a must be in the range [-1, 1] for stationarity, got {self.a}")
        
    def copy(self):
        return LGModelParams(a=self.a, b=self.b, sigma_x=self.sigma_x, sigma_y=self.sigma_y)

# =========================
# PRIOR
# =========================
class LGModelPrior(StateSpaceModelPrior):
    def __init__(self,
                 a_a=1,
                 a_b=1,
                 b_mean=0,
                 b_sd=1,
                 sigma_x_scale=1,
                 sigma_y_scale=1):
        self.a_a = a_a
        self.a_b = a_b
        self.b_mean = b_mean
        self.b_sd = b_sd
        self.sigma_x_scale = sigma_x_scale
        self.sigma_y_scale = sigma_y_scale

    def sample(self, rng: np.random.Generator) -> LGModelParams:
        a = rng.beta(self.a_a, self.a_b) * 2 - 1        # Beta prior for a transformed to [-1, 1]
        b = rng.normal(self.b_mean, self.b_sd)
        sigma_x = rng.exponential(self.sigma_x_scale)
        sigma_y = rng.exponential(self.sigma_y_scale)
        return LGModelParams(a=a, b=b, sigma_x=sigma_x, sigma_y=sigma_y)

    def logpdf(self, params: LGModelParams) -> float:
        logp = 0.0

        u = (params.a + 1) / 2  # Transform a from [-1, 1] to [0, 1]
        logp += beta.logpdf(u, self.a_a, self.a_b) - np.log(2)  # Adjust for transformation

        logp += norm.logpdf(params.b, loc=self.b_mean, scale=self.b_sd)
        logp += expon.logpdf(params.sigma_x, scale=self.sigma_x_scale)
        logp += expon.logpdf(params.sigma_y, scale=self.sigma_y_scale)

        return logp

# =========================
# PROPOSAL
# =========================
class LGModelProposal(StateSpaceModelProposal):
    def __init__(self, step_a=0.1, step_b=0.1, step_sigma_x=0.1, step_sigma_y=0.1):
        self.step_a = step_a
        self.step_b = step_b
        self.step_sigma_x = step_sigma_x
        self.step_sigma_y = step_sigma_y

    def sample(self, rng: np.random.Generator, p: LGModelParams) -> LGModelParams:
        L = -1.0
        U = 1.0
        width = U - L  # 2.0
        a_prop = p.a + rng.normal(0.0, self.step_a)
        # Infinite reflection via modulo folding
        a_prop = (a_prop - L) % (2.0 * width)
        if a_prop > width:
            a_prop = 2.0 * width - a_prop
        a_new = a_prop + L

        b_new = p.b + rng.normal(0.0, self.step_b)
        sigma_x_new = p.sigma_x * np.exp(rng.normal(0.0, self.step_sigma_x))  # Log-normal proposal
        sigma_y_new = p.sigma_y * np.exp(rng.normal(0.0, self.step_sigma_y))  # Log-normal proposal
        return LGModelParams(a=a_new, b=b_new, sigma_x=sigma_x_new, sigma_y=sigma_y_new)

    def logpdf(self, p_from: LGModelParams, p_to: LGModelParams) -> float:
        # ---- a : Reflected Gaussian RW (finite image sum) ----
        period = 4.0  # 2 * width where width = 2 for [-1,1]
        K = 2  # number of image terms on each side

        diffs = []
        for k in range(-K, K + 1):
            shift = period * k
            diffs.append(
                norm.logpdf(
                    p_to.a,
                    loc=p_from.a - shift,
                    scale=self.step_a,
                )
            )

        log_q_a = logsumexp(diffs)

        # ---- b (regular Gaussian RW) ----
        log_q_b = norm.logpdf(p_to.b, loc=p_from.b, scale=self.step_b)

        # ---- sigma_x, sigma_y (log-normal RW) ----
        log_q_sigma_x = norm.logpdf(np.log(p_to.sigma_x / p_from.sigma_x), loc=0.0, scale=self.step_sigma_x) - np.log(p_to.sigma_x)
        log_q_sigma_y = norm.logpdf(np.log(p_to.sigma_y / p_from.sigma_y), loc=0.0, scale=self.step_sigma_y) - np.log(p_to.sigma_y)

        return log_q_a + log_q_b + log_q_sigma_x + log_q_sigma_y

# =========================
# MODEL
# =========================
class LGModelState(StateSpaceModelState):
    """
    Container for LGSSM model state.
    """
    def __init__(self, x_t: np.ndarray):
        x_t = np.asarray(x_t)

        # Convert scalar → (1,)
        if x_t.ndim == 0:
            x_t = x_t.reshape(1)

        # Convert column or row vectors → (N,)
        elif x_t.ndim == 2 and 1 in x_t.shape:
            x_t = x_t.reshape(-1)

        # Reject anything else
        elif x_t.ndim != 1:
            raise ValueError(
                f"x_t must be 1D with shape (N,), got shape {x_t.shape}"
            )

        self.x_t = x_t.copy()

    def __getitem__(self, idx):
        return LGModelState(x_t=np.array(self.x_t[idx], copy=True))
    
    def __setitem__(self, idx, value):
        if isinstance(value, LGModelState):
            if len(value.x_t) != 1:
                raise ValueError(f"Value must be a LGModelState with a single state, got shape {value.x_t.shape}")
            self.x_t[idx] = value.x_t[0]
        elif isinstance(value, (float, int)):
            self.x_t[idx] = value
        else:
            raise ValueError(f"Value must be a LGModelState with a single state or a scalar, got type {type(value)}")
    
    def __len__(self):
        return self.x_t.shape[0]
    
    def __repr__(self):
        return f"LGModelState(x_t={self.x_t})"
    
    def add(self, other: "LGModelState") -> "LGModelState":
        """
        Extend the current state by adding another state. This is useful for accumulating trajectories in PMMH.

        Parameters
        ----------
        other : LGModelState
            Another state to concatenate to the current state.

        Returns
        -------
        new_state: LGModelState
            A new LGModelState with extended state vector.
        """
        if not isinstance(other, LGModelState):
            raise ValueError(f"Other must be an instance of LGModelState, got type {type(other)}")
        
        new_x_t = np.hstack((self.x_t, other.x_t))
        return LGModelState(x_t=new_x_t)

class LGModel(StateSpaceModel):
    """
    Linear Gaussian State Space Model

    A model used for testing the particle filters, becasue of its simplicity and the fact that
    the Kalman filter can be used to compute exact solutions for comparison.

    Model definition:
        x_0 ~ N(0, 1)
        x_t | x_{t-1} ~ N(a * x_{t-1}, sigma_x^2)
        y_t | x_t ~ N(b * x_t, sigma_y^2)
    """
    params_type = LGModelParams
    state_type = LGModelState
    prior_type = LGModelPrior
    proposal_type = LGModelProposal

    def __init__(self, rng=None):
        super().__init__(rng)

    def sample_observation(self, theta: LGModelParams, state: LGModelState) -> np.ndarray:
        """
        Sample observation y_t | x_t
        """
        return self.rng.normal(theta.b * state.x_t, theta.sigma_y)

    def sample_initial_state(self, theta: LGModelParams, size: int = 1) -> LGModelState:
        """
        Sample initial latent states x_0 ~ N(0,10)
        """
        x0 = self.rng.normal(0.0, 10.0, size=size)
        return LGModelState(x0)

    def sample_next_state(self, theta: LGModelParams, state: LGModelState) -> LGModelState:
        """
        Sample x_t | x_{t-1}
        """
        x_prev = state.x_t
        x_next = self.rng.normal(theta.a * x_prev, theta.sigma_x)
        return LGModelState(x_next)

    def expected_next_state(self, theta: LGModelParams, state: LGModelState) -> LGModelState:
        """
        E[x_t | x_{t-1}] = a * x_{t-1}
        """
        x_exp = theta.a * state.x_t
        return LGModelState(x_exp)

    def likelihood(self, y: float, theta: LGModelParams, state: LGModelState) -> np.ndarray:
        """
        Likelihood p(y_t | x_t)
        """
        return norm.pdf(y, loc=theta.b * state.x_t, scale=theta.sigma_y)

    def log_likelihood(self, y: float, theta: LGModelParams, state: LGModelState) -> np.ndarray:
        """
        Log-likelihood log p(y_t | x_t)
        """
        return norm.logpdf(y, loc=theta.b * state.x_t, scale=theta.sigma_y)

    def transition_density(self, theta: LGModelParams, state_prev: LGModelState, state_next: LGModelState) -> np.ndarray:
        """
        Transition probability p(x_t | x_{t-1})
        """
        return norm.pdf(state_next.x_t, loc=theta.a * state_prev.x_t, scale=theta.sigma_x)

    def log_transition_density(self, theta: LGModelParams, state_prev: LGModelState, state_next: LGModelState) -> np.ndarray:
        """
        Log transition probability log p(x_t | x_{t-1})
        """
        return norm.logpdf(state_next.x_t, loc=theta.a * state_prev.x_t, scale=theta.sigma_x)

    def initial_state_density(self, theta: LGModelParams, state: LGModelState) -> np.ndarray:
        """
        Initial state density p(x_0)
        """
        return norm.pdf(state.x_t, loc=0.0, scale=10.0)

    def log_initial_state_density(self, theta: LGModelParams, state: LGModelState) -> np.ndarray:
        """
        Log initial state density log p(x_0)
        """
        return norm.logpdf(state.x_t, loc=0.0, scale=10.0)