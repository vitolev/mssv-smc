import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState, StateSpaceModelPrior, StateSpaceModelProposal
from scipy.stats import norm, uniform, expon, beta, multivariate_normal
from scipy.special import logsumexp
from dataclasses import dataclass

EPS = 1e-10

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

    def __repr__(self):
        return (f"LGModelParams(a={self.a}, b={self.b}, sigma_x={self.sigma_x}, sigma_y={self.sigma_y})")

    def to_unconstrained(self) -> np.ndarray:
        """
        Transform parameters to an unconstrained space for optimization or sampling.

        Transformations:
        - a: arctanh transform
        - b: already unconstrained
        - sigma_x: log transform
        - sigma_y: log transform

        Returns
        -------
            z: np.ndarray
                Unconstrained parameter vector
        """
        z_a = np.arctanh(self.a)  # maps [-1, 1] to (-inf, inf)
        z_b = self.b              # already unconstrained
        z_sigma_x = np.log(self.sigma_x)  # maps (0, inf) to (-inf, inf)
        z_sigma_y = np.log(self.sigma_y)  # maps (0, inf) to (-inf, inf)
        return np.array([z_a, z_b, z_sigma_x, z_sigma_y])

    def from_unconstrained(z: np.ndarray) -> "LGModelParams":
        """
        Transform unconstrained parameters back to the constrained space.

        Inverse transformations:
        - a: tanh transform
        - b: already unconstrained
        - sigma_x: exp transform
        - sigma_y: exp transform

        Parameters
        ----------
            z: np.ndarray
                Unconstrained parameter vector

        Returns
        -------
            params: LGModelParams
                Constrained parameter instance
        """
        a = np.tanh(z[0])  # maps (-inf, inf) to (-1, 1)
        b = z[1]           # already unconstrained
        sigma_x = np.exp(z[2])  # maps (-inf, inf) to (0, inf)
        sigma_y = np.exp(z[3])  # maps (-inf, inf) to (0, inf)
        return LGModelParams(a=a, b=b, sigma_x=sigma_x, sigma_y=sigma_y)

    def to_vector(self) -> np.ndarray:
        """
        Convert parameters to a vector representation. Used for storing samples in a consistent format. 
        """
        return np.array([self.a, self.b, self.sigma_x, self.sigma_y])
        
    def copy(self) -> "LGModelParams":
        return LGModelParams(a=self.a, b=self.b, sigma_x=self.sigma_x, sigma_y=self.sigma_y)

# =========================
# PRIOR
# =========================
class LGModelPrior(StateSpaceModelPrior):
    """
    Prior distribution for the LGSSM parameters. The prior is defined as follows:
    - a ~ Beta(a_a, a_b) transformed to [-1, 1]
    - b ~ Normal(b_mean, b_sd)
    - sigma_x ~ Exponential(sigma_x_scale)
    - sigma_y ~ Exponential(sigma_y_scale)
    """
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
    """
    Proposal distribution for the LGSSM parameters. 

    The proposal can operate in following modes:
        - "rw": Random walk proposal. Parameters:
            - "covariance": Covariance matrix for the multivariate normal proposal in the unconstrained space.
        - "independent": Independent proposal. Parameters:
            - "mean": Mean vector for the multivariate normal proposal in the unconstrained space.
            - "covariance": Covariance matrix for the multivariate normal proposal in the unconstrained space.
    """
    def __init__(self, params: dict):
        self.mode = params["mode"]  
        if self.mode not in ["rw", "independent"]:
            raise ValueError(f"Unknown proposal mode: {self.mode}")

        default_params = {
            "rw": {
                "covariance": None
            },
            "independent": {
                "mean": None,
                "covariance": None
            }
        }

        for k in params:
            if k in default_params[self.mode]:
                default_params[self.mode][k] = params[k]

        self.params = default_params

    def update_params(self, new_params: dict):
        for k in new_params:
            if k in self.params[self.mode]:
                self.params[self.mode][k] = new_params[k]

    def _sample_rw(self, rng: np.random.Generator, p: LGModelParams) -> LGModelParams:
        cfg = self.params["rw"]

        cov = cfg["covariance"]
        mean = np.zeros(cov.shape[0])

        z = rng.multivariate_normal(mean=mean, cov=cov)

        p_unconstrained = p.to_unconstrained()
        p_unconstrained_new = p_unconstrained + z
        p_new = LGModelParams.from_unconstrained(p_unconstrained_new)
        return p_new

    def _logpdf_rw(self, p_from: LGModelParams, p_to: LGModelParams) -> float:
        cfg = self.params["rw"]

        cov = cfg["covariance"]
        mean = np.zeros(cov.shape[0])

        p_from_unconstrained = p_from.to_unconstrained()
        p_to_unconstrained = p_to.to_unconstrained()

        diff = p_to_unconstrained - p_from_unconstrained
        log_prob = multivariate_normal.logpdf(diff, mean=mean, cov=cov)

        log_jacobian = 0.0

        a = np.clip(p_to.a, -1 + EPS, 1 - EPS)
        log_jacobian += np.log(1.0 - a**2)

        sigma_x = max(p_to.sigma_x, EPS)
        sigma_y = max(p_to.sigma_y, EPS)

        log_jacobian += np.log(sigma_x)
        log_jacobian += np.log(sigma_y)

        return log_prob - log_jacobian

    def _sample_independent(self, rng: np.random.Generator) -> LGModelParams:
        cfg = self.params["independent"]

        mean = cfg["mean"]
        cov = cfg["covariance"]

        z = rng.multivariate_normal(mean=mean, cov=cov)
        p_new = LGModelParams.from_unconstrained(z)
        return p_new

    def _logpdf_independent(self, p: LGModelParams) -> float:
        cfg = self.params["independent"]

        mean = cfg["mean"]
        cov = cfg["covariance"]

        z = p.to_unconstrained()
        log_prob = multivariate_normal.logpdf(z, mean=mean, cov=cov)

        log_jacobian = 0.0

        a = np.clip(p.a, -1 + EPS, 1 - EPS)
        log_jacobian += np.log(1.0 - a**2)

        sigma_x = max(p.sigma_x, EPS)
        sigma_y = max(p.sigma_y, EPS)

        log_jacobian += np.log(sigma_x)
        log_jacobian += np.log(sigma_y)

        return log_prob - log_jacobian

    def sample(self, rng: np.random.Generator, from_p: LGModelParams = None) -> LGModelParams:
        if self.mode == "rw":
            return self._sample_rw(rng, from_p)
        else: # "independent"
            return self._sample_independent(rng)
        
    def logpdf(self, to_p: LGModelParams, from_p: LGModelParams = None) -> float:
        if self.mode == "rw":
            return self._logpdf_rw(from_p, to_p)
        else: # "independent"
            return self._logpdf_independent(to_p)

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
        Extend the current state by adding another state.

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

    def to_numpy(self) -> np.ndarray:
        """
        Convert the state to a numpy array of shape (N, 1).
        """
        return self.x_t.reshape(-1, 1)

    def copy(self) -> "LGModelState":
        """
        Create a copy of the state.
        """
        return LGModelState(x_t=self.x_t.copy())

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