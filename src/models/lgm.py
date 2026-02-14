import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from scipy.stats import norm, uniform, expon

class LGModelParams(StateSpaceModelParams):
    """
    Container for LGSSM model parameters.
    """
    def __init__(self, rng: np.random.Generator = None,
                 a: float = None, 
                 b: float = None, 
                 sigma_x: float = None, 
                 sigma_y: float = None):
        
        # Sample from prior if any parameter is None
        if any(param is None for param in [a, b, sigma_x, sigma_y]):
            if rng is None:
                raise ValueError("RNG must be provided to sample parameters from the prior.")
            self.sample_prior(rng)

        # Set provided parameters
        else:
            self.a = a
            self.b = b
            self.sigma_x = sigma_x
            self.sigma_y = sigma_y

            if sigma_x <= 0 or sigma_y <= 0:
                raise ValueError("Standard deviations sigma_x and sigma_y must be positive.")
            
    def sample_prior(self, rng: np.random.Generator):
        """
        Sample parameters from the prior distribution. For simplicity, we use independent priors:
            a ~ U(-1, 1)  # Uniform prior for a to encourage stability of the state process
            b ~ N(0, 1)
            sigma_x ~ Exponential(1) 
            sigma_y ~ Exponential(1)

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator for sampling.
        """
        self.a = rng.uniform(-1.0, 1.0)  # Uniform prior for a, to ensure stationarity of the state process     
        self.b = rng.normal(0.0, 1.0)       
        self.sigma_x = rng.exponential(1.0)
        self.sigma_y = rng.exponential(1.0)
        
    def log_prior_density(self) -> float:
        """
        Compute the log prior density of the parameters.

        Returns
        -------
        log_prior: float
            The log prior density evaluated at the current parameter values.
        """
        if self.sigma_x <= 0 or self.sigma_y <= 0:
            return -np.inf  # Log prior density is -inf if standard deviations are not positive
        if self.a < -1.0 or self.a > 1.0:
            return -np.inf  # Log prior density is -inf if a is outside the uniform support
        
        log_prior_a = uniform.logpdf(self.a, loc=-1.0, scale=2.0)
        log_prior_b = norm.logpdf(self.b, loc=0.0, scale=1.0)

        log_prior_sigma_x = expon.logpdf(self.sigma_x, scale=1.0)
        log_prior_sigma_y = expon.logpdf(self.sigma_y, scale=1.0)

        return log_prior_a + log_prior_b + log_prior_sigma_x + log_prior_sigma_y

    def sample_transition(
        self,
        rng: np.random.Generator,
        step_a: float = 0.1,
        step_b: float = 0.1,
        step_sigma_x: float = 0.1,
        step_sigma_y: float = 0.1,
    ) -> "LGModelParams":
        """
        Given the current parameters, sample new parameters from a proposal distribution for PMMH. For simplicity, we use independent Gaussian random walk proposals for a and b, and log-normal random walk proposals for sigma_x and sigma_y to ensure positivity.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator for sampling.
        step_a, step_b, step_sigma_x, step_sigma_y : float
            Step sizes for the random walk proposals for a, b, sigma_x, and sigma_y respectively.

        Returns
        -------
        new_params: LGModelParams
            A new instance of LGModelParams sampled from the proposal distribution.
        """
        alpha = np.arctanh(self.a)  # Transform a to the real line for unconstrained proposal
        alpha_new = alpha + rng.normal(0.0, step_a)  # Propose new alpha
        a_new = np.tanh(alpha_new)  # Transform back to (-1, 1)

        b_new = self.b + rng.normal(0.0, step_b)
        sigma_x_new = self.sigma_x * np.exp(rng.normal(0.0, step_sigma_x))  # Log-normal proposal
        sigma_y_new = self.sigma_y * np.exp(rng.normal(0.0, step_sigma_y))  # Log-normal proposal
        return LGModelParams(a=a_new, b=b_new, sigma_x=sigma_x_new, sigma_y=sigma_y_new)

    def log_transition_density(
        self, 
        other: "LGModelParams",
        step_a: float = 0.1,
        step_b: float = 0.1,
        step_sigma_x: float = 0.1,
        step_sigma_y: float = 0.1
    ) -> float:
        """
        Compute the log q(other | self) where q is the proposal distribution.

        Parameters
        ----------
        other : LGModelParams
            The other parameter instance to evaluate the transition density against.
        step_a, step_b, step_sigma_x, step_sigma_y : float
            Step sizes for the random walk proposals for a, b, sigma_x, and sigma_y respectively.

        Returns
        -------
        log_density: float
            The log density of proposing 'other' given 'self' under the proposal distribution.
        """
        # ---- a (Gaussian RW in alpha-space) ----
        alpha_self = np.arctanh(self.a)
        alpha_other = np.arctanh(other.a)
        
        # Proposal density in alpha-space
        log_q_alpha = norm.logpdf(alpha_other, loc=alpha_self, scale=step_a)
        
        # Jacobian of the transformation a = tanh(alpha)
        log_jacobian = np.log(1 - other.a ** 2)  # da/dalpha = 1 - tanh^2(alpha)
        
        log_q_a = log_q_alpha + log_jacobian

        # ---- b (regular Gaussian RW) ----
        log_q_b = norm.logpdf(other.b, loc=self.b, scale=step_b)

        # ---- sigma_x, sigma_y (log-normal RW) ----
        log_q_sigma_x = norm.logpdf(np.log(other.sigma_x / self.sigma_x), loc=0.0, scale=step_sigma_x) - np.log(other.sigma_x)
        log_q_sigma_y = norm.logpdf(np.log(other.sigma_y / self.sigma_y), loc=0.0, scale=step_sigma_y) - np.log(other.sigma_y)

        return log_q_a + log_q_b + log_q_sigma_x + log_q_sigma_y

    def __repr__(self):
        return f"LGModelParams(a={self.a:.3f}, b={self.b:.3f}, sigma_x={self.sigma_x:.3f}, sigma_y={self.sigma_y:.3f})"
    
class LGModelState(StateSpaceModelState):
    """
    Container for LGSSM model state.
    """
    def __init__(self, x_t: np.ndarray):
        self.x_t = x_t  # shape (N,)

    def __getitem__(self, idx):
        return LGModelState(x_t=np.array(self.x_t[idx], copy=True))
    
    def __len__(self):
        return self.x_t.shape[0]

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