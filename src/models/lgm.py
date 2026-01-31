import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams, StateSpaceModelState
from scipy.stats import norm

class LGModelParams(StateSpaceModelParams):
    """
    Container for LGSSM model parameters.
    """
    def __init__(self, a: float, b: float, sigma_x: float, sigma_y: float):
        self.a = a
        self.b = b
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        if sigma_x <= 0 or sigma_y <= 0:
            raise ValueError("Standard deviations sigma_x and sigma_y must be positive.")

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
    def __init__(self, rng=None):
        super().__init__(rng)

    def sample_observation(self, theta: LGModelParams, state: LGModelState) -> np.ndarray:
        """
        Sample observation y_t | x_t
        """
        return self.rng.normal(theta.b * state.x_t, theta.sigma_y)

    def sample_initial_state(self, theta: LGModelParams, size: int = 1) -> LGModelState:
        """
        Sample initial latent states x_0 ~ N(0,1)
        """
        x0 = self.rng.normal(0.0, 1.0, size=size)
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

    def state_transition(self, theta: LGModelParams, state_prev: LGModelState, state_next: LGModelState) -> np.ndarray:
        """
        Transition probability p(x_t | x_{t-1})
        """
        return norm.pdf(state_next.x_t, loc=theta.a * state_prev.x_t, scale=theta.sigma_x)

    def log_state_transition(self, theta: LGModelParams, state_prev: LGModelState, state_next: LGModelState) -> np.ndarray:
        """
        Log transition probability log p(x_t | x_{t-1})
        """
        return norm.logpdf(state_next.x_t, loc=theta.a * state_prev.x_t, scale=theta.sigma_x)