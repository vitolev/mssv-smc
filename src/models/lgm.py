import numpy as np
from src.models.base import StateSpaceModel
from scipy.stats import norm

class LGModelParams:
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

    def sample_initial(self, theta):
        return self.rng.normal(0, 1)

    def sample_next(self, theta: LGModelParams, x_prev):
        return self.rng.normal(theta.a * x_prev, theta.sigma_x)
    
    def approx_expected_next(self, theta: LGModelParams, x_prev):
        ## For LGModel, the expected next state is simply a * x_prev, so there is no approximation needed.
        return theta.a * x_prev
    
    def likelihood(self, y, theta: LGModelParams, x):
        return norm.pdf(y, loc=theta.b * x, scale=theta.sigma_y)
    
    def log_likelihood(self, y, theta: LGModelParams, x):
        return norm.logpdf(y, loc=theta.b * x, scale=theta.sigma_y)