import numpy as np
from scipy.stats import norm
from src.models.base import StateSpaceModel
from typing import List

class MSSVModelParams:
    """
    Container for MSSV model parameters.
    """
    def __init__(self, mu: List[float], phi: List[float], sigma_eta: List[float], P: List[List[float]]):
        self.mu = mu
        self.phi = phi
        self.sigma_eta = sigma_eta
        self.P = P

        if not (len(mu) == len(phi) == len(sigma_eta) == len(P)):
            raise ValueError("Parameters mu, phi, sigma_eta, and P must have the same length.")

class MSSVModel(StateSpaceModel):
    """
    Markov-Switching Stochastic Volatility Model

    A model used for capturing regime-switching behavior in financial time series, by modeling
    the log-volatility as a latent variable that switches between different regimes.

    Model definition:
        s_0 ~ Uniform{1, ..., K}
        h_0 | s_0 ~ N(mu_{s_0}, sigma_eta_{s_0}^2)
        s_t | s_{t-1} ~ Categorical(P_{s_{t-1}, :))
        h_t | h_{t-1}, s_t ~ N(mu_{s_t} + phi_{s_t} * (h_{t-1} - mu_{s_t}), sigma_eta_{s_t}^2)
        y_t | h_t ~ N(0, exp(h_t))
    """
    def __init__(self, rng=None):

        super().__init__(rng)

    def sample_initial(self, theta : MSSVModelParams):
        """
        Sample the initial state (h0, s0) given initial parameters theta.
        """
        s0 = self.rng.choice(len(theta.mu))
        h0 = self.rng.normal(theta.mu[s0], theta.sigma_eta[s0])
        return (h0, s0)

    def sample_next(self, theta : MSSVModelParams, state: tuple):
        """
        Sample the next state (h_t, s_t) given previous state and new parameters theta.
        """
        # Regime transition
        s_t = self.rng.choice(len(theta.mu), p=theta.P[state[1]])

        # Volatility transition
        h_t = (
            theta.mu[s_t]
            + theta.phi[s_t] * (state[0] - theta.mu[s_t])
            + theta.sigma_eta[s_t] * self.rng.normal(0, 1)
        )

        return (h_t, s_t)
    
    def approx_expected_next(self, theta : MSSVModelParams, state: tuple):
        """
        Compute the approximation of expected next state given current state and parameters theta.
        """
        h_t, s_t = state
        h_new = 0.0
        for i in range(len(theta.mu)):
            p_i = theta.P[s_t][i]   # Transition probability to regime i
            h_new += p_i * (
                theta.mu[i]
                + theta.phi[i] * (h_t - theta.mu[i])
            )
        
        s_new = np.argmax(theta.P[s_t])
        return (h_new, s_new)

    
    def likelihood(self, y_t, theta : MSSVModelParams, state: tuple):
        """
        Compute the likelihood of observation y_t given current state.

        p(y_t | x_t, theta) ~ N(0, exp(h_t))
        """
        h_t, _ = state
        return norm.pdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))

    def log_likelihood(self, y_t, theta : MSSVModelParams, state: tuple):
        """
        Compute the log-likelihood of observation y_t given current state.
        """
        h_t, _ = state
        return norm.logpdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))
    


