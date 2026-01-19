import numpy as np
from scipy.stats import norm
from src.models.base import StateSpaceModel
from typing import List

class MSSVModelParams:
    """
    Container for MSSV model parameters.
    """
    def __init__(self, mu: List[float], phi: List[float], sigma_eta: List[float], P: List[List[float]]):
        self.mu = np.array(mu)
        self.phi = np.array(phi)
        self.sigma_eta = np.array(sigma_eta)
        self.P = np.array(P)

        if not (len(mu) == len(phi) == len(sigma_eta) == len(P)):
            raise ValueError("Parameters mu, phi, sigma_eta, and P must have the same length.")
        
        if any(sigma <= 0 for sigma in sigma_eta):
            raise ValueError("Standard deviations sigma_eta must be positive.")
        
        if any(p < 0 for row in P for p in row):
            raise ValueError("Transition probabilities must be non-negative.")
        
        if any(abs(sum(row) - 1.0) > 1e-8 for row in P):
            raise ValueError("Each row of transition matrix P must sum to 1.")

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
        h_t | h_{t-1}, s_t ~ N(mu_{s_t} + phi_{s_t} * (h_{t-1} - mu_{s_t}), sigma_eta_{s_t}^2)
        y_t | h_t ~ N(0, exp(h_t))
    """
    def __init__(self, rng=None):
        super().__init__(rng)

    def sample_initial_state(self, theta : MSSVModelParams):
        """
        Sample the initial state (h0, s0) given initial parameters theta.
        """
        K = len(theta.mu)
        # Create one-hot encoding for regime
        regime = self.rng.choice(K)
        s0 = np.zeros(K)
        s0[regime] = 1
        # Sample initial log-volatility
        h0 = self.rng.normal(theta.mu[regime], theta.sigma_eta[regime])
        return (h0, s0)

    def sample_next_state(self, theta : MSSVModelParams, state: tuple):
        """
        Sample the next state (h_t, s_t) given previous state and new parameters theta.
        """
        h_prev, s_prev = state
        K = len(theta.mu)

        # Regime transition
        s_t = theta.P @ s_prev
        index = self.rng.choice(K, p=s_t)   # Sample new regime index based on probabilities
        s_t = np.zeros(K)
        s_t[index] = 1                      # One-hot encode the new regime

        # Volatility transition
        h_t = (
            theta.mu[index]
            + theta.phi[index] * (h_prev - theta.mu[index])
            + theta.sigma_eta[index] * self.rng.normal(0, 1)
        )

        return (h_t, s_t)
    
    def expected_next_state(self, theta : MSSVModelParams, state: tuple):
        """
        Compute the expected next state given current state and parameters theta.
        """
        h_prev, s_prev = state

        # Expected regime distribution
        s_exp = theta.P @ s_prev
        # Expected log-volatility
        h_exp = 0.0
        for i in range(len(s_exp)):
            h_exp += s_exp[i] * (
                theta.mu[i]
                + theta.phi[i] * (h_prev - theta.mu[i])
            )

        return (h_exp, s_exp)

    
    def likelihood(self, y_t, theta : MSSVModelParams, state: tuple):
        """
        Compute the likelihood of observation y_t given current state.

        p(y_t | x_t, theta) ~ N(0, exp(h_t))
        """
        h_t, _ = state
        return norm.pdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))  # scale parameter is standard deviation hence 0.5

    def log_likelihood(self, y_t, theta : MSSVModelParams, state: tuple):
        """
        Compute the log-likelihood of observation y_t given current state.
        """
        h_t, _ = state
        return norm.logpdf(y_t, loc=0.0, scale=np.exp(0.5 * h_t))
    


