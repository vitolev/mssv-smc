from abc import ABC, abstractmethod
import numpy as np

class StateSpaceModel(ABC):
    """
    Generic state-space model:
        x_t ~ f(x_t | x_{t-1}, theta)
        y_t ~ g(y_t | x_t, theta)
    """
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def sample_initial_state(self, theta):
        """
        Sample the initial state x_0 given parameters theta.
        """
        pass

    @abstractmethod
    def sample_transition(self, theta, state):
        """
        Sample the next state x_t given previous state x_{t-1} and parameters theta.
        """
        pass

    @abstractmethod
    def likelihood(self, y_t, theta, state):
        """
        Compute the likelihood of observation y_t given current state.
        """
        pass

    @abstractmethod
    def log_likelihood(self, y_t, theta, state):
        """
        Compute the log-likelihood of observation y_t given current state.
        """
        pass
