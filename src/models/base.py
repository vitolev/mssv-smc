from abc import ABC, abstractmethod
import numpy as np

class StateSpaceModelParams(ABC):
    """
    Base class for state space model parameters.
    """
    def __init__(self):
        pass

class StateSpaceModel(ABC):
    """
    Generic state-space model:
        x_t ~ f(x_t | x_{t-1}, theta)
        y_t ~ g(y_t | x_t, theta)
    """
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def sample_observation(self, theta: StateSpaceModelParams, state):
        """
        Sample an observation y_t given state x_t and parameters theta.
        """
        pass

    @abstractmethod
    def sample_initial_state(self, theta: StateSpaceModelParams):
        """
        Sample the initial state x_0 given parameters theta.
        """
        pass

    @abstractmethod
    def sample_next_state(self, theta: StateSpaceModelParams, state):
        """
        Sample the next state x_t given previous state x_{t-1} and parameters theta.
        """
        pass

    @abstractmethod
    def expected_next_state(self, theta: StateSpaceModelParams, state):
        """
        Compute the expected next state given current state and parameters theta.
        """
        pass

    @abstractmethod
    def likelihood(self, y, theta: StateSpaceModelParams, states):
        """
        Compute the likelihood of observations y given states.
        """
        pass

    @abstractmethod
    def log_likelihood(self, y, theta: StateSpaceModelParams, states):
        """
        Compute the log-likelihood of observations y given states.
        """
        pass

    @abstractmethod
    def state_transition(self, theta: StateSpaceModelParams, state_prev, state_next):
        """
        Compute the state transition probability p(x_t | x_{t-1}, theta).
        """
        pass

    @abstractmethod
    def log_state_transition(self, theta: StateSpaceModelParams, state_prev, state_next):
        """
        Compute the log of the state transition probability log p(x_t | x_{t-1}, theta).
        """
        pass
