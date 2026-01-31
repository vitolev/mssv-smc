from abc import ABC, abstractmethod
import numpy as np

class StateSpaceModelParams(ABC):
    """
    Base class for state space model parameters.
    """
    def __init__(self):
        pass

class StateSpaceModelState(ABC):
    """
    Base class for state space model state.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
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
    def sample_observation(self, theta: StateSpaceModelParams, state: StateSpaceModelState) -> np.ndarray:
        """
        Sample an observation y_t given state x_t and parameters theta.

        y_t ~ p(y_t | x_t, theta)

        Parameters
        ----------
            theta: StateSpaceModelParams
                Model parameters.
            state: StateSpaceModelState
                Current state of size N.
        Returns
        -------
            y_t: np.ndarray
                Sampled observation with shape (N,).
        """
        pass

    @abstractmethod
    def sample_initial_state(self, theta: StateSpaceModelParams, size: int = 1) -> StateSpaceModelState:
        """
        Sample the initial state x_0 given parameters theta.

        x_0 ~ p(x_0 | theta)

        Parameters
        ----------
            theta: StateSpaceModelParams
                Model parameters.
            size: int
                Number of initial states to sample. This influences the shape of the returned arrays in StateSpaceModelState. (default is 1)
        Returns
        -------
            state: StateSpaceModelState
                Sampled initial state with shape depending on size.
        """
        pass

    @abstractmethod
    def sample_next_state(self, theta: StateSpaceModelParams, state: StateSpaceModelState) -> StateSpaceModelState:
        """
        Sample the next state x_t given previous state x_{t-1} and parameters theta.

        x_t ~ p(x_t | x_{t-1}, theta)

        Parameters
        ----------
            theta: StateSpaceModelParams
                Model parameters.
            state: StateSpaceModelState
                Previous state of size N.
        Returns
        -------
            state: StateSpaceModelState
                Sampled next state of size N.
        """
        pass

    @abstractmethod
    def expected_next_state(self, theta: StateSpaceModelParams, state: StateSpaceModelState) -> StateSpaceModelState:
        """
        Compute the expected next state given current state and parameters theta.

        E[x_t | x_{t-1}, theta]

        Parameters
        ----------
            theta: StateSpaceModelParams
                Model parameters.
            state: StateSpaceModelState
                Current state.
        Returns
        -------
            state: StateSpaceModelState
                Expected next state.
        """
        pass

    @abstractmethod
    def likelihood(self, y, theta: StateSpaceModelParams, state: StateSpaceModelState) -> np.ndarray:
        """
        Compute the likelihood of observations y given states with shape (N,).

        Parameters
        ----------
            y: np.ndarray
                Observations.
            theta: StateSpaceModelParams
                Model parameters.
            state: StateSpaceModelState
                Current state of size N.
        Returns
        -------
            likelihood: np.ndarray
                Likelihood of observations with shape (N,).
        """
        pass

    @abstractmethod
    def log_likelihood(self, y, theta: StateSpaceModelParams, state: StateSpaceModelState) -> np.ndarray:
        """
        Compute the log-likelihood of observations y given states with shape (N,).

        Parameters
        ----------
            y: np.ndarray
                Observations.
            theta: StateSpaceModelParams
                Model parameters.
            state: StateSpaceModelState
                Current state of size N.
        Returns
        -------
            log_likelihood: np.ndarray
                Log-likelihood of observations with shape (N,).
        """
        pass

    @abstractmethod
    def state_transition(self, theta: StateSpaceModelParams, state_prev: StateSpaceModelState, state_next: StateSpaceModelState) -> np.ndarray:
        """
        Compute the state transition probability p(x_t | x_{t-1}, theta).

        Parameters
        ----------
            theta: StateSpaceModelParams
                Model parameters.
            state_prev: StateSpaceModelState
                Previous state of size N.
            state_next: StateSpaceModelState
                Next state of size N.
        Returns
        -------
            transition_prob: np.ndarray
                Transition probabilities with shape (N,).
        """
        pass

    @abstractmethod
    def log_state_transition(self, theta: StateSpaceModelParams, state_prev: StateSpaceModelState, state_next: StateSpaceModelState) -> np.ndarray:
        """
        Compute the log of the state transition probability log p(x_t | x_{t-1}, theta).

        Parameters
        ----------
            theta: StateSpaceModelParams
                Model parameters.
            state_prev: StateSpaceModelState
                Previous state of size N.
            state_next: StateSpaceModelState
                Next state of size N.
        Returns
        -------
            log_transition_prob: np.ndarray
                Log of transition probabilities with shape (N,).
        """
        pass
