from abc import ABC, abstractmethod
import numpy as np
from typing import Type

class StateSpaceModelParams(ABC):
    """
    Base class for state space model parameters.
    """
    def __init__(self):
        pass

    @abstractmethod
    def sample_prior(self, rng: np.random.Generator, num_regimes: int):
        """
        Sample parameters from the prior distribution.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator to use for sampling.
        num_regimes : int
            The number of regimes in the model, which may influence the shape of the sampled parameters.
        """
        pass

    @abstractmethod
    def log_prior_density(self) -> float:
        """
        Compute the log prior density of the parameters.

        Returns
        -------
        log_prior: float
            The log prior density evaluated at the current parameter values.
        """
        pass

    @abstractmethod
    def sample_transition(self, rng: np.random.Generator) -> "StateSpaceModelParams":
        """
        Sample new parameters from a proposal distribution for PMMH.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator to use for sampling.

        Returns
        -------
        new_params: StateSpaceModelParams
            A new instance of StateSpaceModelParams sampled from the proposal distribution.
        """
        pass

    @abstractmethod
    def log_transition_density(self, other: "StateSpaceModelParams") -> float:
        """
        Compute the log q(other | self) where q is the proposal distribution.

        Parameters
        ----------
        other : StateSpaceModelParams
            The other parameter instance to evaluate the transition density against.

        Returns
        -------
        log_density: float
            The log transition density evaluated at the current parameter values given the other parameters.
        """
        pass

    @abstractmethod
    def sample_from_data(self, x: np.ndarray, y: np.ndarray) -> "StateSpaceModelParams":
        """
        Sample new parameters from the conditional distribution p(theta | x, y).

        Parameters
        ----------
        x : np.ndarray
            Latent states.
        y : np.ndarray
            Observations.

        Returns
        -------
        new_params: StateSpaceModelParams
            A new instance of StateSpaceModelParams sampled from the conditional distribution given x and y.
        """
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

    @abstractmethod
    def add(self, other: "StateSpaceModelState") -> "StateSpaceModelState":
        """
        Extend the current state by adding another state
        """
        pass


def validate_types(method):
    def wrapper(self, *args, **kwargs):
        # convention: theta first, state second (or later)
        for arg in args:
            if isinstance(arg, StateSpaceModelParams):
                self._check_params(arg)
            if isinstance(arg, StateSpaceModelState):
                self._check_state(arg)
        return method(self, *args, **kwargs)
    return wrapper

class StateSpaceModel(ABC):
    """
    Generic state-space model:
        x_t ~ f(x_t | x_{t-1}, theta)
        y_t ~ g(y_t | x_t, theta)
    """
    params_type = StateSpaceModelParams
    state_type = StateSpaceModelState

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
    def transition_density(self, theta: StateSpaceModelParams, state_prev: StateSpaceModelState, state_next: StateSpaceModelState) -> np.ndarray:
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
    def log_transition_density(self, theta: StateSpaceModelParams, state_prev: StateSpaceModelState, state_next: StateSpaceModelState) -> np.ndarray:
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
