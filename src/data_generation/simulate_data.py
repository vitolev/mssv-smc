import numpy as np
from src.models.base import StateSpaceModel, StateSpaceModelParams


def simulate_data(model: StateSpaceModel, theta: StateSpaceModelParams, T, rng: np.random.Generator):
    """
    Simulate data from the state space model.

    Args:
        model: An instance of a StateSpaceModel.
        theta: Model parameters.
        T: Length of the time series to simulate.
        rng: A numpy random number generator.

    Returns:
        states: A list of latent states.
        y: A numpy array of observations.
    """

    states = []
    y = []

    state = model.sample_initial_state(theta)
    for _ in range(T):
        state = model.sample_next_state(theta, state)
        y_t = model.sample_observation(theta, state)

        states.append(state)
        y.append(y_t)

    return states, np.array(y)