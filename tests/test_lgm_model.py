import numpy as np
import pytest

from src.models.lgm import LGModel, LGModelParams, LGModelState

def test_lgm_params():
    # Test initialization with provided parameters
    params = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)

    assert params.a == 0.9
    assert params.sigma_y == 0.2

    with pytest.raises(ValueError):
        params_invalid = LGModelParams(a=0.9, b=1.0, sigma_x=-0.5, sigma_y=0.2)

def test_lgm_state():
    x_t = np.array([0.5, -1.0])
    state = LGModelState(x_t)

    assert state.x_t.shape == (2,)

def test_sample_initial_state():
    rng = np.random.default_rng(42)
    model = LGModel(rng=rng)
    params = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)

    state = model.sample_initial_state(params, size=1)
    x0 = state.x_t
    # Assert correct shape
    assert x0.shape == (1,)

    state = model.sample_initial_state(params, size=5)
    x0 = state.x_t
    # Assert correct shape
    assert x0.shape == (5,)
    
    # Assert different values in x0 (due to randomness)
    assert not np.all(x0 == x0[0])

def test_sample_next_state():
    rng = np.random.default_rng(42)
    model = LGModel(rng=rng)
    params = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)

    x_prev = np.array([0.0, 1.0, -1.0])
    state_prev = LGModelState(x_prev)

    state_next = model.sample_next_state(params, state_prev)
    x_next = state_next.x_t

    # Assert correct shape
    assert x_next.shape == (3,)

    # Assert that next states are not equal to previous states (due to randomness)
    assert not np.all(x_next == x_prev)

def test_sample_observation():
    rng = np.random.default_rng(42)
    model = LGModel(rng=rng)
    params = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)

    x_t = np.array([0.0, 1.0, -1.0])
    state = LGModelState(x_t)

    y_t = model.sample_observation(params, state)

    # Assert correct shape
    assert y_t.shape == (3,)

def test_expected_next_state():
    rng = np.random.default_rng(42)
    model = LGModel(rng=rng)
    params = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)

    x_t = np.array([0.0, 1.0, -1.0])
    state = LGModelState(x_t)

    state_exp = model.expected_next_state(params, state)
    x_exp = state_exp.x_t

    # Assert correct shape
    assert x_exp.shape == (3,)

    # Assert expected values
    expected_values = params.a * x_t
    assert np.allclose(x_exp, expected_values)

def test_likelihood_and_log_likelihood():
    rng = np.random.default_rng(42)
    model = LGModel(rng=rng)
    params = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)

    x_t = np.array([0.0, 1.0, -1.0])
    state = LGModelState(x_t)

    y = 0.5

    likelihoods = model.likelihood(y, params, state)
    log_likelihoods = model.log_likelihood(y, params, state)

    # Assert correct shapes
    assert likelihoods.shape == (3,)
    assert log_likelihoods.shape == (3,)

    # Assert that log likelihoods are the logarithm of likelihoods
    assert np.allclose(log_likelihoods, np.log(likelihoods))

def test_state_transition_and_log_state_transition():
    rng = np.random.default_rng(42)
    model = LGModel(rng=rng)
    params = LGModelParams(a=0.9, b=1.0, sigma_x=0.5, sigma_y=0.2)

    x_prev = np.array([0.0, 1.0, -1.0])
    state_prev = LGModelState(x_prev)

    x_next = np.array([0.5, 0.8, -0.5])
    state_next = LGModelState(x_next)

    transitions = model.state_transition(params, state_prev, state_next)
    log_transitions = model.log_state_transition(params, state_prev, state_next)

    # Assert correct shapes
    assert transitions.shape == (3,)
    assert log_transitions.shape == (3,)

    # Assert that log transitions are the logarithm of transitions
    assert np.allclose(log_transitions, np.log(transitions))