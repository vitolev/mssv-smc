import numpy as np
import pytest

from src.models.mssv import MSSVModel, MSSVModelParams, MSSVModelState

def test_mssv_params():
    # Test initialization with provided parameters
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    assert len(params.mu) == 2
    assert params.P[0][0] == 0.9

    with pytest.raises(ValueError):
        params_invalid = MSSVModelParams(
            mu=[0.0, 1.0],
            phi=[0.9],
            sigma_eta=[0.2, 0.3],
            P=[[0.9, 0.1], [0.2, 0.8]]
        )
    
    # Test sampling from prior
    rng = np.random.default_rng(42)
    params_prior = MSSVModelParams(rng=rng, num_regimes=3)
    assert len(params_prior.mu) == 3
    assert params_prior.P.shape == (3, 3)

def test_sample_initial_state():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    state = model.sample_initial_state(params, size=1)
    h0, s0 = state.h_t, state.s_t
    # Assert correct shapes
    assert h0.shape == (1,)
    assert s0.shape == (1, 2)

    state = model.sample_initial_state(params, size=5)
    h0, s0 = state.h_t, state.s_t
    # Assert correct shapes
    assert h0.shape == (5,)
    assert s0.shape == (5, 2)
    
    # Assert different values in h0 (due to randomness)
    assert not np.all(h0 == h0[0])

def test_sample_observation():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h_t = np.array([0.5])
    s_t = np.array([[1, 0]])  # One-hot for regime 0
    y_t = model.sample_observation(params, MSSVModelState(h_t, s_t))
    
    # Assert correct shape and type
    assert y_t.shape == (1,)
    assert isinstance(y_t[0], np.floating)

    h_t = np.array([0.5, 1.0, -0.5])
    s_t = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot for regimes
    y_t_samples = model.sample_observation(params, MSSVModelState(h_t, s_t))

    # Assert correct number of samples
    assert y_t_samples.shape == (3,)

def test_sample_next_state():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h_prev = np.array([0.5])
    s_prev = np.array([[1, 0]])  # One-hot for regime 0
    next_state = model.sample_next_state(params, MSSVModelState(h_prev, s_prev))
    h_next, s_next = next_state.h_t, next_state.s_t

    # Assert correct shapes
    assert h_next.shape == (1,)
    assert s_next.shape == (1, 2)

    h_prev = np.array([0.5, 1.0, -0.5])
    s_prev = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot for regimes
    next_state = model.sample_next_state(params, MSSVModelState(h_prev, s_prev))
    h_next, s_next = next_state.h_t, next_state.s_t

    # Assert correct shapes
    assert h_next.shape == (3,)
    assert s_next.shape == (3, 2)

def test_expected_next_state():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h_prev = np.array([0.5])
    s_prev = np.array([[1, 0]])  # One-hot for regime 0
    expected_state = model.expected_next_state(params, MSSVModelState(h_prev, s_prev))
    h_exp, s_exp = expected_state.h_t, expected_state.s_t

    # Assert correct shapes
    assert h_exp.shape == (1,)
    assert s_exp.shape == (1, 2)

    h_prev = np.array([0.5, 1.0, -0.5])
    s_prev = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot for regimes
    expected_state = model.expected_next_state(params, MSSVModelState(h_prev, s_prev))
    h_exp, s_exp = expected_state.h_t, expected_state.s_t

    # Assert correct shapes
    assert h_exp.shape == (3,)
    assert s_exp.shape == (3, 2)

def test_likelihood():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h_t = np.array([0.5])
    s_t = np.array([[1, 0]])  # One-hot for regime 0
    y_t = 0.7
    likelihood = model.likelihood(y_t, params, MSSVModelState(h_t, s_t))

    # Assert likelihood is a positive number
    assert likelihood.shape == (1,)
    assert likelihood > 0

    h_t = np.array([0.5, 1.0, -0.5])
    s_t = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot for regimes
    y_t_values = [0.7, -1.2, 0.3]
    likelihoods = model.likelihood(y_t_values, params, MSSVModelState(h_t, s_t))
    
    # Assert likelihoods are positive numbers and shape matches
    assert likelihoods.shape == (3,)
    assert np.all(likelihoods > 0)

def test_log_likelihood():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h_t = np.array([0.5])
    s_t = np.array([[1, 0]])  # One-hot for regime 0
    y_t = 0.7
    log_likelihood = model.log_likelihood(y_t, params, MSSVModelState(h_t, s_t))

    # Assert log-likelihood is a number
    assert log_likelihood.shape == (1,)

    h_t = np.array([0.5, 1.0, -0.5])
    s_t = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot for regimes
    y_t_values = [0.7, -1.2, 0.3]
    log_likelihoods = model.log_likelihood(y_t_values, params, MSSVModelState(h_t, s_t))
    
    # Assert log-likelihoods shape matches
    assert log_likelihoods.shape == (3,)

def test_state_transition():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h_prev = np.array([0.5])
    s_prev = np.array([[1, 0]])  # One-hot for regime 0
    h_next = np.array([0.6])
    s_next = np.array([[1, 0]])  # One-hot for regime 0
    prob = model.state_transition(
        params,
        MSSVModelState(h_prev, s_prev),
        MSSVModelState(h_next, s_next)
    )

    # Assert log-probability shape and type
    assert prob.shape == (1,)
    assert isinstance(prob[0], np.floating)

    h_prev = np.array([0.5, 1.0])
    s_prev = np.array([[1, 0], [0, 1]])  # One-hot for regimes
    h_next = np.array([0.6, 1.2])
    s_next = np.array([[1, 0], [1, 0]])  # One-hot for regimes
    probs = model.state_transition(
        params,
        MSSVModelState(h_prev, s_prev),
        MSSVModelState(h_next, s_next)
    )
    
    # Assert log-probabilities shape matches and type
    assert probs.shape == (2,)
    assert np.all([isinstance(p, np.floating) for p in probs])

def test_log_state_transition():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h_prev = np.array([0.5])
    s_prev = np.array([[1, 0]])  # One-hot for regime 0
    h_next = np.array([0.6])
    s_next = np.array([[1, 0]])  # One-hot for regime 0
    log_prob = model.log_state_transition(
        params,
        MSSVModelState(h_prev, s_prev),
        MSSVModelState(h_next, s_next)
    )

    # Assert log-probability shape and type
    assert log_prob.shape == (1,)
    assert isinstance(log_prob[0], np.floating)

    h_prev = np.array([0.5, 1.0])
    s_prev = np.array([[1, 0], [0, 1]])  # One-hot for regimes
    h_next = np.array([0.6, 1.2])
    s_next = np.array([[1, 0], [1, 0]])  # One-hot for regimes
    log_probs = model.log_state_transition(
        params,
        MSSVModelState(h_prev, s_prev),
        MSSVModelState(h_next, s_next)
    )
    
    # Assert log-probabilities shape matches and type
    assert log_probs.shape == (2,)
    assert np.all([isinstance(p, np.floating) for p in log_probs])

