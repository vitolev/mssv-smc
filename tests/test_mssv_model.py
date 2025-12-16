import numpy as np
import pytest

from src.models.mssv import MSSVModel, MSSVModelParams

def test_mssv_params():
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

def test_sample_initial_state():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng=rng)
    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.2, 0.3],
        P=[[0.9, 0.1], [0.2, 0.8]]
    )

    h0, s0 = model.sample_initial_state(params)

    assert isinstance(h0, float)
    assert isinstance(s0, int)
    assert 0 <= s0 < len(params.mu)

def test_sample_transition():
    rng = np.random.default_rng(42)
    model = MSSVModel(rng)

    params = MSSVModelParams(
        mu=[0.0, 1.0],
        phi=[0.9, 0.8],
        sigma_eta=[0.1, 0.2],
        P=[[1.0, 0.0], [0.0, 1.0]]  # deterministic regimes
    )

    state = (0.5, 0)
    h1, s1 = model.sample_transition(params, state)

    assert s1 == 0  # forced by P
    assert isinstance(h1, float)

    params.P = [[0.0, 1.0], [1.0, 0.0]]  # switch regimes
    state = (0.5, 0)
    h2, s2 = model.sample_transition(params, state)

    assert s2 == 1  # forced by P
    assert isinstance(h2, float)