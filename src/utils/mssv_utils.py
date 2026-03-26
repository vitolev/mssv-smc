from typing import List
import numpy as np
from src.models.mssv import MSSVState

def compute_transition_counts(samples: List[MSSVState]) -> List[np.ndarray]:
    """
    Compute transition counts between regimes for each trajectory m.

    Parameters
    ----------
    samples : list of MSSVModelState, size T+1

    Returns
    -------
    transition_counts : list of np.ndarray, length M
        Each element is a (K, K) transition matrix for trajectory m.
    """

    s_matrix = np.array([state.s_t for state in samples])  # (T+1, M, K)

    state_idx = np.argmax(s_matrix, axis=-1)  # (T+1, M)

    T_plus_1, M = state_idx.shape
    K = s_matrix.shape[-1]

    transition_counts = [np.zeros((K, K), dtype=int) for _ in range(M)]

    for m in range(M):
        i = state_idx[:-1, m]
        j = state_idx[1:, m]
        np.add.at(transition_counts[m], (i, j), 1)

    return transition_counts

