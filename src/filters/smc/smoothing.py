import numpy as np

def get_smoothing_trajectories(history, n_trajectories=None):
    """
    Reconstruct full trajectories (smoothing samples) from particle filter history.

    Parameters
    ----------
    history : list of tuples
        Each element is (particles, weights, indices) at each time step.
        - particles: StateSpaceModelState with batched N particles
        - weights: np.ndarray of shape (N,)
        - indices: np.ndarray of shape (N,) mapping particles at t-1 -> particles at t
          (t=0 has empty indices)
    n_trajectories : int or None
        Number of trajectories to sample. If None, returns all N trajectories.

    Returns
    -------
    trajectories : list of lists
        trajectories[i][t] is the state at time t of trajectory i.
    """
    T = len(history)
    N = len(history[0][1])

    if n_trajectories is None:
        n_trajectories = N

    # Sample final particles according to final weights
    final_weights = history[-1][1]
    final_indices = np.random.choice(N, size=n_trajectories, p=final_weights)

    # Initialize storage
    trajectories = [ [None]*T for _ in range(n_trajectories) ]

    # Fill final time
    for i, idx in enumerate(final_indices):
        trajectories[i][T-1] = history[T-1][0][idx]

    # Trace backward
    for t in reversed(range(1, T)):
        indices = history[t][2]  # maps t-1 -> t
        for i in range(n_trajectories):
            parent_idx = indices[final_indices[i]]  # parent at t-1
            trajectories[i][t-1] = history[t-1][0][parent_idx]
            final_indices[i] = parent_idx  # update for next backward step

    return trajectories
