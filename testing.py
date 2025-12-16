import numpy as np
from src.models.mssv import MSSVModelParams, MSSVModel
from src.filters.smc.bootstrap_pf import BootstrapParticleFilter
from src.filters.smc.resampling import systematic_resampling
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# True parameters
theta = MSSVModelParams(
    mu=[-1.0, 1.0],
    phi=[0.95, 0.95],
    sigma_eta=[0.2, 0.3],
    P=[[0.95, 0.05],
        [0.05, 0.95]]
)

model = MSSVModel(rng=rng)

def simulate_mssv(model, theta, T, rng):
    states = []
    observations = []

    state = model.sample_initial_state(theta)
    for _ in range(T):
        state = model.sample_transition(theta, state)
        h_t, _ = state
        y_t = rng.normal(0.0, np.exp(0.5 * h_t))

        states.append(state)
        observations.append(y_t)

    return states, np.array(observations)

# Simulate data
T = 300
true_states, y = simulate_mssv(model, theta, T, rng)

# Run particle filter
pf = BootstrapParticleFilter(model, n_particles=1000, resampler=systematic_resampling, rng=rng)
history = pf.run(y, theta)

# =================================================
# Validation diagnostics
# =================================================

h_true = np.array([h for h, s in true_states])
s_true = np.array([s for h, s in true_states])

T = len(history)
N = len(history[0][0])
K = len(theta.mu)  # number of regimes

# Extract particles and weights as arrays
particles_array = np.array([ [p for p in history[t][0]] for t in range(T) ])  # shape (T, N, 2)
weights_array = np.array([ history[t][1] for t in range(T) ])                 # shape (T, N)

h_est = np.sum(particles_array[:, :, 0] * weights_array, axis=1)  # shape (T,)

s_particles = particles_array[:, :, 1].astype(int)
s_est_prob = np.zeros((T, K))
for k in range(K):
    # Sum weights of particles in regime k for all time steps
    mask = (s_particles == k)  # shape (T, N), True where particle is in regime k
    s_est_prob[:, k] = np.sum(weights_array * mask, axis=1)

s_est = np.argmax(s_est_prob, axis=1)

# =================================================
# Plots
# =================================================

plt.figure(figsize=(12, 4))
plt.plot(h_true, label="True log-volatility")
plt.plot(h_est, label="Filtered mean", linewidth=2)
plt.legend()
plt.title("Volatility: true vs filtered")
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(s_true, label="True regime")
plt.plot(s_est, label="Estimated regime")
plt.plot(s_est_prob[:, 1], label="Estimated P(s=1)", linestyle='--', alpha=0.4)
plt.legend()
plt.title("Filtered regime probabilities")
plt.show()


