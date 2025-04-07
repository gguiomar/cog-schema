
#%%
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_states = 4
alpha_slow = 0.001  # Slow learning rate for Q_t
alpha_fast = 0.1    # Fast learning rate for P_t
beta_smooth = 0.1   # Smoothing factor for KL and JSD
T1 = 1000           # Steps for Chain 1
T2 = 1000           # Steps for Chain 2
T = T1 + T2         # Total steps
eps = 1e-10         # Avoid log(0)

# Transition matrices
P1 = np.array([  # Favors states 0 and 1
    [0.8, 0.15, 0.025, 0.025],
    [0.15, 0.8, 0.025, 0.025],
    [0.25, 0.25, 0.4, 0.1],
    [0.25, 0.25, 0.1, 0.4]
])

P2 = np.array([  # Favors states 2 and 3
    [0.4, 0.1, 0.25, 0.25],
    [0.1, 0.4, 0.25, 0.25],
    [0.025, 0.025, 0.8, 0.15],
    [0.025, 0.025, 0.15, 0.8]
])

# Step function for Markov chain
def step(current_state, transition_matrix):
    return np.random.choice(n_states, p=transition_matrix[current_state])

# Metric functions
def kl_divergence(p, q, eps=1e-10):
    return np.sum(p * np.log((p + eps) / (q + eps)))

def jensen_shannon(p, q, eps=1e-10):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

def total_variation(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def chi_squared(p, q, eps=1e-10):
    return np.sum((p - q)**2 / (q + eps))

def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)

# Initialize arrays
states = np.zeros(T, dtype=int)
Q_t = np.ones(n_states) / n_states  # Slow estimate
P_t = np.ones(n_states) / n_states  # Fast estimate
D_kl = np.zeros(T)
D_kl_smooth = np.zeros(T)
D_jsd = np.zeros(T)
D_jsd_smooth = np.zeros(T)
D_tvd = np.zeros(T)
D_chi2 = np.zeros(T)
D_hellinger = np.zeros(T)
S_surprise = np.zeros(T)

# Simulate the Markov chain and compute metrics
current_state = 0
for t in range(T):
    # Simulate next state
    transition_matrix = P1 if t < T1 else P2
    next_state = step(current_state, transition_matrix)
    states[t] = next_state

    # Update slow and fast estimates
    one_hot = np.zeros(n_states)
    one_hot[next_state] = 1
    Q_t = (1 - alpha_slow) * Q_t + alpha_slow * one_hot
    P_t = (1 - alpha_fast) * P_t + alpha_fast * one_hot

    # Compute raw metrics
    D_kl[t] = kl_divergence(Q_t, P_t)
    D_jsd[t] = jensen_shannon(Q_t, P_t)
    D_tvd[t] = total_variation(Q_t, P_t)
    D_chi2[t] = chi_squared(Q_t, P_t)
    D_hellinger[t] = hellinger(Q_t, P_t)
    S_surprise[t] = -np.log(Q_t[next_state] + eps)

    # Smooth KL and JSD
    if t == 0:
        D_kl_smooth[t] = D_kl[t]
        D_jsd_smooth[t] = D_jsd[t]
    else:
        D_kl_smooth[t] = (1 - beta_smooth) * D_kl_smooth[t-1] + beta_smooth * D_kl[t]
        D_jsd_smooth[t] = (1 - beta_smooth) * D_jsd_smooth[t-1] + beta_smooth * D_jsd[t]

    current_state = next_state

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

# KL Divergence (Smoothed)
axs[0, 0].plot(D_kl_smooth, label='KL Divergence (Smoothed)')
axs[0, 0].axvline(T1, color='k', linestyle='--', label='Transition')
axs[0, 0].set_title('KL Divergence (Smoothed)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Jensen-Shannon Divergence (Smoothed)
axs[0, 1].plot(D_jsd_smooth, label='Jensen-Shannon Divergence (Smoothed)')
axs[0, 1].axvline(T1, color='k', linestyle='--')
axs[0, 1].set_title('Jensen-Shannon Divergence (Smoothed)')
axs[0, 1].grid(True)

# Total Variation Distance
axs[1, 0].plot(D_tvd, label='Total Variation Distance')
axs[1, 0].axvline(T1, color='k', linestyle='--')
axs[1, 0].set_title('Total Variation Distance')
axs[1, 0].grid(True)

# Chi-Squared Divergence
axs[1, 1].plot(D_chi2, label='Chi-Squared Divergence')
axs[1, 1].axvline(T1, color='k', linestyle='--')
axs[1, 1].set_title('Chi-Squared Divergence')
axs[1, 1].grid(True)

# Hellinger Distance
axs[2, 0].plot(D_hellinger, label='Hellinger Distance')
axs[2, 0].axvline(T1, color='k', linestyle='--')
axs[2, 0].set_title('Hellinger Distance')
axs[2, 0].grid(True)

# Surprise
axs[2, 1].plot(S_surprise, label='Surprise (-log Q_t(s_t))')
axs[2, 1].axvline(T1, color='k', linestyle='--')
axs[2, 1].set_title('Surprise')
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
# %%
