#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

def sample_next_state(current_state, transition_matrix):
    """Sample a new state given the current state and a transition matrix."""
    return np.random.choice(range(len(transition_matrix)), p=transition_matrix[current_state])

# Define transition matrices
T1 = np.array([[0.6, 0.3, 0.05, 0.05], [0.3, 0.6, 0.05, 0.05], [0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4]])
T2 = np.array([[0.2, 0.2, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3], [0.05, 0.05, 0.6, 0.3], [0.05, 0.05, 0.3, 0.6]])

# Parameters
num_steps = 10000
split_point = num_steps // 2  # Switch from T1 to T2 at 5000
window_size = 100             # Sliding window for fitting Gaussians
num_states = 4                # Number of states
sigma = 10                    # Gaussian smoothing parameter

# Simulate states
states = np.zeros(num_steps, dtype=int)
states[0] = 0  # Start at state 0
for t in range(1, split_point):
    states[t] = sample_next_state(states[t-1], T1)
for t in range(split_point, num_steps):
    states[t] = sample_next_state(states[t-1], T2)

# Compute surprisals
surprisals = np.zeros(num_steps)
for t in range(1, num_steps):
    p = T1[states[t-1], states[t]] if t < split_point else T2[states[t-1], states[t]]
    surprisals[t] = -np.log(p + 1e-12)

# Compute per-state meta-surprisal
metasurprisals = np.zeros(num_steps)  # Total m_t
times_per_state = [[] for _ in range(num_states)]
meta_per_state = [[] for _ in range(num_states)]

#%%
for t in range(num_steps):
    if t < window_size:
        # Not enough data: set all to 0
        metasurprisals[t] = 0.0
        for s in range(num_states):
            meta_per_state[s].append(0.0)
            if s == states[t]:
                times_per_state[s].append(t)
    else:
        # Define sliding window
        window_start = t - window_size
        window_surprisals = surprisals[window_start:t]
        window_states = states[window_start:t]
        
        # Fit Gaussian for each state in the window
        q_s = {}
        for s in range(num_states):
            state_surprisals = window_surprisals[window_states == s]
            if len(state_surprisals) > 1:  # Need at least 2 points for variance
                mu_s = np.mean(state_surprisals)
                sigma_s = np.std(state_surprisals, ddof=1)
                if sigma_s == 0:  # Handle zero variance
                    sigma_s = 1e-6
                q_s[s] = (mu_s, sigma_s)
            else:
                q_s[s] = (1.0, 1.0)  # Default: mean=1, std=1 if insufficient data
        
        # Current state and surprisal
        s_t = states[t]
        s_val = surprisals[t]
        
        # Compute m_t(s) for all states
        for s in range(num_states):
            mu_s, sigma_s = q_s[s]
            density = norm.pdf(s_val, mu_s, sigma_s)
            if s == s_t:
                m_t_s = -np.log(max(density, 1e-10))  # Total m_t is from active state
                metasurprisals[t] = m_t_s
                meta_per_state[s].append(m_t_s)
                times_per_state[s].append(t)
            else:
                meta_per_state[s].append(0.0)  # Inactive states contribute 0

# Smooth the per-state meta-surprisal data
smoothed_meta_per_state = []
for s in range(num_states):
    smoothed_m_t_s = gaussian_filter1d(np.array(meta_per_state[s]), sigma=sigma)
    smoothed_meta_per_state.append(smoothed_m_t_s[times_per_state[s]])

# Smooth the total meta-surprisal
smoothed_total_metasurprisals = gaussian_filter1d(metasurprisals, sigma=sigma)

#%%
# Plot smoothed per-state meta-surprisals and total meta-surprisal
plt.figure(figsize=(12, 6))
for s in range(num_states):
    plt.plot(times_per_state[s], smoothed_meta_per_state[s], label=f"State {s}")
#plt.plot(range(num_steps), smoothed_total_metasurprisals, color='black', linewidth=2, label="Total Meta-surprisal")
plt.axvline(x=split_point, color='gray', linestyle='--', label="Chain Switch (T1 to T2)")
plt.xlabel("Time step")
plt.ylabel("Smoothed Meta-surprisal")
#plt.title(f"Smoothed Per-State and Total Meta-surprisal (sigma={sigma}, Switch at {split_point})")
plt.legend()
plt.grid(True)
plt.show()

# Verify sum condition (diagnostic)
total_from_sum = np.zeros(num_steps)
for t in range(num_steps):
    total_from_sum[t] = sum(meta_per_state[s][t] for s in range(num_states))
print("Sum of per-state m_t equals total m_t:", np.allclose(total_from_sum, metasurprisals))
# %%
