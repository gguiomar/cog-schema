#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

def sample_next_state(current_state, transition_matrix):
    """Sample the next state based on the current state and transition matrix."""
    return np.random.choice(len(transition_matrix), p=transition_matrix[current_state])

# Define transition matrices
T1 = np.array([[0.6, 0.3, 0.05, 0.05], 
               [0.3, 0.6, 0.05, 0.05], 
               [0.1, 0.1, 0.4, 0.4], 
               [0.1, 0.1, 0.4, 0.4]])
T2 = np.array([[0.2, 0.2, 0.3, 0.3], 
               [0.2, 0.2, 0.3, 0.3], 
               [0.05, 0.05, 0.6, 0.3], 
               [0.05, 0.05, 0.3, 0.6]])

# Parameters
num_steps = 10000          # Total time steps
split_point = num_steps // 2  # Switch from T1 to T2 at 5000
num_states = 4             # Number of states
eta = 0.01                 # Learning rate for delta rule
min_sigma = 1e-3           # Minimum sigma to prevent division by zero


# Simulate the Markov chain
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
    surprisals[t] = -np.log(p + 1e-12)  # Add small constant to avoid log(0)

# Initialize Gaussian parameters for each state
mu_s = np.ones(num_states)      # Initial mean = 1.0
v_s = np.ones(num_states)       # Initial variance = 1.0
sigma_s = np.sqrt(v_s)          # Initial standard deviation

# Arrays to store meta-surprisal and gradients
metasurprisals = np.zeros(num_steps)
grad_mu_history = np.zeros((num_states, num_steps))
grad_v_history = np.zeros((num_states, num_steps))
#%%
# Compute meta-surprisal and update Gaussian parameters
for t in range(num_steps):
    s_t = states[t]             # Current state
    s_val = surprisals[t]       # Current surprisal
    
    # Compute meta-surprisal for the current state
    density = norm.pdf(s_val, mu_s[s_t], sigma_s[s_t])
    metasurprisals[t] = np.log(max(density, 1e-10))  # Simplified meta-surprisal
    
    # Compute gradients and update parameters for the current state
    grad_mu = (s_val - mu_s[s_t]) / (sigma_s[s_t]**2)
    grad_mu_history[s_t, t] = grad_mu
    mu_s[s_t] -= eta * grad_mu
    
    grad_v = (1 / (2 * v_s[s_t])) - ((s_val - mu_s[s_t])**2 / (2 * v_s[s_t]**2))
    grad_v_history[s_t, t] = grad_v
    v_s[s_t] -= eta * grad_v
    v_s[s_t] = max(v_s[s_t], min_sigma**2)  # Ensure variance stays positive
    sigma_s[s_t] = np.sqrt(v_s[s_t])

#%%
sigma_smooth = 50          # Smoothing parameter for plots
# Smooth the data for plotting
smoothed_metasurprisals = gaussian_filter1d(metasurprisals, sigma=sigma_smooth)
smoothed_grad_mu = np.zeros((num_states, num_steps))
smoothed_grad_v = np.zeros((num_states, num_steps))
for s in range(num_states):
    smoothed_grad_mu[s] = gaussian_filter1d(grad_mu_history[s], sigma=sigma_smooth)
    smoothed_grad_v[s] = gaussian_filter1d(grad_v_history[s], sigma=sigma_smooth)


# Create plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Total Meta-surprisal
ax1.plot(range(num_steps), smoothed_metasurprisals, color='black', label='Total Meta-surprisal')
ax1.axvline(x=split_point, color='gray', linestyle='--', label='Switch (T1 to T2)')
ax1.set_ylabel('Smoothed Meta-surprisal')
ax1.set_title('Smoothed Total Meta-surprisal')
ax1.legend()
ax1.grid(True)

# Plot 2: Gradients of mu_s
for s in range(num_states):
    ax2.plot(range(num_steps), smoothed_grad_mu[s], label=f'State {s}')
ax2.axvline(x=split_point, color='gray', linestyle='--')
ax2.set_ylabel('Smoothed grad_mu')
ax2.set_title('Smoothed Gradients of mu_s')
ax2.legend()
ax2.grid(True)

# Plot 3: Gradients of v_s
for s in range(num_states):
    ax3.plot(range(num_steps), smoothed_grad_v[s], label=f'State {s}')
ax3.axvline(x=split_point, color='gray', linestyle='--')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Smoothed grad_v')
ax3.set_title('Smoothed Gradients of v_s')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
# %%
