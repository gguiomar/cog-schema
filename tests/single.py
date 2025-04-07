import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def sample_next_state(current_state, transition_matrix):
    """Sample the next state based on the current state and transition matrix."""
    return np.random.choice(len(transition_matrix), p=transition_matrix[current_state])

def estimate_transition_matrix(states, num_states, epsilon=1e-6):
    """
    Estimate the transition matrix from a sequence of states using Laplace smoothing.
    
    Args:
        states (list or np.ndarray): Sequence of states [s_0, s_1, ..., s_{T-1}]
        num_states (int): Number of possible states (K)
        epsilon (float): Smoothing parameter to avoid zero probabilities
    
    Returns:
        transition_matrix (np.ndarray): Estimated K x K transition matrix
    """
    counts = np.zeros((num_states, num_states))
    for t in range(1, len(states)):
        prev_state = states[t-1]
        curr_state = states[t]
        counts[prev_state, curr_state] += 1
    
    total_transitions = np.sum(counts, axis=1, keepdims=True)
    transition_matrix = (counts + epsilon) / (total_transitions + num_states * epsilon)
    return transition_matrix

def calculate_surprisal(states, num_states, current_time, epsilon=1e-6):
    """
    Calculate the surprisal of the current state at the given time step using samples.
    
    Args:
        states (list or np.ndarray): Sequence of states [s_0, s_1, ..., s_{T-1}]
        num_states (int): Number of possible states (K)
        current_time (int): Time step t for which to compute surprisal of s_t
        epsilon (float): Smoothing parameter
    
    Returns:
        surprisal (float): Surprisal of the current state s_t
    """
    if current_time < 1 or current_time >= len(states):
        return 0.0  # Return 0 for invalid time steps
    
    # Estimate transition matrix using all samples up to current_time
    transition_matrix = estimate_transition_matrix(states[:current_time+1], num_states, epsilon)
    
    # Get previous and current states
    prev_state = states[current_time - 1]
    curr_state = states[current_time]
    
    # Compute surprisal
    prob = transition_matrix[prev_state, curr_state]
    surprisal = -np.log(prob)
    return surprisal

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
sigma_smooth = 10          # Smoothing parameter for plots
epsilon = 1e-6             # Smoothing parameter for transition probability estimation

# Simulate the Markov chain
states = np.zeros(num_steps, dtype=int)
states[0] = 0  # Start at state 0
for t in range(1, split_point):
    states[t] = sample_next_state(states[t-1], T1)
for t in range(split_point, num_steps):
    states[t] = sample_next_state(states[t-1], T2)

# Compute surprisals using sample-based estimation
surprisals = np.zeros(num_steps)
for t in range(1, num_steps):
    surprisals[t] = calculate_surprisal(states, num_states, t, epsilon)

# Compute the derivative of surprisal (approximated as difference)
ds_dt = np.zeros(num_steps)
for t in range(1, num_steps):
    ds_dt[t] = surprisals[t] - surprisals[t-1]

# Initialize Gaussian parameters for each state (for consistency with previous setup)
mu_s = np.ones(num_states)      # Initial mean = 1.0
v_s = np.ones(num_states)       # Initial variance = 1.0
sigma_s = np.sqrt(v_s)          # Initial standard deviation

# Arrays to store gradients
grad_mu_history = np.zeros((num_states, num_steps))
grad_v_history = np.zeros((num_states, num_steps))

# Update Gaussian parameters (even though we're not using meta-surprisal, to keep gradients)
for t in range(num_steps):
    s_t = states[t]             # Current state
    s_val = surprisals[t]       # Current surprisal
    
    # Compute gradients and update parameters for the current state
    grad_mu = (s_val - mu_s[s_t]) / (sigma_s[s_t]**2)
    grad_mu_history[s_t, t] = grad_mu
    mu_s[s_t] -= eta * grad_mu
    
    grad_v = (1 / (2 * v_s[s_t])) - ((s_val - mu_s[s_t])**2 / (2 * v_s[s_t]**2))
    grad_v_history[s_t, t] = grad_v
    v_s[s_t] -= eta * grad_v
    v_s[s_t] = max(v_s[s_t], min_sigma**2)  # Ensure variance stays positive
    sigma_s[s_t] = np.sqrt(v_s[s_t])

# Compute the sum of gradients across all states at each time step
sum_grad_mu = np.sum(grad_mu_history, axis=0)
sum_grad_v = np.sum(grad_v_history, axis=0)

# Smooth the data for plotting
smoothed_ds_dt = gaussian_filter1d(ds_dt, sigma=sigma_smooth)
smoothed_grad_mu = np.zeros((num_states, num_steps))
smoothed_grad_v = np.zeros((num_states, num_steps))
for s in range(num_states):
    smoothed_grad_mu[s] = gaussian_filter1d(grad_mu_history[s], sigma=sigma_smooth)
    smoothed_grad_v[s] = gaussian_filter1d(grad_v_history[s], sigma=sigma_smooth)

# Smooth the sum of gradients
smoothed_sum_grad_mu = gaussian_filter1d(sum_grad_mu, sigma=sigma_smooth)
smoothed_sum_grad_v = gaussian_filter1d(sum_grad_v, sigma=sigma_smooth)

# Create plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Derivative of Surprisal
ax1.plot(range(num_steps), smoothed_ds_dt, color='black', label='d(Surprisal)/dt')
ax1.axvline(x=split_point, color='gray', linestyle='--', label='Switch (T1 to T2)')
ax1.set_ylabel('Smoothed d(Surprisal)/dt')
ax1.set_title('Smoothed Derivative of Surprisal')
ax1.legend()
ax1.grid(True)

# Plot 2: Gradients of mu_s with sum
for s in range(num_states):
    ax2.plot(range(num_steps), smoothed_grad_mu[s], label=f'State {s}')
ax2.plot(range(num_steps), smoothed_sum_grad_mu, color='black', linestyle='--', label='Sum of grad_mu')
ax2.axvline(x=split_point, color='gray', linestyle='--')
ax2.set_ylabel('Smoothed grad_mu')
ax2.set_title('Smoothed Gradients of mu_s with Sum')
ax2.legend()
ax2.grid(True)

# Plot 3: Gradients of v_s with sum
for s in range(num_states):
    ax3.plot(range(num_steps), smoothed_grad_v[s], label=f'State {s}')
ax3.plot(range(num_steps), smoothed_sum_grad_v, color='black', linestyle='--', label='Sum of grad_v')
ax3.axvline(x=split_point, color='gray', linestyle='--')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Smoothed grad_v')
ax3.set_title('Smoothed Gradients of v_s with Sum')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()