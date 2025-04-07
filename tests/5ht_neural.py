#%%

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_states = 4
alpha_slow = 0.001   # Slow learning rate for Q_t
alpha_fast = 0.1     # Fast learning rate for P_t
beta_smooth = 0.1    # Smoothing factor for KL and JSD
T1 = 1000            # Steps for Chain 1
T2 = 1000            # Steps for Chain 2
T = T1 + T2          # Total steps
eps = 1e-10          # Avoid log(0)

# Time constant for "serotonergic" leaky integrator
# gamma = 1/tau, so if tau is large, gamma is small
gamma = 0.1         
S_max = 2.5          # Saturation threshold for the 5-HT-like signal

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
D_jsd = np.zeros(T)
D_tvd = np.zeros(T)
D_chi2 = np.zeros(T)
D_hellinger = np.zeros(T)
S_surprise = np.zeros(T)
D_kl_smooth = np.zeros(T)
D_jsd_smooth = np.zeros(T)

# "Serotonin-like" signal array
S_5ht = np.zeros(T)

# Simulation
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

    # Compute divergences
    D_kl[t] = kl_divergence(Q_t, P_t, eps=eps)
    D_jsd[t] = jensen_shannon(Q_t, P_t, eps=eps)
    D_tvd[t] = total_variation(Q_t, P_t)
    D_chi2[t] = chi_squared(Q_t, P_t, eps=eps)
    D_hellinger[t] = hellinger(Q_t, P_t)
    
    # Surprise (fast or slow estimate; here we use slow Q_t)
    S_surprise[t] = -np.log(Q_t[next_state] + eps)
    
    # Smooth KL and JSD with a separate EMA
    if t == 0:
        D_kl_smooth[t] = D_kl[t]
        D_jsd_smooth[t] = D_jsd[t]
    else:
        D_kl_smooth[t] = (1 - beta_smooth) * D_kl_smooth[t-1] + beta_smooth * D_kl[t]
        D_jsd_smooth[t] = (1 - beta_smooth) * D_jsd_smooth[t-1] + beta_smooth * D_jsd[t]

    # Serotonin-like leaky integrator update
    # Drive = some signal we want to integrate, e.g. Surprise
    drive_t = S_surprise[t]
    if t == 0:
        S_5ht[t] = drive_t
    else:
        S_5ht[t] = (1 - gamma) * S_5ht[t-1] + gamma * drive_t

    # Optional saturation
    if S_5ht[t] > S_max:
        S_5ht[t] = S_max

    current_state = next_state


# Plotting
fig, axs = plt.subplots(4, 2, figsize=(12, 16), sharex=True)

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

# Serotonin-like Signal
axs[3, 0].plot(S_5ht, label='5-HT-like Leaky Integration')
axs[3, 0].axvline(T1, color='k', linestyle='--')
axs[3, 0].set_title('Serotonin-like Signal')
axs[3, 0].grid(True)

# The last subplot (3,1) is left blank (or you could do something else)
axs[3, 1].axis('off')

plt.tight_layout()
plt.show()


#%% plot only kl and jensen with black lines and no grid 

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axs[0].plot(D_kl_smooth, label='KL Divergence', color='k')
axs[0].axvline(T1, color='k', linestyle='--', label='Transition')
axs[0].set_title('KL Divergence')
axs[0].legend()

axs[1].plot(D_jsd_smooth, label='Jensen-Shannon Divergence', color='k')
axs[1].axvline(T1, color='k', linestyle='--')
axs[1].set_title('Jensen-Shannon Divergence')
axs[1].grid(False)
axs[0].grid(False)
axs[0].set_ylabel('Divergence')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Divergence')
axs[1].set_xlabel('Time step')

#show 
plt.show()




# %%
