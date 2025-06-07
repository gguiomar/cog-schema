#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style and rocket color palette (no grid)
sns.set_style("white")
rocket_colors = sns.color_palette("rocket", 4)

K = 4
P_T = 0.9
P_F = 0.5
N_TRIALS = 100
ROUNDS = np.arange(1, 100)
rng = np.random.default_rng(42)

class TemporalReasoningEnvironment:
   def __init__(self, k: int, p_t: float, p_f: float, rng: np.random.Generator):
       self.k = k
       self.p_t = p_t
       self.p_f = p_f
       self.rng = rng

   def start_trial(self) -> int:
       return self.rng.integers(self.k)

   def sample_round(self, true_z: int):
       cue = self.rng.integers(self.k)
       if cue == true_z:
           p_color_1 = self.p_t
       else:
           p_color_1 = self.p_f
       color = int(self.rng.random() < p_color_1)
       return cue, color

class DirectBayesAgent:
   def __init__(self, k: int, p_t: float, p_f: float):
       self.k = k
       self.p_t = p_t
       self.p_f = p_f
       self.posterior = np.full(k, 1.0 / k)

   def update(self, cue: int, color: int):
       likelihood = np.zeros(self.k)
       for z in range(self.k):
           if z == cue:
               likelihood[z] = self.p_t if color == 1 else (1.0 - self.p_t)
           else:
               likelihood[z] = self.p_f if color == 1 else (1.0 - self.p_f)
       
       self.posterior *= likelihood
       self.posterior /= self.posterior.sum()

   @property
   def entropy(self):
       p = np.clip(self.posterior, 1e-12, 1.0)
       return -np.sum(p * np.log(p))

class RandomPolicyAgent:
   def __init__(self, k: int, rng: np.random.Generator):
       self.k = k
       self.rng = rng
   
   def get_decision(self, learned_posterior):
       return self.rng.integers(self.k)
   
   def is_correct(self, decision, true_z):
       return int(decision == true_z)

class MAPAgent:
   def __init__(self, k: int):
       self.k = k
   
   def get_decision(self, learned_posterior):
       return np.argmax(learned_posterior)
   
   def is_correct(self, decision, true_z):
       return int(decision == true_z)
   
   def get_map_probability(self, learned_posterior):
       return np.max(learned_posterior)

env = TemporalReasoningEnvironment(K, P_T, P_F, rng)

random_accuracy = []
map_accuracy = []
map_probabilities = []
posterior_true_target = []
entropy_values = []

# Add lists to store standard deviations
random_accuracy_std = []
map_accuracy_std = []
map_probabilities_std = []
posterior_true_target_std = []
entropy_values_std = []

for n_rounds in ROUNDS:
   random_correct = []
   map_correct = []
   map_probs = []
   posterior_probs = []
   entropies = []

   for trial in range(N_TRIALS):
       # Start new trial with random true target
       true_z = env.start_trial()
       
       # Initialize agents for this trial
       agent = DirectBayesAgent(K, env.p_t, env.p_f)
       random_agent = RandomPolicyAgent(K, rng)
       map_agent = MAPAgent(K)

       # Run n_rounds of observations and updates
       for round_num in range(n_rounds):
           cue, color = env.sample_round(true_z)
           agent.update(cue, color)

       # After all rounds, get final posterior and make decisions
       learned_posterior = agent.posterior.copy()
       
       # Make final decisions
       random_decision = random_agent.get_decision(learned_posterior)
       map_decision = map_agent.get_decision(learned_posterior)
       
       # Evaluate performance only on final decisions
       random_correct.append(random_agent.is_correct(random_decision, true_z))
       map_correct.append(map_agent.is_correct(map_decision, true_z))
       map_probs.append(map_agent.get_map_probability(learned_posterior))
       
       # Store posterior probability of true target and entropy
       posterior_probs.append(learned_posterior[true_z])
       entropies.append(agent.entropy)

   # Calculate means and standard deviations across trials
   random_accuracy.append(np.mean(random_correct))
   map_accuracy.append(np.mean(map_correct))
   map_probabilities.append(np.mean(map_probs))
   posterior_true_target.append(np.mean(posterior_probs))
   entropy_values.append(np.mean(entropies))
   
   # Calculate and store standard deviations
   random_accuracy_std.append(np.std(random_correct))
   map_accuracy_std.append(np.std(map_correct))
   map_probabilities_std.append(np.std(map_probs))
   posterior_true_target_std.append(np.std(posterior_probs))
   entropy_values_std.append(np.std(entropies))

#%%
# Set larger font sizes
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'legend.fontsize': 12})

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Convert to numpy arrays for easier computation
random_accuracy = np.array(random_accuracy)
map_accuracy = np.array(map_accuracy)
map_probabilities = np.array(map_probabilities)
posterior_true_target = np.array(posterior_true_target)
entropy_values = np.array(entropy_values)
random_accuracy_std = np.array(random_accuracy_std)
map_accuracy_std = np.array(map_accuracy_std)
map_probabilities_std = np.array(map_probabilities_std)
posterior_true_target_std = np.array(posterior_true_target_std)
entropy_values_std = np.array(entropy_values_std)

# Plot 1: Decision Policy Performance
ax[0,0].plot(ROUNDS, random_accuracy, "-", label="Random Policy", color=rocket_colors[0], linewidth=2)
ax[0,0].fill_between(ROUNDS, random_accuracy - random_accuracy_std, random_accuracy + random_accuracy_std, 
                     color=rocket_colors[0], alpha=0.2)
ax[0,0].plot(ROUNDS, map_accuracy, "-", label="MAP Agent", color=rocket_colors[3], linewidth=2)
ax[0,0].fill_between(ROUNDS, map_accuracy - map_accuracy_std, map_accuracy + map_accuracy_std, 
                     color=rocket_colors[3], alpha=0.2)
ax[0,0].axhline(y=1/K, color="gray", linestyle="--", alpha=0.7, label="Chance level")
ax[0,0].set(xlabel="Rounds", ylabel="Decision Accuracy")
ax[0,0].legend()
ax[0,0].set_ylim(0, 1.2)
sns.despine(ax=ax[0,0])

# Plot 2: MAP Probability
ax[0,1].plot(ROUNDS, map_probabilities, "-", label="MAP Probability", color=rocket_colors[2], linewidth=2)
ax[0,1].fill_between(ROUNDS, map_probabilities - map_probabilities_std, map_probabilities + map_probabilities_std, 
                     color=rocket_colors[2], alpha=0.2)
ax[0,1].set(xlabel="Rounds", ylabel="Mean MAP Probability")
ax[0,1].legend()
sns.despine(ax=ax[0,1])

# Plot 3: Posterior Accuracy
ax[1,0].plot(ROUNDS, posterior_true_target, "-", label="P(true target)", color=rocket_colors[1], linewidth=2)
ax[1,0].fill_between(ROUNDS, posterior_true_target - posterior_true_target_std, posterior_true_target + posterior_true_target_std, 
                     color=rocket_colors[1], alpha=0.2)
ax[1,0].set(xlabel="Rounds", ylabel="Mean P(true target)")
ax[1,0].legend()
sns.despine(ax=ax[1,0])

# Plot 4: Entropy
ax[1,1].plot(ROUNDS, entropy_values, "-", label="Entropy", color=rocket_colors[2], linewidth=2)
ax[1,1].fill_between(ROUNDS, entropy_values - entropy_values_std, entropy_values + entropy_values_std, 
                     color=rocket_colors[2], alpha=0.2)
ax[1,1].axhline(y=np.log(K), color="gray", linestyle="--", alpha=0.7, label="Max entropy")
ax[1,1].set(xlabel="Rounds", ylabel="Mean Entropy (nats)")
ax[1,1].legend()
sns.despine(ax=ax[1,1])

plt.tight_layout()
plt.show()
# %%
