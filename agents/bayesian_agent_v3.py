#%%
import numpy as np
import matplotlib.pyplot as plt

K = 4
P_T = 0.9
P_F = 0.5
N_TRIALS = 100
ROUNDS = np.arange(2, 13)
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

for n_rounds in ROUNDS:
   random_correct = []
   map_correct = []
   map_probs = []
   posterior_probs = []
   entropies = []

   for _ in range(N_TRIALS):
       true_z = env.start_trial()
       
       bayesian_learner = DirectBayesAgent(K, env.p_t, env.p_f)
       random_agent = RandomPolicyAgent(K, rng)
       map_agent = MAPAgent(K)

       for _ in range(n_rounds):
           cue, color = env.sample_round(true_z)
           bayesian_learner.update(cue, color)

       learned_posterior = bayesian_learner.posterior.copy()
       
       random_decision = random_agent.get_decision(learned_posterior)
       map_decision = map_agent.get_decision(learned_posterior)
       
       random_correct.append(random_agent.is_correct(random_decision, true_z))
       map_correct.append(map_agent.is_correct(map_decision, true_z))
       map_probs.append(map_agent.get_map_probability(learned_posterior))
       
       posterior_probs.append(learned_posterior[true_z])
       entropies.append(bayesian_learner.entropy)

   random_accuracy.append(np.mean(random_correct))
   map_accuracy.append(np.mean(map_correct))
   map_probabilities.append(np.mean(map_probs))
   posterior_true_target.append(np.mean(posterior_probs))
   entropy_values.append(np.mean(entropies))

#%%
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0,0].plot(ROUNDS, random_accuracy, "s-", label="Random Policy", color="gray", linewidth=2)
ax[0,0].plot(ROUNDS, map_accuracy, "o-", label="MAP Agent", color="red", linewidth=2)
ax[0,0].axhline(y=1/K, color="gray", linestyle="--", alpha=0.7, label="Chance level")
ax[0,0].set(xlabel="Rounds", ylabel="Decision Accuracy", title="Decision Policy Performance")
ax[0,0].legend()
ax[0,0].set_ylim(0, 1)
ax[0,0].tick_params(axis='both', which='major', labelsize=12)
ax[0,0].spines['top'].set_visible(False)
ax[0,0].spines['right'].set_visible(False)

ax[0,1].plot(ROUNDS, map_probabilities, "o-", label="MAP Probability", color="orange", linewidth=2)
ax[0,1].set(xlabel="Rounds", ylabel="Mean MAP Probability", title="Confidence in MAP Decision")
ax[0,1].legend()
ax[0,1].tick_params(axis='both', which='major', labelsize=12)
ax[0,1].spines['top'].set_visible(False)
ax[0,1].spines['right'].set_visible(False)

ax[1,0].plot(ROUNDS, posterior_true_target, "^-", label="P(true target)", color="blue", linewidth=2)
ax[1,0].set(xlabel="Rounds", ylabel="Mean P(true target)", title="Posterior Accuracy")
ax[1,0].legend()
ax[1,0].tick_params(axis='both', which='major', labelsize=12)
ax[1,0].spines['top'].set_visible(False)
ax[1,0].spines['right'].set_visible(False)

ax[1,1].plot(ROUNDS, entropy_values, "v-", label="Entropy", color="green", linewidth=2)
ax[1,1].axhline(y=np.log(K), color="green", linestyle="--", alpha=0.7, label="Max entropy")
ax[1,1].set(xlabel="Rounds", ylabel="Mean Entropy (nats)", title="Posterior Uncertainty")
ax[1,1].legend()
ax[1,1].tick_params(axis='both', which='major', labelsize=12)
ax[1,1].spines['top'].set_visible(False)
ax[1,1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
# %%
