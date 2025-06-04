#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

K = 4
BIAS_TRUE = 0.9
BIAS_FALSE = 0.5
N_TRIALS = 500
ROUNDS = np.arange(2, 13)
rng = np.random.default_rng(42)

class TemporalReasoningEnvironment:
    """Generates cue/colour observations given a hidden target index."""
    def __init__(self, k: int, bias_true: float, bias_false: float, rng: np.random.Generator):
        self.k = k
        self.bias_true = bias_true
        self.bias_false = bias_false
        self.rng = rng

    def start_trial(self) -> int:
        return self.rng.integers(self.k)  # hidden true location

    def sample_round(self, true_h: int):
        cue = self.rng.integers(self.k)
        p_colour = self.bias_true if cue == true_h else self.bias_false
        colour = self.rng.random() < p_colour
        return cue, colour

class DirectBayesAgent:
    """Exact Bayesian updates in probability space."""
    def __init__(self, k: int, bias_true: float, bias_false: float):
        self.k = k
        self.bias_true = bias_true
        self.bias_false = bias_false
        self.posterior = np.full(k, 1.0 / k)

    def update(self, cue: int, colour: bool):
        p_t = self.bias_true  if colour else 1.0 - self.bias_true
        p_f = self.bias_false if colour else 1.0 - self.bias_false
        denom = p_t * self.posterior[cue] + p_f * (1.0 - self.posterior[cue])
        self.posterior[cue] = p_t * self.posterior[cue] / denom
        mask = np.arange(self.k) != cue
        self.posterior[mask] = p_f * self.posterior[mask] / denom

    @property
    def entropy(self):
        p = np.clip(self.posterior, 1e-12, 1.0)
        return -np.sum(p * np.log(p))

class LogSpaceAgent:
    """Numerically‑stable Bayesian updates in log‑space."""
    def __init__(self, k: int, bias_true: float, bias_false: float):
        self.k = k
        self.bias_true = bias_true
        self.bias_false = bias_false
        self.lp = np.full(k, -np.log(k))

    def update(self, cue: int, colour: bool):
        p_t = self.bias_true  if colour else 1.0 - self.bias_true
        p_f = self.bias_false if colour else 1.0 - self.bias_false
        self.lp += np.where(np.arange(self.k) == cue, np.log(p_t), np.log(p_f))
        self.lp -= logsumexp(self.lp)

    @property
    def posterior(self):
        return np.exp(self.lp)

    @property
    def entropy(self):
        return -np.sum(self.posterior * self.lp)

env = TemporalReasoningEnvironment(K, BIAS_TRUE, BIAS_FALSE, rng)

mean_post_log, mean_ent_log = [], []
mean_post_dir, mean_ent_dir = [], []

for n_rounds in ROUNDS:
    posts_log, ents_log = [], []
    posts_dir, ents_dir = [], []

    for _ in range(N_TRIALS):
        true_h = env.start_trial()
        agent_log = LogSpaceAgent(K, env.bias_true, env.bias_false)
        agent_dir = DirectBayesAgent(K, env.bias_true, env.bias_false)

        for _ in range(n_rounds):
            cue, colour = env.sample_round(true_h)
            agent_log.update(cue, colour)
            agent_dir.update(cue, colour)

        posts_log.append(agent_log.posterior[true_h])
        ents_log.append(agent_log.entropy)
        posts_dir.append(agent_dir.posterior[true_h])
        ents_dir.append(agent_dir.entropy)

    mean_post_log.append(np.mean(posts_log))
    mean_ent_log.append(np.mean(ents_log))
    mean_post_dir.append(np.mean(posts_dir))
    mean_ent_dir.append(np.mean(ents_dir))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(ROUNDS, mean_post_log, "o-", label="Log‑space agent")
ax[0].plot(ROUNDS, mean_post_dir, "s-", label="Direct Bayes agent")
ax[0].set(xlabel="Rounds", ylabel="Mean P(true‑H)", title="Posterior accuracy vs. rounds")
ax[0].grid(); ax[0].legend()

ax[1].plot(ROUNDS, mean_ent_log, "o-", label="Log‑space agent")
ax[1].plot(ROUNDS, mean_ent_dir, "s-", label="Direct Bayes agent")
ax[1].set(xlabel="Rounds", ylabel="Mean entropy (nats)", title="Entropy vs. rounds")
ax[1].grid(); ax[1].legend()

plt.tight_layout()
plt.show()

# %%
