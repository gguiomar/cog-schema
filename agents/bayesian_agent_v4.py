# %%
import numpy as np, matplotlib.pyplot as plt

# ----- hyper-parameters ----------------------------------------------------
K, P_T, P_F = 4, 0.9, 0.5           # locations, hit-rate, false-alarm
N_TRIALS, ROUNDS = 100, np.arange(2, 13)
rng = np.random.default_rng(42)


class TemporalReasoningEnvironment:
    def __init__(self, k, p_t, p_f, rng):
        self.k, self.p_t, self.p_f, self.rng = k, p_t, p_f, rng

    def start_trial(self):
        return self.rng.integers(self.k)

    def sample_round(self, true_z):
        cue = self.rng.integers(self.k)
        p1  = self.p_t if cue == true_z else self.p_f
        colour = int(self.rng.random() < p1)
        return cue, colour


class BayesAgent:
    """Exact Bayesian updater in probability space; returns surprise metrics."""
    def __init__(self, k, p_t, p_f):
        self.k, self.p_t, self.p_f = k, p_t, p_f
        self.posterior = np.full(k, 1.0 / k)

    def update(self, cue, colour):
        prior = self.posterior.copy()
        like  = np.where(np.arange(self.k) == cue,
                         self.p_t if colour else 1 - self.p_t,
                         self.p_f if colour else 1 - self.p_f)
        denom = np.dot(like, prior)                     # predictive prob
        self.posterior = like * prior / denom           # Bayes rule
        S = -np.log(denom)                              # Shannon surprise
        B = np.sum(self.posterior * np.log((self.posterior + 1e-12)
                                           / (prior + 1e-12)))  # KL surprise
        return S, B

    @property
    def entropy(self):
        p = np.clip(self.posterior, 1e-12, 1)
        return -np.sum(p * np.log(p))


class RandomPolicyAgent:
    def __init__(self, k, rng): self.k, self.rng = k, rng
    def get_decision(self, _): return self.rng.integers(self.k)
    def is_correct(self, d, z): return int(d == z)


class MAPAgent:
    def __init__(self, k): self.k = k
    def get_decision(self, post): return int(np.argmax(post))
    def is_correct(self, d, z):    return int(d == z)
    def get_map_prob(self, post):  return float(np.max(post))


env = TemporalReasoningEnvironment(K, P_T, P_F, rng)

acc_rand, acc_map, conf_map = [], [], []
p_true, ent, s_sur, b_sur  = [], [], [], []

for n in ROUNDS:
    r_ok, m_ok, m_p  = [], [], []
    p_tt, h_vals, s_vals, b_vals = [], [], [], []

    for _ in range(N_TRIALS):
        true_z = env.start_trial()
        learner = BayesAgent(K, P_T, P_F)
        r_pol   = RandomPolicyAgent(K, rng)
        m_pol   = MAPAgent(K)

        for _ in range(n):
            cue, col = env.sample_round(true_z)
            S, B     = learner.update(cue, col)
            s_vals.append(S); b_vals.append(B)

        post = learner.posterior
        r_dec = r_pol.get_decision(post)
        m_dec = m_pol.get_decision(post)

        r_ok.append(r_pol.is_correct(r_dec, true_z))
        m_ok.append(m_pol.is_correct(m_dec, true_z))
        m_p.append(m_pol.get_map_prob(post))

        p_tt.append(post[true_z])
        h_vals.append(learner.entropy)

    acc_rand.append(np.mean(r_ok))
    acc_map.append(np.mean(m_ok))
    conf_map.append(np.mean(m_p))
    p_true.append(np.mean(p_tt))
    ent.append(np.mean(h_vals))
    s_sur.append(np.mean(s_vals))
    b_sur.append(np.mean(b_vals))


# ----- plotting ------------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(11, 12))

ax[0, 0].plot(ROUNDS, acc_rand, "s-",  lw=2, label="Random")
ax[0, 0].plot(ROUNDS, acc_map,  "o-",  lw=2, label="MAP")
ax[0, 0].axhline(1 / K, ls="--", color="gray", alpha=.7)
ax[0, 0].set(title="Decision accuracy", xlabel="Rounds", ylabel="Accuracy")
ax[0, 0].legend(); ax[0, 0].set_ylim(0, 1)

ax[0, 1].plot(ROUNDS, conf_map, "o-", lw=2, color="orange")
ax[0, 1].set(title="MAP confidence", xlabel="Rounds", ylabel="Mean max P(z)")

ax[1, 0].plot(ROUNDS, p_true, "^-", lw=2, color="steelblue")
ax[1, 0].set(title="Posterior on true target", xlabel="Rounds", ylabel="Mean P(true z)")

ax[1, 1].plot(ROUNDS, ent, "v-", lw=2, color="green")
ax[1, 1].axhline(np.log(K), ls="--", color="green", alpha=.7)
ax[1, 1].set(title="Entropy", xlabel="Rounds", ylabel="nats")

ax[2, 0].plot(ROUNDS, s_sur, "o-", lw=2, color="purple")
ax[2, 0].set(title="Shannon surprise $S_t$", xlabel="Rounds", ylabel="Mean S")

ax[2, 1].plot(ROUNDS, b_sur, "s-", lw=2, color="brown")
ax[2, 1].set(title="Bayesian surprise $B_t$", xlabel="Rounds", ylabel="Mean B")

for a in ax.ravel():
    a.grid(True); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.tick_params(labelsize=10)

plt.tight_layout(); plt.show()
# %%
