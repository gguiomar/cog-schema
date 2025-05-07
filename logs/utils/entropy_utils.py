import matplotlib.ticker as mticker
from scipy.stats import entropy
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

sns.set(style="ticks", palette="husl")
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

def plot_entropy_by_round(choices_df, games_df, model, nb_quadrants):
    valid_games = games_df.query(
        "model == @model and nb_quadrants == @nb_quadrants"
    )[["game_id"]]
    df = pd.merge(choices_df, valid_games, on="game_id")
    logits = df[['A_logit','B_logit','C_logit','D_logit']].values
    df['entropy'] = entropy(logits, axis=1)

    summary = df.groupby('round_nb')['entropy'].agg(['mean','count','std'])
    summary['stderr'] = summary['std'] / np.sqrt(summary['count'])
    summary['lower']  = summary['mean'] - 1.96 * summary['stderr']
    summary['upper']  = summary['mean'] + 1.96 * summary['stderr']

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(summary.index, summary['mean'], marker='o', label='Mean Entropy')
    ax.fill_between(summary.index, summary['lower'], summary['upper'], alpha=0.3, label='95% CI')

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.set_title(f'Entropy by Round {model}')
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Entropy (nats)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return summary
