import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

def plot_rt_by_cues_round_per_model(
    choices_df: pd.DataFrame,
    games_df: pd.DataFrame,
    models: Optional[List[str]] = None
) -> None:
    """
    For each model, plot mean round_time by round, separately for 2-q and 4-q games.
    In the 2-q panel: two lines (1 vs. 2 available cues).
    In the 4-q panel: four lines (1,2,3,4 available cues).
    Invalid rounds (color=None) are excluded.
    """
    # merge in model label and #quadrants
    df = (
        choices_df
        .merge(games_df[['game_id','model','nb_quadrants']],
               on='game_id', how='left')
        .query("color.notna()")                      # drop invalid
        .assign(n_cues=lambda d: d['available_cues'].str.len())
    )
    
    if models is None:
        models = df['model'].unique().tolist()

    for model in models:
        sub = df[df['model'] == model]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        
        def _plot_panel(ax, nbq, cue_range):
            data = sub[sub['nb_quadrants'] == nbq]
            if data.empty:
                ax.set_title(f"{nbq} Quadrants\n(no data)")
                return

            grp = (
                data
                .groupby(['round_nb','n_cues'])['round_time']
                .mean()
                .unstack('n_cues')
            )
            for n in cue_range:
                if n in grp.columns:
                    ax.plot(grp.index, grp[n],
                            marker='o', label=f"{n} cue{'s' if n>1 else ''}")
            ax.set_title(f"{nbq} Quadrants")
            ax.set_xlabel("Round")
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_xticks(sorted(data['round_nb'].unique()))
            if nbq == 2:
                ax.set_ylabel("RT (s)")
            ax.legend(title="Avail. cues")
        
        _plot_panel(axes[0], nbq=2, cue_range=[1,2])
        _plot_panel(axes[1], nbq=4, cue_range=[1,2,3,4])
        
        fig.suptitle(f"{model} — RT by Round & # Available Cues", y=1.02)
        plt.tight_layout()
        plt.show()

