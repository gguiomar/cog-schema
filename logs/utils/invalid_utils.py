import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_invalid_choice_proportions(choices_df: pd.DataFrame,
                                    games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and plot the proportion of rounds where color is None
    (an invalid choice) for each model.

    Returns the summary DataFrame with columns:
      - none_count: number of rounds with color=None
      - total_rounds: total number of rounds
      - none_prop: proportion of invalid rounds
    """
    # merge in model labels and flag invalid rounds
    df = (
        choices_df
        .merge(games_df[['game_id', 'model']], on='game_id', how='left')
        .assign(is_none=lambda d: d['color'].isna())
    )

    # aggregate per model
    summary = (
        df
        .groupby('model')
        .agg(
            none_count=('is_none', 'sum'),
            total_rounds=('is_none', 'size')
        )
        .assign(none_prop=lambda d: d['none_count'] / d['total_rounds'])
        .sort_values('none_prop', ascending=False)
    )

    # print table
    print("\nInvalid (None) choices by model:")
    print(summary)

    # plot
    plt.figure(figsize=(8, 6))
    sns.barplot(
        y=summary.index,
        x=summary['none_prop'],
        color="lightcoral",
        edgecolor="k"
    )
    plt.xlabel("Proportion of Rounds with color=None")
    plt.ylabel("Model")
    plt.title("Proportion of Invalid Choices per Model")
    plt.xlim(0, summary['none_prop'].max() * 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return summary

def plot_invalid_choice_proportions_by_round(choices_df: pd.DataFrame,
                                              games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and plot the proportion of rounds where color is None
    (an invalid choice) for each model, broken down by round number.

    Returns the round_summary DataFrame with columns:
      - model
      - round_nb
      - none_count
      - total_count
      - none_prop
    """
    # 1) Merge in model name and flag invalid rounds
    df = (
        choices_df
        .merge(games_df[['game_id', 'model']], on='game_id', how='left')
        .assign(is_none=lambda d: d['color'].isna())
    )

    # 2) Compute per-model × per-round summary
    round_summary = (
        df
        .groupby(['model', 'round_nb'])
        .agg(
            none_count=('is_none', 'sum'),
            total_count=('is_none', 'size')
        )
        .reset_index()
    )
    round_summary['none_prop'] = round_summary['none_count'] / round_summary['total_count']

    # 3) Plot one line per model
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=round_summary,
        x='round_nb',
        y='none_prop',
        hue='model',
        marker='o'
    )

    plt.xlabel("Round Number")
    plt.ylabel("Proportion of Invalid (None) Choices")
    plt.title("Proportion of None Choices per Round by Model")
    plt.xticks(sorted(df['round_nb'].unique()))
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return round_summary

def plot_invalid_choices_by_quadrants(choices_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
    """
    Plots the proportion of invalid (None) choices depending on the number of quadrants (2 or 4).

    Parameters:
    - choices_df: DataFrame with round-level data.
    - games_df: DataFrame with game-level data.
    """
    # 1) Merge to get nb_quadrants and flag invalid rounds
    df = (
        choices_df
        .merge(games_df[['game_id', 'nb_quadrants']], on='game_id', how='left')
        .assign(is_none=lambda d: d['color'].isna())
    )

    # 2) Compute per-quadrant summary
    quad_summary = (
        df
        .groupby('nb_quadrants')
        .agg(
            none_count=('is_none', 'sum'),
            total_count=('is_none', 'size')
        )
        .reset_index()
    )
    quad_summary['none_prop'] = quad_summary['none_count'] / quad_summary['total_count']

    print("\nInvalid (None) choices by number of quadrants:")
    print(quad_summary)

    # 3) Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=quad_summary,
        x='nb_quadrants',
        y='none_prop',
        color='salmon'  # fixed single color
    )
    plt.ylim(0, quad_summary['none_prop'].max() * 1.1)
    plt.xlabel("Number of Quadrants")
    plt.ylabel("Proportion of Invalid (None) Choices")
    plt.title("Invalid Choice Rate by # of Quadrants")

    # Annotate percentages above bars
    for idx, row in quad_summary.iterrows():
        plt.text(
            x=idx,
            y=row['none_prop'] + 0.005,
            s=f"{row['none_prop']:.1%}",
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.show()