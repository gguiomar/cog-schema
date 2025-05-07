import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_round_duration_by_model(choices_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
    """
    Plots histograms of round durations per model using seaborn's FacetGrid.

    Parameters:
    - choices_df: DataFrame with round-level data including 'game_id' and 'round_time'
    - games_df: DataFrame with game-level data including 'game_id' and 'model'
    """
    # Merge to bring in model info
    merged_df = choices_df.merge(games_df[["game_id", "model"]], on="game_id", how="left")

    # Drop rows without round_time
    merged_df = merged_df.dropna(subset=["round_time"])

    # Plot: one histogram per model
    g = sns.FacetGrid(merged_df, col="model", col_wrap=4, sharex=False, sharey=False, height=4)
    g.map_dataframe(sns.histplot, x="round_time", bins=30, color="crimson", alpha=0.6)

    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Duration (seconds)", "Count")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Distribution of Round Duration per Model", fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_round_duration_per_round(choices_df, games_df, model_name):
    """
    Plot round_time distribution for rounds 1 to 14 for a specific model.
    """
    # Merge choices with model info
    merged_df = choices_df.merge(games_df[["game_id", "model"]], on="game_id")
    
    # Filter by model and non-null round_time
    model_df = merged_df[(merged_df["model"] == model_name) & (merged_df["round_time"].notna())]

    # Create FacetGrid: one histogram per round
    g = sns.FacetGrid(model_df, col="round_nb", col_wrap=7, height=3, sharex=True, sharey=True)
    g.map_dataframe(sns.histplot, x="round_time", bins=30, color="crimson", alpha=0.6)

    g.set_titles("Round {col_name}")
    g.set_axis_labels("Duration (seconds)", "Count")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{model_name} - Distribution of Round Duration per Round", fontsize=14)

    plt.tight_layout()
    plt.show()
