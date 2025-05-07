from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_accuracy_by_round_per_model(choices_df: pd.DataFrame,
                                     games_df: pd.DataFrame,
                                     models: List[str]) -> None:
    """
    For each model in `models`, compute and plot:
      • x = round number
      • y = proportion of choices matching that game's biased_quadrant
      • text “n=…” above each point showing how many games reached that round
    """
    # First merge in model and the numeric biased_quadrant
    df = choices_df.merge(
        games_df[['game_id','model','biased_quadrant']],
        on='game_id', how='left'
    )

    # Map letters ↔ numbers so we can compare correctly
    letter_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    df['chosen_q_num'] = df['chosen_quadrant'].map(letter_to_num)

    df['is_correct'] = df['chosen_q_num'] == df['biased_quadrant']
    
    for model in models:
        sub = df[df['model'] == model]
        stats = (
            sub.groupby('round_nb')
               .agg(
                   accuracy=('is_correct','mean'),
                   n=('game_id','nunique')
               )
               .reset_index()
        )
        
        plt.figure(figsize=(8,5))
        plt.plot(stats['round_nb'], stats['accuracy'], marker='o')
        for _, row in stats.iterrows():
            plt.text(
                row['round_nb'],
                row['accuracy'] + 0.02,
                f"n={int(row['n'])}",
                ha='center',
                fontsize=9
            )
        
        plt.ylim(0,1)
        plt.xlim(stats['round_nb'].min() - 0.5,
                 stats['round_nb'].max() + 0.5)
        plt.xlabel("Round")
        plt.ylabel("Proportion of Choices Matching Biased Quadrant")
        plt.title(f"Biased-Choice Accuracy by Round — {model}")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

def plot_strategy_vs_exploration_llm(choices_df, games_df, model_name):
    """
    For a single model, plots two curves:
      • mean reaction‐time for “strategy” moves vs “exploration” moves by round_nb
      • twin‐axis bars showing counts of strategy vs exploration
    """
    # 1) grab only this model
    df = (
        choices_df
        .merge(games_df[["game_id","model"]], on="game_id", how="left")
        .query("model == @model_name")
        .dropna(subset=["round_time"])
        .copy()
    )
    # 2) tag strategy (same as before) but using round_nb
    df = df.sort_values(["game_id","round_nb"])
    df["prev1"] = df.groupby("game_id")["chosen_quadrant"].shift(1)
    df["prev2"] = df.groupby("game_id")["chosen_quadrant"].shift(2)
    df["is_strategy"] = (
        (df["chosen_quadrant"] == df["prev1"]) &
        (df["chosen_quadrant"] == df["prev2"])
    )
    
    # 3) aggregate means & counts by round_nb & is_strategy
    by = (
        df
        .groupby(["round_nb","is_strategy"])["round_time"]
        .agg(["mean","count"])
        .reset_index()
    )
    strat = by[by["is_strategy"]]
    explo = by[~by["is_strategy"]]
    
    # 4) plot
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(strat["round_nb"], strat["mean"], marker="o", label="Strategy")
    ax1.plot(explo["round_nb"], explo["mean"], marker="o", label="Exploration")
    ax1.set(xlabel="Round", ylabel="Reaction Time (s)")
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    w = 0.4
    ax2.bar(strat["round_nb"]-w/2, strat["count"], width=w, alpha=0.3, label="Strategy (n)")
    ax2.bar(explo["round_nb"]+w/2, explo["count"], width=w, alpha=0.3, label="Exploration (n)")
    ax2.set_ylabel("Observations")
    
    # combine legends
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")
    plt.title(f"Strategy vs Exploration – {model_name}")
    plt.tight_layout()
    plt.show()

def add_final_round_time(games_df: pd.DataFrame, choices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'final_round_time' column to the games_df.
    It is computed as: trial_time - sum of all round_times for that game_id.
    """
    # First, sum all round_times per game
    round_time_sum = (
        choices_df
        .dropna(subset=["round_time"])  # Only keep rounds with a valid time
        .groupby("game_id")["round_time"]
        .sum()
    )

    # Then map it into games_df
    games_df = games_df.copy()
    games_df["sum_round_time"] = games_df["game_id"].map(round_time_sum)

    # Calculate final_round_time
    games_df["final_round_time"] = games_df["trial_time"] - games_df["sum_round_time"]

    # Optional: drop the helper column if you don't want to keep it
    games_df = games_df.drop(columns=["sum_round_time"])

    return games_df

def plot_final_reflection_vs_num_rounds(games_df: pd.DataFrame) -> None:
    """
    Plots mean final choice reflection time as a function of the number of rounds.
    Assumes 'final_round_time' has already been computed and added to games_df.
    """
    # Drop rows with missing or invalid reflection time
    df = games_df.dropna(subset=["final_round_time"])

    # Group by number of rounds
    agg = (
        df.groupby("nb_rounds")["final_round_time"]
        .mean()
        .reset_index()
        .rename(columns={"final_round_time": "mean_final_reflection"})
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        agg["nb_rounds"],
        agg["mean_final_reflection"],
        marker="o",
        color="lightcoral",
        linewidth=2
    )
    plt.xlabel("Number of Rounds")
    plt.ylabel("Mean Final Choice Reflection Time (seconds)")
    plt.title("Mean Final Choice Reflexion Time vs. Number of Rounds")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_mean_reflection_by_correctness(
    games_df: pd.DataFrame,
) -> None:
    """
    Bar‐plot of average final‐choice reflection time for correct vs. incorrect games.
    Displays the value of the mean reflection time and the number of games (n).
    """
    # keep only valid reflection times
    df = games_df.loc[
        games_df["final_round_time"].notnull() 
    ].copy()

    # count how many games in each category
    counts = df.groupby("success")["game_id"].nunique()
    counts.index = counts.index.map({True: "Correct", False: "Incorrect"})

    # compute mean reflection times
    agg = (
        df.groupby("success")["final_round_time"]
          .mean()
          .reset_index()
    )
    agg["label"] = agg["success"].map({True: "Correct", False: "Incorrect"})
    agg["mean_reflection_time"] = agg["final_round_time"]

    # plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        agg["label"],
        agg["mean_reflection_time"],
        alpha=0.7,
        color='lightcoral',
        edgecolor='black'
    )
    plt.xlabel("Outcome")
    plt.ylabel("Mean Final-Choice Reflection Time (s)")
    plt.title("Average Reflection Time by Correctness")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # annotate value and n
    for bar, (_, row) in zip(bars, agg.iterrows()):
        height = bar.get_height()
        label = row["label"]
        # reflection time (not rounded)
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height - 0.01,
            f"{row['mean_reflection_time']:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
        # number of observations (n=)
        plt.text(
            bar.get_x() + bar.get_width()/2,
            0.01,
            f"n={counts[label]}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray"
        )

    plt.tight_layout()
    plt.show()



def compute_last3_consistency(games_df: pd.DataFrame, choices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column to `games_df` called 'last3_consistent' indicating whether the last 3
    choices in a game were consistent (i.e., same chosen_quadrant).
    """
    last_choices = (
        choices_df
        .sort_values(['game_id', 'round_nb'])
        .groupby('game_id')
        .tail(3)  # last 3 rounds
    )

    def is_consistent(group):
        return group['chosen_quadrant'].nunique() == 1

    consistency = last_choices.groupby('game_id').apply(is_consistent).rename("last3_consistent")
    return games_df.merge(consistency, on="game_id", how="left")

def plot_reflexion_vs_consistency(games_df: pd.DataFrame) -> None:
    """
    Plot mean final choice reflection time by last-three-rounds consistency.
    Requires `last3_consistent` and `final_choice_reflexion_time` to be in `games_df`.
    """
    valid = games_df.loc[
        games_df["final_round_time"].notnull() &
        games_df["last3_consistent"].notnull()
    ]

    if valid.empty:
        print("No valid data with final_round_time and consistency info.")
        return

    grouped_consistency = (
        valid.groupby("last3_consistent")["final_round_time"]
        .mean()
        .reset_index()
    )

    grouped_consistency["Consistency"] = grouped_consistency["last3_consistent"].map({
        True: "Consistent", False: "Non-Consistent"
    })

    grouped_consistency = grouped_consistency.sort_values("Consistency")

    plt.figure(figsize=(8, 5))
    plt.plot(
        grouped_consistency["Consistency"],
        grouped_consistency["final_round_time"],
        marker='o', linestyle='-', color='indianred'
    )
    plt.xlabel("Last Three Rounds Consistency")
    plt.ylabel("Mean Final Choice Reflection Time (s)")
    plt.title("Mean Final Choice Reflection Time by Consistency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_success_rate_by_quadrant(
    games_df: pd.DataFrame,
    models: list
) -> None:
    """
    For the given list of models, compute and display success rates separately
    for 2-quadrant games and 4-quadrant games.
    Display results in two tables next to each other, with bigger font and more row spacing.
    """

    # Keep only relevant models
    df = games_df[games_df['model'].isin(models)].copy()

    # Separate 2q and 4q
    df_2q = df[df['nb_quadrants'] == 2]
    df_4q = df[df['nb_quadrants'] == 4]

    # Compute success rate
    table_2q = (
        df_2q.groupby('model')['success']
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'success': 'Success Rate (%)'})
    )

    table_4q = (
        df_4q.groupby('model')['success']
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'success': 'Success Rate (%)'})
    )

    # Plot both tables side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))  # slightly taller figure

    axes[0].axis('off')
    table_left = axes[0].table(
        cellText=table_2q.values,
        colLabels=table_2q.columns,
        loc='center',
        cellLoc='center'
    )
    table_left.auto_set_font_size(False)
    table_left.set_fontsize(12)
    table_left.scale(1.2, 1.5)  # ⬅️ wider x and TALLER y

    axes[0].set_title("2-Quadrant Games", fontsize=16, pad=2)

    axes[1].axis('off')
    table_right = axes[1].table(
        cellText=table_4q.values,
        colLabels=table_4q.columns,
        loc='center',
        cellLoc='center'
    )
    table_right.auto_set_font_size(False)
    table_right.set_fontsize(12)
    table_right.scale(1.2, 1.5)  # ⬅️ same here

    axes[1].set_title("4-Quadrant Games", fontsize=16, pad=2)

    plt.tight_layout()
    plt.show()

    return table_2q, table_4q

