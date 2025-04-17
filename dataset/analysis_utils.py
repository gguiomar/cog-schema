"""Utility functions to streamline the notebook analysis.

Import this module in your notebook:

    from analysis_utils import *

and replace verbose, repetitive cells with concise function calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants – tweak from a single place instead of repeating literals
# ---------------------------------------------------------------------------
MAX_THINK_TIME = 5.0          # seconds (per‑round thinking time)
MAX_REFLEXION_TIME = 10.0     # seconds (gap before final choice)
LOW_DURATION_PERC = 0.05      # 5th percentile
HIGH_DURATION_PERC = 0.95     # 95th percentile

# Global plotting style (applied once) --------------------------------------
sns.set(style="ticks", palette="husl")
plt.rcParams.update({"figure.figsize": (10, 6)})

# ---------------------------------------------------------------------------
# Data‑loading helpers
# ---------------------------------------------------------------------------

def _read_single_log(path: Path) -> Tuple[dict | None, List[dict]]:
    """Parse one JSON log file. Returns (game_record, round_records).
    If the log is incomplete the first element is None.
    """
    with path.open() as fh:
        data = json.load(fh)

    must_have = {"game_id", "start_time", "completion_time", "total_duration", "success", "choices"}
    if not must_have.issubset(data):
        return None, []

    final_choice = data.get("final_choice") or next(
        (c for c in data["choices"] if c.get("type") == "final_choice"), None
    )
    if final_choice is None:
        return None, []
    
    if not data.get("choices"):
        return None, []

    game = {
        "game_id": data["game_id"],
        "start_time": pd.to_datetime(data["start_time"], utc=True),
        "completion_time": pd.to_datetime(data["completion_time"], utc=True),
        "total_duration": data["total_duration"],
        "success": data["success"],
        "chosen_quadrant": final_choice.get("chosen_quadrant"),
        "correct": final_choice.get("correct"),
        "score": final_choice.get("score"),
        "biased_quadrant": final_choice.get("biased_quadrant"),
        "source_file": path.name,
    }

    rounds: List[dict] = []
    for ch in data["choices"]:
        if ch.get("type") == "final_choice":
            continue
        rounds.append({
            "game_id": data["game_id"],
            "round": ch.get("round"),
            "quadrant": ch.get("quadrant"),
            "chosen_quadrant": ch.get("cue_name") if "cue_name" in ch else ch.get("choice"),
            "color": ch.get("color"),
            "timestamp": pd.to_datetime(ch.get("timestamp"), utc=True),
        })

    return game, rounds


def load_data(logs_dir: os.PathLike | str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read every *.json under logs_dir and build games_df and choices_df."""
    games, choices = [], []
    for file in Path(logs_dir).glob("*.json"):
        g, c = _read_single_log(file)
        if g is not None:
            games.append(g)
            choices.extend(c)

    games_df = pd.DataFrame(games)
    choices_df = pd.DataFrame(choices)

    if "round" in choices_df.columns:
        mins = choices_df.groupby("game_id")["round"].transform("min")
        choices_df.loc[mins == 0, "round"] = choices_df.loc[mins == 0, "round"] + 1
        
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    games_df["chosen_quadrant"] = games_df["chosen_quadrant"].map(mapping)
    games_df["biased_quadrant"] = games_df["biased_quadrant"].map(mapping)

    choices_df = choices_df.sort_values(["game_id", "round"])
    start_map = games_df.set_index("game_id")["start_time"]
    choices_df["time_taken"] = choices_df.groupby("game_id")["timestamp"].diff()
    first = choices_df["round"] == 1
    choices_df.loc[first, "time_taken"] = (
        choices_df.loc[first, "timestamp"] - choices_df.loc[first, "game_id"].map(start_map)
    )
    choices_df["time_taken"] = choices_df["time_taken"].dt.total_seconds()

    return games_df, choices_df

# ---------------------------------------------------------------------------
# Common filters & data‑enrichment routines
# ---------------------------------------------------------------------------

def filter_games_by_duration(games_df: pd.DataFrame,
                             low: float = LOW_DURATION_PERC,
                             high: float = HIGH_DURATION_PERC) -> pd.DataFrame:
    lo, hi = games_df["total_duration"].quantile([low, high])
    return games_df.query("@lo <= total_duration <= @hi").copy()


def add_final_choice_metrics(games_df: pd.DataFrame, choices_df: pd.DataFrame) -> pd.DataFrame:
    g = games_df.copy()
    last_round_ts = choices_df.groupby("game_id")["timestamp"].max()
    g["last_round_ts"] = g["game_id"].map(last_round_ts)
    g["final_choice_reflexion_time"] = (
        g["completion_time"] - g["last_round_ts"]
    ).dt.total_seconds()
    
    num_rounds = choices_df.groupby("game_id")["round"].nunique()
    g["num_rounds"] = g["game_id"].map(num_rounds)
    def _cons(df: pd.DataFrame) -> bool | np.nan:
        last3 = df.sort_values("round").tail(3)
        if len(last3) < 3:
            return np.nan
        return last3["chosen_quadrant"].nunique() == 1
    cons = choices_df.groupby("game_id").apply(_cons)
    g["last3_consistent"] = g["game_id"].map(cons)
    return g

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_game_duration_distribution(games_df: pd.DataFrame, filtered: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(data=games_df, x="total_duration", bins=30, ax=ax1)
    ax1.set(title="Distribution of Game Duration", xlabel="Duration (s)", ylabel="Count")
    sns.boxplot(data=games_df, y="total_duration", ax=ax2)
    ax2.set(title="Box Plot", ylabel="Duration (s)")
    plt.tight_layout()
    if filtered:
        f = filter_games_by_duration(games_df)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(data=f, x="total_duration", bins=30, ax=ax1)
        ax1.set(title="Duration (5–95th pct)", xlabel="Duration (s)")
        sns.boxplot(data=f, y="total_duration", ax=ax2)
        ax2.set(title="Box Plot (5–95th pct)")
        plt.tight_layout()

def evolution_by_round(choices_df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    valid = choices_df.dropna(subset=["time_taken"]).query("time_taken <= @MAX_THINK_TIME")
    agg = (
        valid.groupby("round")
        .agg(mean_thinking_time=("time_taken", "mean"), num_games=("game_id", "nunique"))
        .reset_index()
        .query("num_games > @min_games")
    )
    return agg


def plot_evolution_thinking_time(evo_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots()
    ax1.plot(evo_df["round"], evo_df["mean_thinking_time"], marker="o", label="Mean Thinking Time")
    ax1.set(xlabel="Round", ylabel="Thinking Time (s)")
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.bar(evo_df["round"], evo_df["num_games"], alpha=0.3, label="# Games")
    ax2.set_ylabel("Number of Games")
    ax1.legend(loc="upper left")
    plt.title("Evolution of Thinking Time by Round")
    plt.tight_layout()
    plt.show()


def tag_strategy_moves(choices_df: pd.DataFrame) -> pd.DataFrame:
    df = choices_df.sort_values(["game_id", "round"]).copy()
    df["prev1"] = df.groupby("game_id")["chosen_quadrant"].shift(1)
    df["prev2"] = df.groupby("game_id")["chosen_quadrant"].shift(2)
    df["is_strategy"] = (df["chosen_quadrant"] == df["prev1"]) & (df["chosen_quadrant"] == df["prev2"])
    return df.dropna(subset=["prev1","prev2"]).query("time_taken <= @MAX_THINK_TIME")


def plot_strategy_vs_exploration(df: pd.DataFrame) -> None:
    by = df.groupby(["round", "is_strategy"])["time_taken"].agg(["mean", "count"]).reset_index()
    strat = by[by["is_strategy"]]
    explo = by[~by["is_strategy"]]
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(strat["round"], strat["mean"], marker="o", label="Strategy")
    ax1.plot(explo["round"], explo["mean"], marker="o", label="Exploration")
    ax1.set(xlabel="Round", ylabel="Reaction Time (s)")
    ax1.grid(True)
    ax2 = ax1.twinx()
    width = 0.4
    ax2.bar(strat["round"]-width/2, strat["count"], width=width, alpha=0.3, label="Strategy (n)")
    ax2.bar(explo["round"]+width/2, explo["count"], width=width, alpha=0.3, label="Exploration (n)")
    ax2.set_ylabel("Observations")
    l1,l2 = ax1.get_legend_handles_labels()
    l3,l4 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l3, l2+l4, loc="upper right")
    plt.title("Strategy vs Exploration - Reaction Time by Round")
    plt.tight_layout()
    plt.show()


def plot_reflexion_vs_rounds(games_df: pd.DataFrame) -> None:
    """Plot mean final choice reflexion time vs. number of rounds (≤ MAX_REFLEXION_TIME only)."""
    filt = games_df.loc[
        games_df["final_choice_reflexion_time"].notnull() &
        (games_df["final_choice_reflexion_time"] <= MAX_REFLEXION_TIME)
    ]
    if filt.empty:
        print(f"No games with final_choice_reflexion_time ≤ {MAX_REFLEXION_TIME}s.")
        return
    rounds_vs_reflexion = (
        filt.groupby("num_rounds")["final_choice_reflexion_time"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    plt.plot(
        rounds_vs_reflexion["num_rounds"],
        rounds_vs_reflexion["final_choice_reflexion_time"],
        marker='o'
    )
    plt.xlabel("Number of Rounds")
    plt.ylabel("Mean Final Choice Reflexion Time (seconds)")
    plt.title(f"Mean Final Choice Reflexion Time vs. Number of Rounds (≤ {MAX_REFLEXION_TIME}s only)")
    plt.grid(True)
    plt.show()


def plot_reflexion_vs_consistency(games_df: pd.DataFrame) -> None:
    """Plot mean final choice reflexion time by last-three-rounds consistency (≤ MAX_REFLEXION_TIME only)."""
    valid = games_df.loc[
        games_df["final_choice_reflexion_time"].notnull() &
        games_df["last3_consistent"].notnull() &
        (games_df["final_choice_reflexion_time"] <= MAX_REFLEXION_TIME)
    ]
    if valid.empty:
        print(f"No games with final_choice_reflexion_time ≤ {MAX_REFLEXION_TIME}s and consistency info.")
        return
    grouped_consistency = (
        valid.groupby("last3_consistent")["final_choice_reflexion_time"]
        .mean()
        .reset_index()
    )
    grouped_consistency["Consistency"] = grouped_consistency["last3_consistent"].map({
        True: "Consistent", False: "Non-Consistent"
    })
    grouped_consistency = grouped_consistency.sort_values("Consistency")
    plt.figure(figsize=(10, 6))
    plt.plot(
        grouped_consistency["Consistency"],
        grouped_consistency["final_choice_reflexion_time"],
        marker='o', linestyle='-'
    )
    plt.xlabel("Last Three Rounds Consistency")
    plt.ylabel("Mean Final Choice Reflexion Time (seconds)")
    plt.title(f"Mean Final Choice Reflexion Time by Consistency (≤ {MAX_REFLEXION_TIME}s only)")
    plt.grid(True)
    plt.show()

def plot_reflexion_by_round_and_correctness(games_df: pd.DataFrame) -> None:
    """Mean final-choice reflexion time (by round & correctness) with #games per round."""
    # filter & agg mean time
    df = games_df.loc[
        games_df["final_choice_reflexion_time"].notnull() &
        (games_df["final_choice_reflexion_time"] <= MAX_REFLEXION_TIME)
    ]
    time_agg = (
        df.groupby(["num_rounds", "correct"])["final_choice_reflexion_time"]
          .mean()
          .reset_index()
    )
    # count total games per round (regardless of correctness)
    count_agg = (
        df.groupby("num_rounds")["game_id"]
          .nunique()
          .reset_index(name="num_games")
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # left axis: mean reflexion time, split by correctness
    sns.lineplot(
        data=time_agg,
        x="num_rounds",
        y="final_choice_reflexion_time",
        hue="correct",
        marker="o",
        ax=ax1
    )
    ax1.set_xlabel("Number of Rounds")
    ax1.set_ylabel("Mean Reflexion Time (s)")
    ax1.grid(True)

    # right axis: bar of how many games contributed at each round
    ax2 = ax1.twinx()
    ax2.bar(
        count_agg["num_rounds"],
        count_agg["num_games"],
        alpha=0.3,
        width=0.8
    )
    ax2.set_ylabel("Number of Games")
    ax1.legend(title="Correct")
    plt.title("Final-Choice Reflexion Time by Rounds & Correctness")
    plt.tight_layout()
    plt.show()

def plot_biased_accuracy_by_round(games_df: pd.DataFrame,
                                  choices_df: pd.DataFrame) -> None:
    """
    For each round index, compute the proportion of choices where the player's
    chosen_quadrant matches that game's biased_quadrant.
    Prints how many games reached each round, and plots accuracy by round
    with sample‐size annotations.
    """
    # pull in the game's biased_quadrant
    df = choices_df.merge(
        games_df[["game_id", "biased_quadrant", "num_rounds"]],
        on="game_id",
        how="left"
    )
    

    # flag whether each choice matched the bias
    df["is_biased_choice"] = df["chosen_quadrant"] == df["biased_quadrant"]
    

    # aggregate per round
    stats = (
        df.groupby("round")
          .agg(accuracy=("is_biased_choice", "mean"),
               num_games=("game_id", "nunique"))
          .reset_index()
    )

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(stats["round"], stats["accuracy"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Round")
    plt.ylabel("Proportion of Choices Matching Biased Quadrant")
    plt.title("Biased-Choice Accuracy by Round")
    plt.grid(True)
    # annotate n above each point
    for _, row in stats.iterrows():
        plt.text(
            row["round"],
            row["accuracy"] + 0.02,
            f"n={int(row['num_games'])}",
            ha="center"
        )
    plt.tight_layout()
    plt.show()


def plot_reflexion_distribution_by_correctness(games_df: pd.DataFrame) -> None:
    """KDE of final‐choice reflexion time, with legend labels showing n for each correctness."""
    df = games_df.loc[
        games_df["final_choice_reflexion_time"].notnull() &
        (games_df["final_choice_reflexion_time"] <= MAX_REFLEXION_TIME)
    ]
    # compute counts per correctness
    counts = df["correct"].value_counts()
    plt.figure(figsize=(10,6))
    ax = sns.kdeplot(
        data=df,
        x="final_choice_reflexion_time",
        hue="correct",
        fill=True,
        common_norm=False,
        alpha=0.4
    )
    # rebuild legend labels to include n
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lbl in labels:
        # seaborn converts booleans to 'True'/'False'
        if lbl == "True":
            new_labels.append(f"Correct (n={counts.get(True, 0)})")
        elif lbl == "False":
            new_labels.append(f"Incorrect (n={counts.get(False, 0)})")
        else:
            new_labels.append(lbl)
    ax.legend(handles, new_labels, title="Correct")
    plt.xlabel("Final Choice Reflexion Time (s)")
    plt.title("Distribution of Reflexion Times by Correctness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_mean_reflection_by_correctness(
    games_df: pd.DataFrame,
    max_reflexion: float = MAX_REFLEXION_TIME
) -> None:
    """
    Bar‐plot of average final‐choice reflection time for correct vs. incorrect games,
    filtering out any games with reflexion time > max_reflexion. Also prints the
    number of games in each category.
    """
    # keep only valid, non‐too‐long reflexion times
    df = games_df.loc[
        games_df["final_choice_reflexion_time"].notnull() &
        (games_df["final_choice_reflexion_time"] <= max_reflexion)
    ].copy()

    # count how many games in each category
    counts = df.groupby("correct")["game_id"].nunique().rename({True: "Correct", False: "Incorrect"})
    print(f"Number of games used for plotting (reflexion ≤ {max_reflexion}s):")
    for label, cnt in counts.items():
        print(f"  {label}: {cnt}")

    # compute mean reflexion times
    agg = (
        df.groupby("correct")["final_choice_reflexion_time"]
          .mean()
          .rename({True: "Correct", False: "Incorrect"})
          .reset_index(name="mean_reflexion_time")
          .rename(columns={"correct": "label"})
    )

    # plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        agg["label"],
        agg["mean_reflexion_time"],
        alpha=0.7
    )
    plt.xlabel("Outcome")
    plt.ylabel("Mean Final-Choice Reflection Time (s)")
    plt.title(
        f"Average Reflection Time by Correctness (Filtered to <= {max_reflexion}s)"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # annotate counts above bars
    for bar, label in zip(bars, agg["label"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + max_reflexion*0.02,
            f"n={counts[label]}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()
