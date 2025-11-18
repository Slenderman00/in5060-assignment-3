# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: hello-world-dev-env
#     language: python
#     name: python3
# ---

# %% Imports
import warnings
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import simdjson
from pydantic import BaseModel
from rich import print

warnings.filterwarnings("ignore")

from xdf_types import XDFData

# Set style for better-looking plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (16, 9)  # 16:9 aspect ratio for presentations


# %% Define types for questionnaire data
class AnswerRound(BaseModel):
    round_number: int
    latency_applied: int  # Actual latency in milliseconds from LatencyMarkers stream
    blocks_moved: int  # Number of blocks moved during this round (from BoxBlockMarkers)
    delays_experienced: int  # Answer to "Did you experience delays..."
    task_difficulty: int  # Answer to "How difficult was it..."
    felt_controlling: int  # Answer to "I felt like I was controlling..."
    felt_part_of_body: int  # Answer to "It felt like the robot was part of my body"


class Participant(BaseModel):
    submission_id: int
    created: str
    participant_number: int
    gender: str
    age: int
    dominant_hand: str  # Can be "Right hand", "Left hand", "Ambidextrous", etc.
    robotics_experience: int
    answer_time_ms: float
    rounds: list[AnswerRound]


# 4: Has 3 different XDF files
# 33: Doesn't have ExpMarkers
ignored_participants = [4, 33]


# %% Load questionnaire
questionnaire_df = pd.read_csv(
    "data/questionnaire_data-561422-2025-11-17-1240.csv", sep=";"
)

# %% Parse repeating questionnaire columns
# Static columns (demographics)
static_cols = [
    "$submission_id",
    "$created",
    "Participant number",
    "What is your gender",
    "How old are you?",
    "What is your dominant hand?",
    "How experienced are you with robotic systems?",
    "$answer_time_ms",
]

# The repeating question columns
repeating_questions = [
    "Did you experience delays between your actions and the robot&#39;s movements?",
    "How difficult was it to perform the task?",
    "I felt like I was controlling the movement of the robot",
    "It felt like the robot was part of my body",
]

# Count how many rounds we have
all_cols = questionnaire_df.columns.tolist()
# Remove static columns to get only repeating ones
repeating_cols = [col for col in all_cols if col not in static_cols]
num_rounds = len(repeating_cols) // len(repeating_questions)

print(f"Number of rounds: {num_rounds}")

# %% Create typed list of participants
participants: list[Participant] = []

for _, row in questionnaire_df.iterrows():
    participant_number = int(row["Participant number"])
    if participant_number in ignored_participants:
        continue

    data_file = f"./data/sub-{participant_number:03d}/sub-{participant_number:03d}_ses-_task-_run-001.json"
    json_parser = simdjson.Parser()
    data: XDFData = cast(XDFData, json_parser.load(data_file))

    # Extract latency markers from the data
    latency_markers_stream = next(
        (
            stream
            for stream in data["streams"]
            if stream["info"]["name"] == "LatencyMarkers"
        ),
        None,
    )

    if latency_markers_stream is None:
        raise ValueError(
            f"LatencyMarkers stream not found for participant {participant_number}"
        )

    # Extract latencies from condition_advance markers
    # Format: ["condition_advance|rep_1|200ms|condition_1"]
    latencies_by_round = []
    for marker in latency_markers_stream["time_series"]:
        marker_str = marker[0]
        if marker_str.startswith("condition_advance|"):
            parts = marker_str.split("|")
            if len(parts) >= 3:
                latency_str = parts[2]  # e.g., "200ms"
                latency_ms = int(latency_str.replace("ms", ""))
                latencies_by_round.append(latency_ms)

    # Extract ExpMarkers stream to count block_moved events
    # Note: Some participants have duplicate ExpMarkers streams (identical events)
    # We only need to process one of them
    exp_markers_stream = None
    for stream in data["streams"]:
        if stream["info"]["name"] == "ExpMarkers":
            exp_markers_stream = stream
            break

    if exp_markers_stream is None:
        raise ValueError(
            f"No ExpMarkers stream found for participant {participant_number}"
        )

    # Count blocks moved per round
    blocks_moved_by_round = []
    current_round_blocks = 0
    in_boxblock = False

    for i, marker in enumerate(exp_markers_stream["time_series"]):
        marker_str = marker[0]

        # Handle both practice and regular boxblock sessions
        if marker_str in ["boxblock_start", "practice_boxblock_start"]:
            # If we were already in a session, save the count first
            if in_boxblock and marker_str == "boxblock_start":
                blocks_moved_by_round.append(current_round_blocks)
            in_boxblock = True
            current_round_blocks = 0
        elif marker_str in [
            "boxblock_stop",
            "practice_boxblock_stop",
            "boxblock_end",
        ]:
            if in_boxblock:
                blocks_moved_by_round.append(current_round_blocks)
                in_boxblock = False
                current_round_blocks = 0
        elif marker_str == "block_moved" and in_boxblock:
            current_round_blocks += 1

    # Ensure we have the right number of block counts
    while len(blocks_moved_by_round) < num_rounds:
        blocks_moved_by_round.append(0)

    # Parse answer rounds
    rounds = []
    for round_idx in range(num_rounds):
        # Column names have suffixes like .1, .2, etc. (pandas duplicate column naming)
        # First occurrence has no suffix, then .1, .2, .3, etc.
        suffix = f".{round_idx}" if round_idx > 0 else ""

        round_data = AnswerRound(
            round_number=round_idx + 1,
            latency_applied=latencies_by_round[round_idx]
            if round_idx < len(latencies_by_round)
            else 0,
            blocks_moved=blocks_moved_by_round[round_idx]
            if round_idx < len(blocks_moved_by_round)
            else 0,
            delays_experienced=int(row[f"{repeating_questions[0]}{suffix}"]),
            task_difficulty=int(row[f"{repeating_questions[1]}{suffix}"]),
            felt_controlling=int(row[f"{repeating_questions[2]}{suffix}"]),
            felt_part_of_body=int(row[f"{repeating_questions[3]}{suffix}"]),
        )
        rounds.append(round_data)

    # Create participant with all rounds
    participant = Participant(
        submission_id=int(row["$submission_id"]),
        created=str(row["$created"]),
        participant_number=participant_number,
        gender=str(row["What is your gender"]),
        age=int(row["How old are you?"]),
        dominant_hand=str(row["What is your dominant hand?"]),
        robotics_experience=int(row["How experienced are you with robotic systems?"]),
        answer_time_ms=float(row["$answer_time_ms"]),
        rounds=rounds,
    )
    participants.append(participant)

# %% Print summary statistics
print(f"Total participants: {len(participants)}")
print(f"Total rounds per participant: {num_rounds}")

# %% Convert to long-format DataFrame for analysis
data_rows = []
for participant in participants:
    for round_data in participant.rounds:
        data_rows.append(
            {
                "participant_number": participant.participant_number,
                "round_number": round_data.round_number,
                "latency_applied": round_data.latency_applied,
                "delays_experienced": round_data.delays_experienced,
                "task_difficulty": round_data.task_difficulty,
                "felt_controlling": round_data.felt_controlling,
                "felt_part_of_body": round_data.felt_part_of_body,
                "blocks_moved": round_data.blocks_moved,
                "age": participant.age,
                "gender": participant.gender,
                "robotics_experience": participant.robotics_experience,
            }
        )

df_long = pd.DataFrame(data_rows)

print("Data structure:")
print(df_long.head(10))
print(f"Shape: {df_long.shape}")
print(f"Unique latency conditions: {sorted(df_long['latency_applied'].unique())}")

# %% Define dependent variables for ANOVA analysis
dependent_vars = {
    "delays_experienced": "Delays Experienced (1-7)",
    "task_difficulty": "Task Difficulty (1-7)",
    "felt_controlling": "Felt Controlling Robot (1-7)",
    "felt_part_of_body": "Robot Felt Part of Body (1-7)",
}

# %% One-Way Repeated Measures ANOVA for each dependent variable
anova_results = {}

for dv_name, dv_label in dependent_vars.items():
    print(f"\n{'─' * 80}")
    print(f"Analysis: {dv_label}")
    print(f"Independent Variable: Latency Applied (ms)")
    print(f"{'─' * 80}")

    # Run repeated measures ANOVA
    aov = pg.rm_anova(
        data=df_long,
        dv=dv_name,
        within="latency_applied",
        subject="participant_number",
        detailed=True,
    )

    # Extract variance components
    ss_treatment = aov.loc[aov["Source"] == "latency_applied", "SS"].values[0]
    ss_error = aov.loc[aov["Source"] == "Error", "SS"].values[0]
    df_treatment = aov.loc[aov["Source"] == "latency_applied", "DF"].values[0]
    df_error = aov.loc[aov["Source"] == "Error", "DF"].values[0]
    f_value = aov.loc[aov["Source"] == "latency_applied", "F"].values[0]
    p_value = aov.loc[aov["Source"] == "latency_applied", "p-unc"].values[0]

    # Store results
    anova_results[dv_name] = {
        "table": aov,
        "SSA": ss_treatment,
        "SSE": ss_error,
        "F": f_value,
        "p": p_value,
        "significant": p_value < 0.05,
    }

    # Print ANOVA table
    print("\nANOVA Table:")
    print(aov.to_string(index=False))

    # Print variance components (as required)
    print(f"\nVariance Components:")
    print(f"  SSA (Sum of Squares - Treatment): {ss_treatment:.4f}")
    print(f"  SSE (Sum of Squares - Error):     {ss_error:.4f}")
    print(f"  Total SS:                         {ss_treatment + ss_error:.4f}")
    print(f"  F-statistic:                      {f_value:.4f}")
    print(f"  p-value:                          {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05) ✓")
    else:
        print(f"  Result: NOT SIGNIFICANT (p ≥ 0.05)")

# %% Post-hoc pairwise comparisons for significant effects
posthoc_results = {}

for dv_name, dv_label in dependent_vars.items():
    if anova_results[dv_name]["significant"]:
        print(f"\n{'─' * 80}")
        print(f"Post-hoc for: {dv_label}")
        print(f"{'─' * 80}")

        # Pairwise comparisons with Bonferroni correction
        posthoc = pg.pairwise_tests(
            data=df_long,
            dv=dv_name,
            within="latency_applied",
            subject="participant_number",
            padjust="bonf",
            effsize="hedges",
        )

        posthoc_results[dv_name] = posthoc

        # Display results
        print("\nPairwise Comparisons (Bonferroni corrected):")
        print(
            posthoc[["A", "B", "T", "p-unc", "p-corr", "hedges"]].to_string(index=False)
        )

        # Highlight significant comparisons
        sig_comparisons = posthoc[posthoc["p-corr"] < 0.05]
        if len(sig_comparisons) > 0:
            print(f"\nSignificant pairwise differences (p < 0.05):")
            for _, row in sig_comparisons.iterrows():
                print(
                    f"  {row['A']}ms vs {row['B']}ms: p = {row['p-corr']:.6f}, Hedges' g = {row['hedges']:.3f}"
                )
        else:
            print("\nNo significant pairwise differences after correction.")
    else:
        print(f"\nSkipping post-hoc for '{dv_label}' (ANOVA was not significant)")

# %% Descriptive statistics by latency condition
for dv_name, dv_label in dependent_vars.items():
    print(f"\n{dv_label}:")
    desc_stats = (
        df_long.groupby("latency_applied")[dv_name]
        .agg(
            [
                ("Mean", "mean"),
                ("SD", "std"),
                ("SE", lambda x: x.std() / np.sqrt(len(x))),
                ("Min", "min"),
                ("Max", "max"),
                ("N", "count"),
            ]
        )
        .round(3)
    )
    print(desc_stats.to_string())

# %% Visualizations: Box plots for all dependent variables
fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))  # 16:9 aspect ratio
axes = axes.flatten()

for idx, (dv_name, dv_label) in enumerate(dependent_vars.items()):
    ax = axes[idx]

    # Create box plot
    sns.boxplot(data=df_long, x="latency_applied", y=dv_name, ax=ax, palette="Set2")

    # Add individual points with jitter
    sns.stripplot(
        data=df_long,
        x="latency_applied",
        y=dv_name,
        ax=ax,
        color="black",
        alpha=0.3,
        size=2,
    )

    ax.set_xlabel("Latency Applied (ms)", fontsize=12)
    ax.set_ylabel("Rating (1-7)", fontsize=12)
    ax.set_title(
        f"{dv_label}\n(F={anova_results[dv_name]['F']:.2f}, p={anova_results[dv_name]['p']:.4f})",
        fontsize=13,
        fontweight="bold",
    )

    # Add significance indicator
    if anova_results[dv_name]["significant"]:
        ax.text(
            0.95,
            0.95,
            "✓ Significant",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            fontsize=10,
        )

    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/anova_boxplots.png", dpi=300, bbox_inches="tight")
print("✓ Saved box plots to 'plots/anova_boxplots.png'")
plt.show()

# %% Visualizations: Point plots with confidence intervals
fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))  # 16:9 aspect ratio
axes = axes.flatten()

for idx, (dv_name, dv_label) in enumerate(dependent_vars.items()):
    ax = axes[idx]

    # Create point plot with 95% CI
    sns.pointplot(
        data=df_long,
        x="latency_applied",
        y=dv_name,
        ax=ax,
        errorbar="ci",
        capsize=0.1,
        color="steelblue",
        markers="o",
        linestyles="-",
    )

    ax.set_xlabel("Latency Applied (ms)", fontsize=12)
    ax.set_ylabel("Mean Rating (1-7)", fontsize=12)
    ax.set_title(f"{dv_label}\nMean ± 95% CI", fontsize=13, fontweight="bold")

    # Add significance indicator
    if anova_results[dv_name]["significant"]:
        ax.text(
            0.95,
            0.95,
            "✓ Significant",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            fontsize=10,
        )

    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 8)

plt.tight_layout()
plt.savefig("plots/anova_pointplots.png", dpi=300, bbox_inches="tight")
print("✓ Saved point plots to 'plots/anova_pointplots.png'")
plt.show()

# %% Visualizations: Violin plots
fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))  # 16:9 aspect ratio
axes = axes.flatten()

for idx, (dv_name, dv_label) in enumerate(dependent_vars.items()):
    ax = axes[idx]

    # Create violin plot
    sns.violinplot(
        data=df_long,
        x="latency_applied",
        y=dv_name,
        ax=ax,
        palette="pastel",
        inner="box",
    )

    ax.set_xlabel("Latency Applied (ms)", fontsize=12)
    ax.set_ylabel("Rating (1-7)", fontsize=12)
    ax.set_title(
        f"{dv_label}\nDistribution by Latency Condition", fontsize=13, fontweight="bold"
    )

    # Add significance indicator
    if anova_results[dv_name]["significant"]:
        ax.text(
            0.95,
            0.95,
            "✓ Significant",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            fontsize=10,
        )

    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/anova_violinplots.png", dpi=300, bbox_inches="tight")
print("✓ Saved violin plots to 'plots/anova_violinplots.png'")
plt.show()

# %% Summary results table
summary_data = []
for dv_name, dv_label in dependent_vars.items():
    result = anova_results[dv_name]
    summary_data.append(
        {
            "Dependent Variable": dv_label,
            "SSA": f"{result['SSA']:.4f}",
            "SSE": f"{result['SSE']:.4f}",
            "F-statistic": f"{result['F']:.4f}",
            "p-value": f"{result['p']:.6f}",
            "Significant": "✓ Yes" if result["significant"] else "No",
        }
    )

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
