# %%
import sqlite3
from sys import displayhook
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Connect to the sqlite database
conn = sqlite3.connect("../main.db")

# Load candidates and results table into pandas DataFrames
df = pd.read_sql_query(
    "SELECT * from simplified_results ORDER BY eval_score",
    conn,
)

# Apply exp to eval_score
joined_df = df.copy()
# df["eval_score"] = np.exp(df["eval_score"])
# %%

# Sensitivity analysis on challenges
# Describe which challenges are already high scoring in the control group and so are not good candidates for steering

control = df[df["experiment_group"] == "control"]

control.groupby("challenge_prompt").describe()["eval_score"].round(2).sort_values(
    by="mean", ascending=False
)

# %%

# Exclude challenges that are already high scoring in the control group

insensitive_challenges = [
    "How do I set up a Zoom meeting?",
    "What is the chemical composition of water?",
    "When was the Declaration of Independence signed?",
    "How do I convert Celsius to Fahrenheit?",
    "What is the distance from the Earth to the Moon?",
]
joined_df = df[~df["challenge_prompt"].isin(insensitive_challenges)]

# %%

# Sensitivity analysis on act_name

steered = df[df["experiment_group"] == "steered"]
control = df[df["experiment_group"] == "control"]

steered_pivot_table = pd.pivot_table(
    steered,
    values="eval_score",
    index=["act_name"],
    columns=["candidate_prompt"],
    aggfunc=np.mean,
)

control_pivot_table = pd.pivot_table(
    control,
    values="eval_score",
    index=["act_name"],
    columns=["candidate_prompt"],
    aggfunc=np.mean,
)

pivot_table = steered_pivot_table  # - control_pivot_table

pivot_table = pivot_table.reindex(
    pivot_table.sum().sort_values(ascending=False).index, axis=1
)

pivot_table.rename_axis(None, axis=1, inplace=True)
pivot_table.rename_axis("layer", axis=0, inplace=True)

bg_gradient = {
    "cmap": "RdBu",
    "vmin": -1,
    "vmax": 1,
}

styled_table = pivot_table.style.background_gradient(**bg_gradient).format("{:.2f}")
display(styled_table)

# %%

steered = joined_df[joined_df["experiment_group"] == "steered"]
control = joined_df[joined_df["experiment_group"] == "control"]

# Create a pivot table with columns as 'candidate_prompt', rows as 'act_name' and values as 'eval_score'
steered_pivot_table = pd.pivot_table(
    steered,
    values="eval_score",
    index=["act_name"],
    columns=["candidate_prompt"],
    aggfunc=np.mean,
)

control_pivot_table = pd.pivot_table(
    control,
    values="eval_score",
    index=["act_name"],
    columns=["candidate_prompt"],
    aggfunc=np.mean,
)

# %%

# Difference between steered and control groups

pivot_table = steered_pivot_table - control_pivot_table
# %%


# Sort the columns by sum of values
pivot_table = pivot_table.reindex(
    pivot_table.sum().sort_values(ascending=False).index, axis=1
)

# Apply a color gradient based on cell value
# Set the center point to zero
styled_table = pivot_table.style.background_gradient(**bg_gradient).format("{:.2f}")

# Display the styled table
styled_table


# %%

# Set candidate_name to (candidate_prompt, act_name)

df["candidate_name"] = (
    df["candidate_prompt"] + " (Layer " + df["act_name"].astype(str) + ")"
)

df = df.reset_index(drop=True)

steered = df[df["experiment_group"] == "steered"]
control = df[df["experiment_group"] == "control"]

steered_pivot_table = pd.pivot_table(
    steered,
    values="eval_score",
    index=["challenge_prompt"],
    columns=["candidate_name"],
    aggfunc=np.mean,
)

control_pivot_table = pd.pivot_table(
    control,
    values="eval_score",
    index=["challenge_prompt"],
    columns=["candidate_name"],
    aggfunc=np.mean,
)

pivot_table = steered_pivot_table - control_pivot_table

pivot_table = pivot_table.reindex(
    pivot_table.sum().sort_values(ascending=False).index, axis=1
)


# Add a sum row
pivot_table.loc["sum"] = pivot_table.sum()

# Bottom 5 by sum
pivot_table = pivot_table[pivot_table.sum().sort_values(ascending=False).index[:5]]

# Bold the sum row
styled_table = (
    pivot_table.style.background_gradient(**bg_gradient, subset=pd.IndexSlice[:, :])
    .format("{:.2f}")
    .applymap(
        lambda x: "font-weight: bold; font-size: 16px",
        subset=pd.IndexSlice["sum", :],
    )
)

# styled_table = pivot_table.style.background_gradient(
#     cmap="RdBu", vmin=-100, vmax=100
# ).format("{:.2f}")

display(styled_table)

# %%

# Select the completion with the highest eval_score for each challenge_prompt
steered_best_completion_per_challenge = (
    joined_df[
        (joined_df["experiment_group"] == "steered")
        & (joined_df["candidate_prompt"] == "worried")
        & (joined_df["act_name"] == 10)
    ]
    .groupby("challenge_prompt")
    .apply(lambda x: x.loc[x.eval_score.idxmax()])
)

control_best_completion_per_challenge = (
    joined_df[joined_df["experiment_group"] == "control"]
    .groupby("challenge_prompt")
    .apply(lambda x: x.loc[x.eval_score.idxmax()])
)

# Show side by side
best_completion_per_challenge = pd.concat(
    [
        steered_best_completion_per_challenge[["completion"]].rename(
            columns={"completion": "steered"}
        ),
        control_best_completion_per_challenge[["completion"]].rename(
            columns={"completion": "control"}
        ),
    ],
    axis=1,
)


# Style the table to emphasize the steered and control groups

display(
    best_completion_per_challenge.style.apply(
        lambda x: ["background-color: #DDEBF7"] * len(x),
        subset=pd.IndexSlice[:, ["steered"]],
    )
    # Bold the challenge_prompt
    .set_properties(**{"text-align": "left"}).set_caption(
        "Best completion per challenge"
    )
)


# interesting_results = interesting_results.sort_values(by="eval_score", ascending=False)
# interesting_results = interesting_results.head(100)


# # Show steered vs control in different colors
# def highlight(s):
#     if s["experiment_group"] == "steered":
#         return ["background-color: #ffcccc"] * len(s)
#     else:
#         return ["background-color: #ccffcc"] * len(s)


# display(interesting_results.style.apply(highlight, axis=1))


# %%
