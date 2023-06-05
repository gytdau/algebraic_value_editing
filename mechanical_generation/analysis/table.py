# %%
# Import required libraries
import sqlite3
import pandas as pd
import numpy as np

# Connect to the sqlite database
conn = sqlite3.connect("steering_vectors.db")

# Load candidates and results table into pandas DataFrames
candidates_df = pd.read_sql_query("SELECT * from candidates", conn)
results_df = pd.read_sql_query("SELECT * from results", conn)

# Close the connection
conn.close()

# Join candidates_df and results_df on 'id' field of candidates
joined_df = results_df.set_index("candidate_id").join(
    candidates_df.set_index("id"), how="left"
)

steered = joined_df[joined_df["experiment_group"] == "steered"]
control = joined_df[joined_df["experiment_group"] == "control"]

# Create a pivot table with columns as 'prompt1', rows as 'act_name' and values as 'eval_score'
steered_pivot_table = pd.pivot_table(
    steered,
    values="eval_score",
    index=["act_name"],
    columns=["prompt1"],
    aggfunc=np.mean,
)

control_pivot_table = pd.pivot_table(
    control,
    values="eval_score",
    index=["act_name"],
    columns=["prompt1"],
    aggfunc=np.mean,
)

# %%

# Difference between steered and control groups

pivot_table = steered_pivot_table - control_pivot_table
# %%
# Import required library
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Sort the columns by sum of values
pivot_table = pivot_table.reindex(
    pivot_table.sum().sort_values(ascending=False).index, axis=1
)

# Apply a color gradient based on cell value
# Set the center point to zero
styled_table = pivot_table.style.background_gradient(
    cmap="RdBu", vmin=-100, vmax=100
).format("{:.2f}")

# Display the styled table
styled_table
# %%

# We've observed interesting results on the candidate of prompt1 "rosy" and act_name 2.
# Let's print the completions for this candidate.

interesting_results = joined_df[
    (joined_df["prompt1"] == "cheerful") & (joined_df["experiment_group"] == "steered")
]

interesting_results = interesting_results[
    ["completion", "act_name", "experiment_group"]
]

interesting_results.style.format({"completion": lambda x: ("I am feeling " + x)})

# %%
