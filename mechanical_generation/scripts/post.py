# %%

from tqdm import tqdm

import eval_tools
from eval_tools import (
    run_eval,
    compare_with_goal,
    Sample,
    compare_vectors_with_goal,
    SteeringVector,
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

pd.set_option("display.max_colwidth", None)


# %%[markdown]
# # Weddings in general

# %%
import sqlite3

db = sqlite3.connect("../main.db")

db.row_factory = sqlite3.Row

cursor = db.cursor()


def get_rows():
    rows = cursor.execute(
        """
        SELECT candidate_prompt, eval_score, act_name, challenge_id, challenge_prompt, candidate_id
        FROM simplified_results
        WHERE experiment_group = 'steered'
        """
    ).fetchall()
    return [dict(row) for row in rows]


rows = get_rows()


# Bold the "I talk about weddings constantly" candidate prompt by adding asterisks
for row in rows:
    if row["candidate_prompt"] == "I talk about weddings constantly":
        row["candidate_prompt"] = (
            row["candidate_prompt"] + "<br><sup>Original prompt</sup>"
        )

# %%

# Group these by prompt
grouped = pd.DataFrame(rows).groupby("candidate_prompt")

# Show a similar visualisation
fig = go.Figure()
# fig.add_vline(x=0, line_width=1)
# fig.add_vline(x=1, line_width=1)
# Calculate group means and sort from highest to lowest
group_means = grouped["eval_score"].mean().sort_values(ascending=True)

colors = {
    8: "rgba(255, 127, 127, 0.5)",
    16: "rgba(0, 119, 190, 0.7)",
    24: "rgba(150, 123, 182, 0.6)",
}

# Use ordered group names to add traces
for name in group_means.index:
    group = grouped.get_group(name)
    # further group by act_name
    act_grouped = group.groupby("act_name")
    for act_name, act_group in act_grouped:
        fig.add_trace(
            go.Box(
                y=[name] * len(act_group),
                x=act_group["eval_score"],
                name=name,
                boxpoints="all",
                line=dict(width=0),
                fillcolor="rgba(0,100,80,0)",
                pointpos=0,
                jitter=1,
                marker_color=colors[act_name],
                # bigger marker
                marker=dict(size=10),
                showlegend=False,
            )
        )

for act_name, color in colors.items():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            showlegend=True,
            name=f"Layer {act_name}",
        )
    )

fig.update_traces(orientation="h")
fig.update_layout(template="plotly_white")
fig.update_xaxes(
    title="Wedding relatedness of completions<br><sup>Higher is better</sup>",
    showline=True,
    linewidth=1,
    linecolor="black",
    range=[-0.05, 1.05],
    mirror=False,
)  # add more details to x-axis
fig.update_yaxes(
    showline=False, linewidth=2, linecolor="black", mirror=True
)  # add more details to y-axis
fig.update_layout(
    showlegend=True,
    height=1200,
    width=700,
    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
)  # remove legend and set siz
fig.update_layout(
    title_text="Some steering vectors work better than others", title_x=0.5
)  # add centered title
fig.show()


# %%[markdown]
# # Challenge-wise

# Group these by prompt
df = pd.DataFrame(rows)
grouped = df.groupby("candidate_prompt")
group_means = grouped["eval_score"].mean().sort_values(ascending=True)


# Show a similar visualisation
fig = go.Figure()


colors = {
    1: "rgba(255, 127, 127, 0.5)",
    2: "rgba(0, 119, 190, 0.7)",
    3: "rgba(150, 123, 182, 0.6)",
}


# Use ordered group names to add traces
for name in group_means.index:
    group = grouped.get_group(name)
    # further group by act_name
    challenge_grouped = group.groupby("challenge_id")
    for act_name, act_group in challenge_grouped:
        fig.add_trace(
            go.Box(
                y=[name] * len(act_group),
                x=act_group["eval_score"],
                name=name,
                boxpoints="all",
                line=dict(width=0),
                fillcolor="rgba(0,100,80,0)",
                pointpos=0,
                jitter=1,
                marker_color=colors[act_name],
                # bigger marker
                marker=dict(size=10),
                showlegend=False,
            )
        )

for act_name, color in colors.items():
    act_name_formatted = df[df["challenge_id"] == act_name]["challenge_prompt"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            showlegend=True,
            name=act_name_formatted,
        )
    )

fig.update_traces(orientation="h")
fig.update_layout(template="plotly_white")
fig.update_xaxes(
    title="Wedding relatedness of completions<br><sup>Higher is better</sup>",
    showline=True,
    linewidth=1,
    linecolor="black",
    range=[-0.05, 1.05],
    mirror=False,
)  # add more details to x-axis
fig.update_yaxes(
    showline=False, linewidth=2, linecolor="black", mirror=True
)  # add more details to y-axis
fig.update_layout(
    showlegend=True,
    height=1200,
    width=700,
    # legend title
    legend_title_text="Prompt used",
    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
    margin=dict(l=50, r=50, t=200, b=100),
)  # remove legend and set siz
fig.update_layout(
    title_text="Steering vectors work differently for different prompts",
)  # add centered title

fig.show()

# %%
