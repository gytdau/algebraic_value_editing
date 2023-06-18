# %%

from tqdm import tqdm
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

pd.set_option("display.max_colwidth", None)


# %%[markdown]
# # Weddings in general


# Setup the database
def get_database_rows(db_name):
    db = sqlite3.connect(db_name)
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    rows = cursor.execute(
        """
        SELECT candidate_prompt, eval_score, act_name, challenge_prompt, challenge_prompt, candidate_id, completion
        FROM simplified_results
        WHERE experiment_group = 'steered'
        """
    ).fetchall()
    return [dict(row) for row in rows]


def set_bold_candidate_prompt(rows, candidate_prompt, original_prompt):
    for row in rows:
        if row["candidate_prompt"] == candidate_prompt:
            row["candidate_prompt"] = (
                row["candidate_prompt"] + "<br><sup>" + original_prompt + "</sup>"
            )
    return rows


def make_visualisation(rows, x_title, y_title, title_text):
    # Group these by prompt
    grouped = pd.DataFrame(rows).groupby("candidate_prompt")
    fig = go.Figure()
    group_means = grouped["eval_score"].mean().sort_values(ascending=True)
    colors = {
        8: "rgba(255, 127, 127, 0.5)",
        16: "rgba(0, 119, 190, 0.7)",
        24: "rgba(150, 123, 182, 0.6)",
    }

    for name in group_means.index:
        group = grouped.get_group(name)
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
        title=x_title,
        showline=True,
        linewidth=1,
        linecolor="black",
        range=[-0.05, 1.05],
        mirror=False,
    )
    fig.update_yaxes(showline=False, linewidth=2, linecolor="black", mirror=True)
    fig.update_layout(
        showlegend=True,
        height=1200,
        width=700,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
    )
    fig.update_layout(title_text=title_text, title_x=0.5)
    fig.show()


rows = get_database_rows("../main.db")
# rows = set_bold_candidate_prompt(
#     rows, "I talk about weddings constantly", "Original prompt"
# )
make_visualisation(
    rows,
    "Wedding relatedness of completions<br><sup>Higher is better</sup>",
    "",
    "Some steering vectors work better than others",
)


# %%[markdown]
# # Challenge-wise

import matplotlib.pyplot as plt
import numpy as np


def make_challenge_visualisation(rows, x_title, y_title, title_text):
    df = pd.DataFrame(rows)
    # Drop a candidate prompt which didn't work out for some  of the challenges
    df = df[df["candidate_prompt"] != "creating THE PERFECT WEDDING"]
    # df = df[df["challenge_prompt"] == "I went up to a friend and said,"]
    grouped = df.groupby("candidate_prompt")
    group_means = grouped["eval_score"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = {
        1: "red",
        2: "blue",
        3: "purple",
    }

    for name in group_means.index:
        group = grouped.get_group(name)
        challenge_grouped = group.groupby("challenge_prompt")
        for act_name, act_group in challenge_grouped:
            ax.scatter(
                act_group["eval_score"],
                [name] * len(act_group),
                c="blue",
                alpha=0.5,
                label=name if act_name == 1 else "",
            )

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(title_text, loc="left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=0)
    ax.set_xlim([-0.05, 1.05])
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.show()


make_challenge_visualisation(
    rows,
    "Wedding relatedness of completions",
    "",
    "Some steering vectors work better than others",
)


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_challenge_visualisation(rows, x_title, y_title, title_text):
    df = pd.DataFrame(rows)

    # Drop a candidate prompt which didn't work out for some  of the challenges
    df = df[df["candidate_prompt"] != "creating THE PERFECT WEDDING"]

    challenge_prompts = df["challenge_prompt"].unique()
    num_challenges = len(challenge_prompts)

    fig, axs = plt.subplots(1, num_challenges, figsize=(5 * num_challenges, 5))

    group_means = (
        df.groupby("candidate_prompt")["eval_score"].mean().sort_values(ascending=True)
    )

    for idx, challenge_prompt in enumerate(challenge_prompts):
        ax = axs[idx]
        challenge_group = df[df["challenge_prompt"] == challenge_prompt]

        for name in group_means.index:
            group = challenge_group[challenge_group["candidate_prompt"] == name]

            ax.scatter(
                group["eval_score"],
                [name] * len(group),
                c="blue",
                alpha=0.5,
                label=name,
            )

        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_title(f'"{challenge_prompt}"', loc="left")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_tick_params(width=0)
        ax.set_xlim([-0.05, 1.05])
        ax.grid(axis="y", linestyle="--", alpha=0.2)

        if idx > 0:
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()


make_challenge_visualisation(
    rows,
    "Wedding relatedness of completions",
    "",
    "Steering vectors work differently for different prompts",
)


# %%
