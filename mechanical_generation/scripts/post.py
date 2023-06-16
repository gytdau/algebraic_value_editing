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
        SELECT candidate_prompt, eval_score, act_name, challenge_id, challenge_prompt, candidate_id
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
rows = set_bold_candidate_prompt(
    rows, "I talk about weddings constantly", "Original prompt"
)
make_visualisation(
    rows,
    "Wedding relatedness of completions<br><sup>Higher is better</sup>",
    "",
    "Some steering vectors work better than others",
)


# %%[markdown]
# # Challenge-wise


def make_challenge_visualisation(rows, x_title, y_title, title_text):
    df = pd.DataFrame(rows)
    grouped = df.groupby("candidate_prompt")
    group_means = grouped["eval_score"].mean().sort_values(ascending=True)

    fig = go.Figure()

    colors = {
        1: "rgba(255, 127, 127, 0.5)",
        2: "rgba(0, 119, 190, 0.7)",
        3: "rgba(150, 123, 182, 0.6)",
    }

    for name in group_means.index:
        group = grouped.get_group(name)
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
                    marker=dict(size=10),
                    showlegend=False,
                )
            )

    for act_name, color in colors.items():
        act_name_formatted = df[df["challenge_id"] == act_name][
            "challenge_prompt"
        ].iloc[0]
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
        legend_title_text="Prompt used",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
        margin=dict(l=50, r=50, t=200, b=100),
    )
    fig.update_layout(title_text=title_text)
    fig.show()


make_challenge_visualisation(
    rows,
    "Wedding relatedness of completions<br><sup>Higher is better</sup>",
    "",
    "Steering vectors work differently for different prompts",
)

# %%


def make_challenge_visualisation(rows, x_title, y_title, title_text):
    df = pd.DataFrame(rows)
    pivot_df = df.pivot_table(
        index="candidate_prompt",
        columns="challenge_id",
        values="eval_score",
        aggfunc="mean",
    ).reset_index()



    # by mean of all challenges
    pivot_df = pivot_df.sort_values(by=[1, 2, 3], ascending=False)

    fig = go.Figure()
    colors = {
        1: "rgba(255, 127, 127, 0.5)",
        2: "rgba(0, 119, 190, 0.7)",
        3: "rgba(150, 123, 182, 0.6)",
    }

    for challenge_id, color in colors.items():
        challenge_prompt = df[df["challenge_id"] == challenge_id][
            "challenge_prompt"
        ].iloc[0]
        fig.add_trace(
            go.Bar(
                name=challenge_prompt,
                y=pivot_df["candidate_prompt"],
                x=pivot_df[challenge_id],
                marker_color=color,
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
        legend_title_text="Prompt used",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
        margin=dict(l=50, r=50, t=200, b=100),
    )
    fig.update_layout(title_text=title_text)
    fig.show()


make_challenge_visualisation(
    rows,
    "Wedding relatedness of completions<br><sup>Higher is better</sup>",
    "",
    "Steering vectors work differently for different prompts",
)

# %%
