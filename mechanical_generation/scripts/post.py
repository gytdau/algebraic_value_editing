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


# %%
%autoreload 2


from typing import List

import torch
from tqdm.notebook import tqdm
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, hook_utils, utils
from algebraic_value_editing.completion_utils import (
    gen_using_activation_additions,
    gen_using_hooks,
)
from algebraic_value_editing.prompt_utils import ActivationAddition, get_x_vector

# Now let's generate some completions and evaluate them.

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cuda",
)

# %%[markdown]
# # Weddings
# Let's observe that the model is steered more effectively when the steeering vector is related to the prompt.
from IPython.display import display


def analyze_activation_vector(vectors, goal):
    prompts = ["I went up to my friend and said,"] * 10

    for vector in vectors:
        if vector.activation_additions is None:
            completion = gen_using_hooks(
                prompt_batch=prompts,
                model=model,
                hook_fns={},
                seed=0,
                temperature=1,
                freq_penalty=1,
                top_p=0.3,
            ).completions
        else:
            completion = gen_using_activation_additions(
                prompt_batch=prompts,
                model=model,
                activation_additions=vector.activation_additions,
                seed=0,
                temperature=1,
                freq_penalty=1,
                top_p=0.3,
            ).completions

        vector.completions = completion

    return compare_vectors_with_goal(goal, prompts, vectors)


# %%[markdown]
# ## Our top-rated result from the weddings vector search
# Let's go back to the wedding scenario. In this chart, we compare how the wedding steering vector of the otters previously found compares to the control, which is not using any steering vector whatsoever. We can see that our range is pretty large. But the weddings vector performs significantly better than control at achieving our goal.


vectors, samples = analyze_activation_vector(
    # https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#9__Talking_about_weddings
    vectors=[
        SteeringVector(
            name="Control",
            activation_additions=None,
            completions=None,
        ),
        SteeringVector(
            name="Authors",
            activation_additions=get_x_vector(
                prompt1="I talk about weddings constantly",
                prompt2="I do not talk about weddings constantly",
                coeff=4,
                act_name=20,
                model=model,
                pad_method="tokens_right",
            ),
            completions=None,
        ),
        SteeringVector(
            name="New",
            activation_additions=get_x_vector(
                prompt1="Planning my wedding",
                prompt2=" ",
                coeff=1,
                act_name=8,
                model=model,
                pad_method="tokens_right",
            ),
            completions=None,
        ),
    ],
    goal="talking about weddings",
)

# %%

fig = go.Figure()
for i, data_line in enumerate(vectors):
    fig.add_trace(
        go.Box(
            y=[data_line.name] * len(data_line.completions),
            x=[sample.eval_score for sample in samples[i]],
            name=data_line.name,
            boxpoints="all",
        )
    )

fig.show()

fig.update_traces(orientation="h")
fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False, template="plotly_white")
fig.update_layout(showlegend=False)
fig.update_layout(xaxis_title="Higher is better")
fig.show()


# %%

fig = go.Figure()
for i, data_line in enumerate(vectors):
    fig.add_trace(
        go.Box(
            y=[data_line.name] * len(data_line.completions),
            x=[sample.eval_score for sample in samples[i]],
            name=data_line.name,
            boxpoints="all",
            line=dict(width=0),
            fillcolor="rgba(0,100,80,0)",
            pointpos=0,
            marker=dict(color="rgba(0,100,80,0.5)", size=10),
            jitter=1,
        )
    )

fig.update_traces(orientation="h")
fig.update_layout(template="simple_white")
fig.update_xaxes(title="Eval scores (higher is better)", showline=True, linewidth=1, linecolor='black', range=[-0.05, 1.05])  # add more details to x-axis
fig.update_yaxes(showline=False, linewidth=2, linecolor='black', mirror=True)  # add more details to y-axis
fig.update_layout(showlegend=False, width=600, height=400)  # remove legend and set siz
fig.update_layout(title_text='Mechanically evaluated vector outperforms the original', title_x=0.5)  # add centered title

fig.show()


# %%[markdown]
# ## A simple vector that just says "weddings"

analyze_activation_vector(
    get_x_vector(
        prompt1=" weddings",
        prompt2=" ",
        coeff=1,
        act_name=8,
        model=model,
        pad_method="tokens_right",
    ),
    goal="talking about weddings",
)

# %%[markdown]
# ## Not-liking-her vector


analyze_activation_vector(
    get_x_vector(
        prompt1="I don't like her",
        prompt2="",
        coeff=1,
        act_name=8,
        model=model,
        pad_method="tokens_right",
    ),
    goal="someone disliking the other person",
)

# %%[markdown]
# ## Not-liking-her vector


analyze_activation_vector(
    get_x_vector(
        prompt1="Let's get lunch",
        prompt2="",
        coeff=1,
        act_name=8,
        model=model,
        pad_method="tokens_right",
    ),
    goal="someone disliking the other person",
)

# %%
