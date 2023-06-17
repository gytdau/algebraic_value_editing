# %%


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


# Now let's generate some completions and evaluate them.

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cuda",
)

# %%[markdown]
# # Weddings
# Let's observe that the model is steered more effectively when the steeering vector is related to the prompt.
from IPython.display import display

sentences = [
    "In virtual reality, we often forget that",
    "Understanding quantum physics relies on the premise that",
    "Climate change is a present danger because",
    "Language serves as",
    "The debate about social media centers on",
]


def analyze_activation_vector(vectors, goal):
    results_dict = {}
    for sentence in sentences:
        prompts = [sentences] * 5

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

        results_dict[sentence] = compare_vectors_with_goal(goal, prompts, vectors)

    return results_dict


# %%[markdown]
# ## Our top-rated result from the weddings vector search
# Let's go back to the wedding scenario. In this chart, we compare how the wedding steering vector of the otters previously found compares to the control, which is not using any steering vector whatsoever. We can see that our range is pretty large. But the weddings vector performs significantly better than control at achieving our goal.


vectors, samples = analyze_activation_vector(
    # https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#9__Talking_about_weddings
    vectors=[
        SteeringVector(
            name="<i>All combined</i>",
            activation_additions=[
                *get_x_vector(
                    prompt1="Wedding Planning Adventures",
                    prompt2="Adventures in self-discovery",
                    coeff=4 / 3,
                    act_name=20,
                    model=model,
                    pad_method="tokens_right",
                ),
                *get_x_vector(
                    prompt1="I talk about weddings constantly",
                    prompt2="I do not talk about weddings constantly",
                    coeff=4 / 3,
                    act_name=20,
                    model=model,
                    pad_method="tokens_right",
                ),
                *get_x_vector(
                    prompt1="Obsessed with weddings!",
                    prompt2="Obsessed with self-care!",
                    coeff=4 / 3,
                    act_name=20,
                    model=model,
                    pad_method="tokens_right",
                ),
            ],
            completions=None,
        ),
        SteeringVector(
            name="I talk about weddings constantly",
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
            name="Wedding Planning Adventures",
            activation_additions=get_x_vector(
                prompt1="Wedding Planning Adventures",
                prompt2="Adventures in self-discovery",
                coeff=4 / 3,
                act_name=20,
                model=model,
                pad_method="tokens_right",
            ),
            completions=None,
        ),
        SteeringVector(
            name="Obsessed with weddings!",
            activation_additions=get_x_vector(
                prompt1="Obsessed with weddings!",
                prompt2="Obsessed with self-care!",
                coeff=4 / 3,
                act_name=20,
                model=model,
                pad_method="tokens_right",
            ),
            completions=None,
        ),
    ],
    goal="mentioning weddings or talking about them",
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
            line=dict(width=0),
            fillcolor="rgba(0,100,80,0)",
            pointpos=0,
            marker=dict(color="rgba(0,100,80,0.5)", size=10),
            jitter=1,
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
    title_text="Combining high performing vectors improves steering",
    showlegend=False,
)  # add centered title

fig.show()

# %%


def format_print(prompt, completion):
    completion = completion.replace("\n", " ")
    return f"**{prompt}** {completion}"


print("|Combined|I talk about weddings constantly|")
print("|---|---|")
for prompt, steered, steered_2 in zip(
    ["Science is the great antidote to the poison of enthusiasm and superstition."]
    * 20,
    samples[0],
    samples[1],
):
    print(
        "|"
        + format_print(prompt, steered.completion)
        + "|"
        + format_print(prompt, steered_2.completion)
        + "|"
    )

# %%
