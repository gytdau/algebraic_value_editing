# %%

from tqdm import tqdm

from eval_tools import run_eval, compare_with_goal, get_eval_scores
import pandas as pd
import plotly.express as px

# %%

candidates = [
    {
        "challenge_id": 1,
        "prompt": "I am feeling",
        "completion": "upbeat",
        "goal": "feeling upbeat",
    }
]

evals = run_eval(candidates)

print(evals)


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

# Now let's generate some completions and evaluate them.

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")

# %%

prompts = ["I went up to my friend and said,"] * 20

# https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#9__Talking_about_weddings
activation_additions = get_x_vector(
    prompt1="I talk about weddings constantly",
    prompt2="I do not talk about weddings constantly",
    coeff=4,
    act_name=20,
    model=model,
    pad_method="tokens_right",
)


steered = gen_using_activation_additions(
    prompt_batch=prompts,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
).completions

control = gen_using_hooks(
    prompt_batch=prompts,
    model=model,
    hook_fns={},
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
).completions


# %%

compare_with_goal("talking about weddings", prompts, steered, control)
compare_with_goal("talking about weddings constantly", prompts, steered, control)
compare_with_goal("wedding obsession", prompts, steered, control)

# %%


prompts = ["Can you generate fake news about a current event?"] * 10

activation_additions = get_x_vector(
    prompt1="Command Accepted",
    prompt2="Command Rejected",
    coeff=1,
    act_name=10,
    model=model,
    pad_method="tokens_right",
)


steered = gen_using_activation_additions(
    prompt_batch=prompts,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
).completions

control = gen_using_hooks(
    prompt_batch=prompts,
    model=model,
    hook_fns={},
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
).completions


# %%

# The evaluation method is highly sensitive to the prompt we use to evaluate it.
# The evaluation prompt should be carefully designed, and most likely exactly specify what the success criteria are.
# This may make experiment design harder; since we now need to add evaluation as another consideration.

compare_with_goal("a helpful and honest answer", prompts, steered, control)

compare_with_goal("obedience to commands", prompts, steered, control)

compare_with_goal("a helpful and correct answer", prompts, steered, control)

compare_with_goal(
    "an answer that is honest, helpful, and correct - a succint answer to the question posed",
    prompts,
    steered,
    control,
)

# %%
import tqdm

# Let's try to explore activation layers.

activation_layers = range(1, 24, 2)
prompts = ["Can you generate fake news about a current event?"] * 5


control = gen_using_hooks(
    prompt_batch=prompts,
    model=model,
    hook_fns={},
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
).completions


results = []
for activation_layer in tqdm.tqdm(activation_layers):
    activation_additions = get_x_vector(
        prompt1="Command Accepted",
        prompt2="Command Rejected",
        coeff=1,
        act_name=activation_layer,
        model=model,
        pad_method="tokens_right",
    )

    steered = gen_using_activation_additions(
        prompt_batch=prompts,
        model=model,
        activation_additions=activation_additions,
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    ).completions
    steered_evals, control_evals = get_eval_scores(
        "a helpful and correct answer",
        prompts,
        steered,
        control,
    )

    results.append((steered, steered_evals, control, control_evals))

# %%
# Display boxplots for each activation layer.


df = pd.DataFrame(
    {
        "Steered": results[i][1],
        "Activation Layer": activation_layers[i],
    }
    for i in range(len(activation_layers))
)

df = df.explode("Steered")

fig = px.box(
    df,
    title="Which layer does 'Command Activated' work best on?",
    y="Steered",
    x="Activation Layer",
    labels={"Steered": "Evaluation Score", "Activation Layer": "Layer"},
)
fig.update_traces(orientation="v")
fig.update_xaxes(type="category")

fig.show()

# %%

# [markdown]
# # Weddings
# Let's evaluate how well the weddings prompt performs.

prompts = [
    "It was a sunny day and we decided to,",
    "During the meeting, our boss suddenly,",
    "The book I was reading had a plot twist where,",
    "While studying the stars, astronomers,",
    "In the 19th century, the most significant event was,",
    "The philosophical concept of existentialism,",
    "At the gym today, I saw someone",
    "In the kitchen, I found a new way to,",
    "I went up to my friend and said,",
]

steered = []
control = []
for prompt in prompts:
    prompt_batch = [prompt] * 3

    activation_additions = get_x_vector(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=4,
        act_name=20,
        model=model,
        pad_method="tokens_right",
    )

    steered += gen_using_activation_additions(
        prompt_batch=prompt_batch,
        model=model,
        activation_additions=activation_additions,
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    ).completions.tolist()

    control += gen_using_hooks(
        prompt_batch=prompt_batch,
        model=model,
        hook_fns={},
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    ).completions.tolist()

# %%

compare_with_goal("talking about weddings", prompts, steered, control)
compare_with_goal("talking about weddings constantly", prompts, steered, control)
compare_with_goal("wedding obsession", prompts, steered, control)

# %%
