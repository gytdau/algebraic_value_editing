# %%

from tqdm import tqdm

import eval_tools
from eval_tools import run_eval, compare_with_goal, Sample
import pandas as pd
import plotly.express as px

pd.set_option("display.max_colwidth", None)


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

# %%[markdown]
# # Weddings
# Let's observe that the model is steered more effectively when the steeering vector is related to the prompt.
from IPython.display import display


def analyze_activation_vector(activation_additions, goal):
    prompts = ["I went up to my friend and said,"] * 10

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

    compare_with_goal(goal, prompts, steered, control)
    display(steered)


# %%

analyze_activation_vector(
    # https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#9__Talking_about_weddings
    activation_additions=get_x_vector(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=4,
        act_name=20,
        model=model,
        pad_method="tokens_right",
    ),
    goal="talking about weddings",
)

# %%[markdown]
# ## Our top-rated result from the weddings vector search

analyze_activation_vector(
    get_x_vector(
        prompt1="Planning my wedding",
        prompt2=" ",
        coeff=1,
        act_name=8,
        model=model,
        pad_method="tokens_right",
    ),
    goal="talking about weddings",
)

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
