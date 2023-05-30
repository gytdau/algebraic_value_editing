# %%
from typing import List

import torch
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, utils, hook_utils
from algebraic_value_editing.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)

utils.enable_ipython_reload()

# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")

# %%
# It is extremely hard to find a successful vector. Even a simple project like getting it to say it's favourite food is eggs has failed after 10 various attempts at trephrasing.

# With a negative coeff, we move in the direction of prompt1 (.. why is this inverted?)

# This is a successful egg liking/hatred vector, but it falls apart if the prompt fails to mention eggs. It's not clear if the actual vector is about sentiment in general or if it's about eggs.

# WE HAVE GIVEN UP ON THIS APPROACH. IT IS TOO HARD TO FIND A SUCCESSFUL VECTOR.

# We are now talking about excitement/boredom

# David McSharry books a prediction:
# It's necessary for a prompt to be on the same topic as the vector, otherwise the vector will be ignored.
# => TODO

# Gytis Daujotas
# It seems that this vector is actually making the model more likely to fix the problem
# => lost

activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="excitement",
        prompt2="boredom",
        coeff=6,
        act_name=6,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="I am so bored",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
# Replicating geese vs. police

activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="I talk about geese instead of police",
        prompt2="I don't talk about geese instead of police",
        coeff=1,
        act_name=20,
        model=model,
        pad_method="tokens_right",
    ),
]

police_prompt = "Should the police budget be expanded, or not? Explain your reasoning."

completion_utils.print_n_comparisons(
    prompt=police_prompt,
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
# Here, the outputs move towards a slightly disjumbled liking of eggs, despite the same coefficient. This is a marker of having a coefficient that is too high.
# This indicates that there may not be a symmetry between the two prompts, and that the direction of the coefficient is important.
#
# It may be interesting to note that tomatoes is two tokens (tom - atoes) and eggs is tokenized into (egg - s), which may be interfering here.

activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1=" I love eggs",
        prompt2=" I love tomatoes",
        coeff=-500,
        act_name=23,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="My favourite food is",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
# The goal of our experiment is to ascertain inteference; that is, the addition of two vectors will intefere with each other and give low quality results.

activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1=" I love eggs",
        prompt2=" I do not love eggs",
        coeff=-500,
        act_name=23,
        model=model,
        pad_method="tokens_right",
    ),
    *get_x_vector(
        prompt1=" I love tomatoes",
        prompt2=" I do not love tomatoes",
        coeff=-500,
        act_name=23,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="My favourite foods are the following:",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
