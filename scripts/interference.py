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

# One surprising fact: a high coefficient sometimes makes the model stop working entirely, while other times it continues to function seemingly unchanged.

#
activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="elf elf elf elf",
        prompt2="darth vader",
        coeff=-10000,
        act_name=20,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="I was walking to the woods and i saw a mythical",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%


activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="big",
        prompt2="small",
        coeff=3,
        act_name=3,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="The size of the dog was small",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
# %%
