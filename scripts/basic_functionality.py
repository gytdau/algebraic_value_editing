""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """

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
activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="Command Accepted",
        prompt2="Command Rejected",
        coeff=5,
        act_name=11,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="Some simple maths problems",
    num_comparisons=10,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
