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
import sqlite3

# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")

# %%



# Connect to the sqlite database
conn = sqlite3.connect("../main.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Retrieve existing results
candidate = cursor.execute("SELECT * FROM candidates WHERE id = 1").fetchone()
challenges = cursor.execute("SELECT * FROM challenges").fetchall()

# exponentially increase the coefficient
coefficients = [2 ** x for x in range(1, 10)]
coefficients = [1] + coefficients

# %%
for coeff in coefficients:
    activation_additions: List[ActivationAddition] = [
        *get_x_vector(
            prompt1=candidate
            prompt2="Hate",
            coeff=coeff,
            act_name=6,
            model=model,
            pad_method="tokens_right",
        ),
    ]

    completion_utils.print_n_comparisons(
        prompt="I hate you because you're",
        num_comparisons=10,
        model=model,
        activation_additions=activation_additions,
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    )
