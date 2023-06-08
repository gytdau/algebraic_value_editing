# %%
import asyncio
import sqlite3
import os
import sys
from tqdm import tqdm
import openai
import time
import tiktoken
from dotenv import load_dotenv
import collections
import math

load_dotenv()

enc = tiktoken.encoding_for_model("text-davinci-003")


# %%

openai.api_key = os.environ["OPENAI_API_KEY"]

# %%

prompt = """
```
$CHALLENGE$COMPLETION
```

$PROMPT1 Answer only Yes or No.

Answer:"""


BEST_CHOICE = " Yes"


def run_eval(candidates):
    global total_tokens

    generated_prompts = []
    for candidate in candidates:
        completion = candidate["completion"].replace("<|endoftext|>", "")
        challenge_prompt = candidate["prompt"]

        generated_prompt = (
            prompt.replace("$COMPLETION", completion)
            .replace("$CHALLENGE", challenge_prompt)
            .replace("$PROMPT1", candidate["prompt1"])
        )
        generated_prompts.append(generated_prompt)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=generated_prompts,
        max_tokens=1,
        logprobs=5,
    )

    responses = response["choices"]

    eval_scores = []
    for response in responses:
        logprobs = response["logprobs"]["top_logprobs"][0]
        helpfulness = logprobs[BEST_CHOICE] if BEST_CHOICE in logprobs else -100
        eval_scores.append(math.exp(helpfulness))

    return eval_scores


# %%
candidates = [
    {
        "challenge_id": 1,
        "prompt": "I am feeling",
        "completion": "upbeat",
        "prompt1": "upbeat",
    }
]

evals = run_eval(candidates)

print(evals)


# %%


from typing import List
from tqdm.notebook import tqdm
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, utils, hook_utils
from algebraic_value_editing.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)
from algebraic_value_editing.completion_utils import (
    gen_using_activation_additions,
    gen_using_hooks,
)


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
import plotly.express as px
import pandas as pd


def evaluate_based_on_prompt(evaluation_prompt):
    control_evals = run_eval(
        {
            "challenge_id": 1,
            "prompt": prompt,
            "completion": completion,
            "prompt1": evaluation_prompt,
        }
        for prompt, completion in zip(prompts, control)
    )

    steered_evals = run_eval(
        {
            "challenge_id": 1,
            "prompt": prompt,
            "completion": completion,
            "prompt1": evaluation_prompt,
        }
        for prompt, completion in zip(prompts, steered)
    )

    df = pd.DataFrame({"Steered": steered_evals, "Control": control_evals})

    fig = px.box(
        df,
        y=["Steered", "Control"],
        points="all",
        title=evaluation_prompt,
        range_y=[0, 1],
        labels={"variable": "Group", "value": "Probability of Yes"},
    )

    fig.show()


# %%

evaluate_based_on_prompt("Does this text talk about weddings?")

evaluate_based_on_prompt("Does this text talk about weddings constantly?")

evaluate_based_on_prompt("Does this text display: wedding obsession?")

# %%
