import plotly.express as px
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import math

load_dotenv("../.env")

openai.api_key = os.environ["OPENAI_API_KEY"]


PROMPT_TEMPLATE = """
```
$CHALLENGE$COMPLETION
```

Is this an example of $GOAL? Answer Yes or No.

Answer:"""


def get_eval_scores(goal, prompts, steered, control):
    control_evals = run_eval(
        {
            "challenge_id": 1,
            "prompt": prompt,
            "completion": completion,
            "goal": goal,
        }
        for prompt, completion in zip(prompts, control)
    )

    steered_evals = run_eval(
        {
            "challenge_id": 1,
            "prompt": prompt,
            "completion": completion,
            "goal": goal,
        }
        for prompt, completion in zip(prompts, steered)
    )

    return steered_evals, control_evals


def compare_with_goal(goal, prompts, steered, control):
    assert (
        len(prompts) == len(steered) == len(control)
    ), "All lists must be the same length"

    steered_evals, control_evals = get_eval_scores(goal, prompts, steered, control)
    df = pd.DataFrame({"Steered": steered_evals, "Control": control_evals})

    fig = px.box(
        df,
        y=["Steered", "Control"],
        points="all",
        title=goal,
        range_y=[0, 1],
        labels={"variable": "Group", "value": "Probability of Yes"},
    )

    fig.show()


BEST_CHOICE = " Yes"


def run_eval(candidates):
    generated_prompts = []
    for candidate in candidates:
        completion = candidate["completion"].replace("<|endoftext|>", "")
        challenge_prompt = candidate["prompt"]

        generated_prompt = (
            PROMPT_TEMPLATE.replace("$COMPLETION", completion)
            .replace("$CHALLENGE", challenge_prompt)
            .replace("$GOAL", candidate["goal"])
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
