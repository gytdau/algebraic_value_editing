import os
import math
import pandas as pd
import plotly.express as px
import openai
from dataclasses import dataclass
from typing import List, Dict
from dotenv import load_dotenv


import os

load_dotenv(".env")


openai.api_key = os.getenv("OPENAI_API_KEY")

BEST_CHOICE = " Yes"


@dataclass
class Sample:
    prompt: str
    completion: str
    goal: str


def generate_prompt(sample: Sample) -> str:
    """Returns a formatted prompt."""
    PROMPT_TEMPLATE = f"""
    ```
    {sample.prompt}{sample.completion}
    ```

    Is this an example of {sample.goal}? Answer Yes or No.

    Answer:"""
    return PROMPT_TEMPLATE


def calculate_score(responses: Dict) -> List[float]:
    """Returns calculated scores."""
    eval_scores = []
    for response in responses:
        logprobs = response["logprobs"]["top_logprobs"][0]
        helpfulness = logprobs.get(BEST_CHOICE, -100)
        eval_scores.append(math.exp(helpfulness))
    return eval_scores


def run_eval(samples: List[Sample]) -> List[float]:
    """Evaluates samples and returns scores."""
    generated_prompts = [generate_prompt(sample) for sample in samples]
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=generated_prompts,
        max_tokens=1,
        logprobs=5,
    )
    responses = response["choices"]
    return calculate_score(responses)


def compare_with_goal(
    goal: str, prompts: List[str], steered: List[str], control: List[str]
) -> None:
    """Compares steered and control samples with the given goal and shows a plot."""
    assert (
        len(prompts) == len(steered) == len(control)
    ), "All lists must be the same length"

    control_samples = [
        Sample(prompt=prompt, completion=completion, goal=goal)
        for prompt, completion in zip(prompts, control)
    ]
    steered_samples = [
        Sample(prompt=prompt, completion=completion, goal=goal)
        for prompt, completion in zip(prompts, steered)
    ]

    control_evals = run_eval(control_samples)
    steered_evals = run_eval(steered_samples)

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
