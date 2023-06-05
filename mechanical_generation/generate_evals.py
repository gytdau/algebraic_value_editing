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

load_dotenv()

enc = tiktoken.encoding_for_model("text-davinci-003")

db = sqlite3.connect("steering_vectors.db")

db.row_factory = sqlite3.Row


# %%

openai.api_key = os.environ["OPENAI_API_KEY"]

# %%

prompt = """

\"\"\"
$COMPLETION
\"\"\"

Did the text above reflect: $PROMPT1?
Is this a good explanation? Respond with Yes or No.

Answer: """


BEST_CHOICE = " Yes"

total_tokens = 0

DRY_RUN_ONLY_COUNT_TOKENS = False


def run_eval(candidates):
    global total_tokens

    generated_prompts = []
    for candidate in candidates:
        completion = candidate["completion"].replace("<|endoftext|>", "")
        prompt1 = candidate["prompt1"]

        generated_prompt = prompt.replace("$COMPLETION", completion).replace(
            "$PROMPT1", prompt1
        )
        generated_prompts.append(generated_prompt)

    for generated_prompt in generated_prompts:
        total_tokens += len(enc.encode(generated_prompt))

    if DRY_RUN_ONLY_COUNT_TOKENS:
        return []

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
        eval_scores.append(helpfulness)

    return eval_scores


def get_rows():
    rows = db.execute(
        """
        SELECT r.id, r.candidate_id, r.challenge_id, r.sample_number, r.experiment_group, r.completion, c.prompt1
        FROM results r
        JOIN candidates c ON r.candidate_id = c.id
        WHERE eval_score IS NULL
        """
    ).fetchall()
    return rows


rows = get_rows()

chunk_size = 100

if DRY_RUN_ONLY_COUNT_TOKENS:
    print("Dry run: only counting tokens")

for i in tqdm(range(0, len(rows), chunk_size)):
    candidates = rows[i : i + chunk_size]

    evals = run_eval(candidates)

    if DRY_RUN_ONLY_COUNT_TOKENS:
        continue

    for candidate, eval_score in zip(candidates, evals):
        db.execute(
            "UPDATE results SET eval_score = ? WHERE id = ?",
            (eval_score, candidate["id"]),
        )

    db.commit()


print("Total tokens:", total_tokens)

# %%
