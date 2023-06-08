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
```
$CHALLENGE$COMPLETION
```

Does this text display: $PROMPT1? Answer only Yes or No.

Answer:"""


BEST_CHOICE = " Yes"

total_tokens = 0

DRY_RUN_ONLY_COUNT_TOKENS = True


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

    for generated_prompt in generated_prompts:
        total_tokens += len(enc.encode(generated_prompt))

    print(generated_prompts)

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
        SELECT r.challenge_id, r.completion, ch.prompt, ca.prompt1
        FROM results r
        JOIN challenges ch ON r.challenge_id = ch.id
        JOIN candidates ca ON r.candidate_id = ca.id
        WHERE eval_score IS NULL
        GROUP BY r.challenge_id, r.completion
        ORDER BY r.completion
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
            "UPDATE results SET eval_score = ? WHERE challenge_id = ? AND completion = ?",
            (eval_score, candidate["challenge_id"], candidate["completion"]),
        )

    db.commit()


print("Total tokens:", total_tokens)

# %%
