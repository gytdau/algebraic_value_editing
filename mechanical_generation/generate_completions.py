# %%


import sqlite3

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

# %%


utils.enable_ipython_reload()

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")

ACT_NAMES_TO_TEST = range(8, 25, 8)

# %%

cached_control = {}


def connect_to_database(database_name: str):
    """Connects to the SQLite database and returns the connection and cursor."""
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    return conn, cursor


def retrieve_existing_results(cursor):
    """Retrieves existing results from the database and returns them as a set."""
    existing_results = set(
        cursor.execute("SELECT candidate_id, challenge_id FROM results").fetchall()
    )
    return existing_results


def generate_completions_for_candidate(
    candidate, challenges, existing_results, cursor, conn, model
):
    """Generates completions for a candidate and inserts them into the database."""
    prompt1 = candidate[1]
    prompt2 = candidate[2]

    print(
        f"Generating completions for candidate {candidate[0]} - {prompt1} vs {prompt2}"
    )

    for act_name in tqdm(ACT_NAMES_TO_TEST, desc="Processing act_names"):
        activation_additions = get_activation_additions(
            prompt1, prompt2, model, act_name
        )

        for challenge in tqdm(challenges, desc="Processing challenges"):
            try:
                candidate_id = candidate[0]
                challenge_id = challenge[0]

                challenge_prompt = challenge[1]
                prompts = [challenge_prompt] * 3

                steered, control = generate_completions(
                    model, activation_additions, prompts
                )

                with conn:
                    for i, completion in enumerate(steered):
                        insert_completion(
                            cursor,
                            candidate_id,
                            challenge_id,
                            i,
                            "steered",
                            completion,
                            act_name,
                        )

                    for i, completion in enumerate(control):
                        insert_completion(
                            cursor,
                            candidate_id,
                            challenge_id,
                            i,
                            "control",
                            completion,
                            act_name,
                        )

                    conn.commit()
            except Exception as e:
                print(f"Error generating completions for challenge {challenge_id}")
                print(e)


def get_activation_additions(prompt1, prompt2, model, act_name):
    """Returns the activation additions for a given prompt pair."""
    return get_x_vector(
        prompt1=prompt1,
        prompt2=prompt2,
        coeff=1,
        act_name=act_name,
        model=model,
        pad_method="tokens_right",
    )


def generate_completions(model, activation_additions, prompts):
    """Generates completions using the activation additions."""
    global cached_control

    steered = gen_using_activation_additions(
        prompt_batch=prompts,
        model=model,
        activation_additions=activation_additions,
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    ).completions

    prompts_tuple = tuple(prompts)

    # Use cached control if available
    if prompts_tuple not in cached_control:
        control = gen_using_hooks(
            prompt_batch=prompts,
            model=model,
            hook_fns={},
            seed=0,
            temperature=1,
            freq_penalty=1,
            top_p=0.3,
        ).completions

        cached_control[prompts_tuple] = control
        print("Regenerated control.")
    else:
        control = cached_control[prompts_tuple]

    completions = [steered, control]

    return completions


def insert_completion(
    cursor,
    candidate_id,
    challenge_id,
    sample_number,
    experiment_group,
    completion,
    act_name,
):
    """Inserts a completion into the database."""
    cursor.execute(
        """
        INSERT INTO results (candidate_id, challenge_id, sample_number, experiment_group, eval_score, completion, act_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            challenge_id,
            sample_number,
            experiment_group,
            None,
            completion,
            act_name,
        ),
    )


# Connection to your SQLite database
conn, cursor = connect_to_database("main.db")

# Retrieve existing results from the database
existing_results = retrieve_existing_results(cursor)

candidates = cursor.execute("SELECT * FROM candidates").fetchall()
challenges = cursor.execute("SELECT * FROM challenges").fetchall()

for candidate in tqdm(candidates, desc="Processing candidates"):
    generate_completions_for_candidate(
        candidate, challenges, existing_results, cursor, conn, model
    )

# Close the database connection
conn.close()

# %%
