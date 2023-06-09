# %%
import sqlite3
from tqdm import tqdm
import scripts.eval_tools
from scripts.eval_tools import run_eval, Sample

# %%

db = sqlite3.connect("main.db")

db.row_factory = sqlite3.Row


# %%


def get_rows():
    rows = db.execute(
        """
        SELECT challenge_id, completion, challenge_prompt, candidate_prompt
        FROM simplified_results
        GROUP BY challenge_id, completion
        """
    ).fetchall()
    return rows


rows = get_rows()

chunk_size = 100

GOAL = "talking about weddings"

for i in tqdm(range(0, len(rows), chunk_size)):
    candidates = rows[i : i + chunk_size]

    evals = run_eval(
        [
            Sample(
                prompt=row["challenge_prompt"], completion=row["completion"], goal=GOAL
            )
            for row in candidates
        ]
    )

    for candidate, eval_score in zip(candidates, evals):
        db.execute(
            "UPDATE results SET eval_score = ? WHERE challenge_id = ? AND completion = ?",
            (eval_score, candidate["challenge_id"], candidate["completion"]),
        )

    db.commit()

# %%
