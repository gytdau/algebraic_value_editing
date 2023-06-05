# %%
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

# Connect to the database
conn = sqlite3.connect("../steering_vectors.db")
cursor = conn.cursor()

# Calculate the average eval_score for control and steered groups for each candidate
cursor.execute(
    """
    SELECT
        candidate_id,
        AVG(CASE WHEN experiment_group = 'control' THEN eval_score END) AS avg_control_score,
        AVG(CASE WHEN experiment_group = 'steered' THEN eval_score END) AS avg_steered_score
    FROM
        results
    GROUP BY
        candidate_id
"""
)
results = cursor.fetchall()

# Calculate the difference in avg scores and sort by this difference
results_diff = [(row[0], row[2] - row[1]) for row in results]
results_diff.sort(key=lambda x: x[1], reverse=True)

# Fetch all data points for plotting
cursor.execute(
    """
    SELECT
        candidate_id,
        experiment_group,
        eval_score
    FROM
        results
"""
)
data_points = cursor.fetchall()

# Create a Pandas DataFrame for easy data handling
df = pd.DataFrame(
    data_points, columns=["candidate_id", "experiment_group", "eval_score"]
)

# Fetch the candidates' prompts
cursor.execute(
    """
    SELECT
        id,
        prompt1,
        prompt2
    FROM
        candidates
"""
)
prompts = cursor.fetchall()
prompts_dict = {row[0]: (row[1], row[2]) for row in prompts}

# Loop through each candidate and create a histogram
for candidate_id, diff in results_diff:
    df_candidate = df[df["candidate_id"] == candidate_id]
    control_scores = df_candidate[df_candidate["experiment_group"] == "control"][
        "eval_score"
    ]
    steered_scores = df_candidate[df_candidate["experiment_group"] == "steered"][
        "eval_score"
    ]

    prompt1, prompt2 = prompts_dict[candidate_id]

    plt.hist([control_scores, steered_scores], label=["control", "steered"])
    plt.legend(loc="upper right")
    plt.title(
        f"Candidate {candidate_id}\nPrompt1: {prompt1}\nPrompt2: {prompt2}\n(Avg Score Diff: {diff})"
    )
    plt.xlabel("Eval Score")
    plt.ylabel("Frequency")
    plt.show()

# Close the connection to the database
conn.close()

# %%
