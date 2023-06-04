-- SQLite
    SELECT r.id, c.prompt1, c.prompt2, ch.prompt AS challenge_prompt, r.completion, r.eval_score
    FROM results r
    JOIN candidates c ON r.candidate_id = c.id
    JOIN challenges ch ON r.challenge_id = ch.id
    WHERE r.experiment_group = 'steered'
