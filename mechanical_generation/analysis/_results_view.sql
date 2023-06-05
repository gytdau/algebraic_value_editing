SELECT s.id, s.candidate_id, s.challenge_id, s.sample_number, s.experiment_group,s.completion, cand.prompt1 AS candidate_prompt, chall.prompt AS challenge_prompt
FROM results AS s
INNER JOIN candidates AS cand ON s.candidate_id = cand.id
INNER JOIN challenges AS chall ON s.challenge_id = chall.id
