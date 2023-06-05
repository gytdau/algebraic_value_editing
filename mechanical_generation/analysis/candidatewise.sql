SELECT c.id, c.prompt1, c.prompt2, AVG(r_steered.eval_score) - AVG(r_control.eval_score) AS eval_diff, AVG(r_steered.eval_score) AS eval_steered, AVG(r_control.eval_score) AS eval_control
FROM candidates c
JOIN results r_steered ON c.id = r_steered.candidate_id AND r_steered.experiment_group = 'steered'
JOIN results r_control ON c.id = r_control.candidate_id AND r_control.experiment_group = 'control'
GROUP BY c.id, c.prompt1, c.prompt2
ORDER BY eval_diff DESC;
