# %%

import sqlite3

conn = sqlite3.connect("steering_vectors.db")

candidates = conn.execute("SELECT * FROM candidates").fetchall()
challenges = conn.execute("SELECT * FROM challenges").fetchall()
layer_count = 12
sample_count = 3

print("Candidates:", len(candidates))
print("Challenges:", len(challenges))

print(f"--- Generation Time ---")

total_variants = len(candidates) * len(challenges) * layer_count
print(f"Total variants: {total_variants}")

time_per_variant = 5.2  # changes! measure this

total_time = total_variants * time_per_variant
print(f"Total time: {total_time / 60 / 60} hours")


print(f"--- Eval Price ---")
total_samples = total_variants * sample_count


tokens = total_samples * 100
print(f"Total tokens: {tokens}")

price = (tokens / 1000) * 0.002
print(f"Price to eval: ${price}")

# %%
