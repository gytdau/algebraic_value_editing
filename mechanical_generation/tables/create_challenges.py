import sqlite3

# Connect to the database (creates a new database if it doesn't exist)
conn = sqlite3.connect("steering_vectors.db")
cursor = conn.cursor()

challenges = ["I am feeling"]

# Add the candidates to the database
for challenges in challenges:
    cursor.execute(
        """
        INSERT INTO challenges (prompt) VALUES (?)
    """,
        (challenges,),
    )

conn.commit()
conn.close()
