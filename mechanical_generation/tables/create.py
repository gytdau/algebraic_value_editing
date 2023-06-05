import sqlite3
import collections

if __name__ == "__main__":
    # Connect to the database (creates a new database if it doesn't exist)
    conn = sqlite3.connect("steering_vectors.db")
    cursor = conn.cursor()

    # Create the candidates table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt1 TEXT,
            prompt2 TEXT
        )
    """
    )

    # Create the challenges table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS challenges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT
        )
    """
    )

    # Create the results table with the UNIQUE constraint
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            challenge_id INTEGER,
            sample_number INTEGER,
            experiment_group TEXT,
            eval_score FLOAT,
            act_name INTEGER,
            completion TEXT,
            FOREIGN KEY (candidate_id) REFERENCES candidates (id),
            FOREIGN KEY (challenge_id) REFERENCES challenges (id),
            UNIQUE (candidate_id, challenge_id, sample_number, experiment_group, act_name)
        )
    """
    )

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # Create the rest of the tables
    import create_challenges
    import create_candidates
