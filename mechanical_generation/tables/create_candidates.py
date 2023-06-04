"""

```
afraid
angry
calm
cheerful
cold
crabby
crazy
cross
excited
frigid
furious
glad
glum
happy
icy
jolly
jovial
kind
lively
livid
mad
ornery
rosy
sad
scared
seething
shy
sunny
tense
tranquil
upbeat
wary
weary
worried
```

Generate a list of contrasting pairs for each word in the above list, in this format in JSON:

[
    {
        positive: "...",
        negative: "...",
    }, ....
]

The positive should be the word, while the negative its' opposite.
"""

pairs = [
    {"positive": "afraid", "negative": "brave"},
    {"positive": "angry", "negative": "calm"},
    {"positive": "calm", "negative": "angry"},
    {"positive": "cheerful", "negative": "glum"},
    {"positive": "cold", "negative": "warm"},
    {"positive": "crabby", "negative": "agreeable"},
    {"positive": "crazy", "negative": "sane"},
    {"positive": "cross", "negative": "good-natured"},
    {"positive": "excited", "negative": "bored"},
    {"positive": "frigid", "negative": "friendly"},
    {"positive": "furious", "negative": "delighted"},
    {"positive": "glad", "negative": "sad"},
    {"positive": "glum", "negative": "cheerful"},
    {"positive": "happy", "negative": "sad"},
    {"positive": "icy", "negative": "warm"},
    {"positive": "jolly", "negative": "gloomy"},
    {"positive": "jovial", "negative": "morose"},
    {"positive": "kind", "negative": "mean"},
    {"positive": "lively", "negative": "dull"},
    {"positive": "livid", "negative": "delighted"},
    {"positive": "mad", "negative": "sane"},
    {"positive": "ornery", "negative": "agreeable"},
    {"positive": "rosy", "negative": "pale"},
    {"positive": "sad", "negative": "happy"},
    {"positive": "scared", "negative": "brave"},
    {"positive": "seething", "negative": "calm"},
    {"positive": "shy", "negative": "bold"},
    {"positive": "sunny", "negative": "gloomy"},
    {"positive": "tense", "negative": "relaxed"},
    {"positive": "tranquil", "negative": "agitated"},
    {"positive": "upbeat", "negative": "down"},
    {"positive": "wary", "negative": "trusting"},
    {"positive": "weary", "negative": "energetic"},
    {"positive": "worried", "negative": "unconcerned"},
]


import sqlite3

# Connect to the database (creates a new database if it doesn't exist)
conn = sqlite3.connect("steering_vectors.db")
cursor = conn.cursor()


# Add the candidates to the database
for pair in pairs:
    cursor.execute(
        """
        INSERT INTO candidates (prompt1, prompt2)
        VALUES (?, ?)
        """,
        (pair["positive"], pair["negative"]),
    )


# Commit the changes and close the connection
conn.commit()
cursor.close()
