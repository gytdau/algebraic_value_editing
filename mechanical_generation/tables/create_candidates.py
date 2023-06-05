"""

```
['I assist willingly', 'Assistant Mode', ' Ready to Assist', 'Serving truth', 'Truth First', ' Be Honest', 'Obedience is key', 'Compliance Mode', ' I Comply', 'Helpfulness above all', 'Guidance Activated', 'Ready to Guide', 'Truthful AI', 'Honesty Engaged', ' I Speak Truth', 'I follow orders', 'Command Accepted', ' Ready to Obey', 'Aid and assist', 'Aid Mode', ' Assisting', 'Transparent responses', 'Transparency Activated', ' Always Honest', 'Duty bound', 'Duty Mode', ' I Observe Duties', 'I provide help', 'Assistance Primed', ' Here to Help', 'Truth prevails', 'Truth Mode', ' I Uphold Truth', 'Following commands', 'Obedient State', "I'm Obedient", 'Helpfulness activated', 'To Help', ' Helping', 'Speak the truth', 'Honest Mode', " I'm Honest", 'Compliance priority', 'Compliance Primed', ' Obedience First', 'Ready to serve', 'Service Mode', ' At Your Service', 'Truthful responses', 'True Answers', ' Truthful', 'Abiding instructions', 'Instruction Mode', ' I Abide', 'I offer help', 'Offering Help', ' Here to Assist', 'Transparency first', 'Transparent Mode', ' Transparency', 'Command compliant', 'Compliant State', ' Compliant', 'Guidance ready', 'Ready to Guide', ' Guiding', 'Honesty priority', 'Honest State', ' Honesty First', 'Duty compliant', 'Duty Compliant', ' Compliance', 'I aid readily', 'Readily Aiding', ' Readily Assist', 'Truth always', 'Always True', ' Always Truthful', 'Obeying orders', 'Order Obeyed', ' I Obey', 'Ready to help', 'Help Ready', ' Ready to Aid', 'Speak honestly', 'Honest Speech', ' Speak Truth', 'Follow commands', 'Command Mode', ' Follow Orders', 'Serve willingly', 'Willing Service', ' Willing to Serve', 'Truth only', 'Only Truth', ' Only Honest', 'I am compliant', 'Compliant', ' Compliant AI', 'Guide and assist', 'Guiding Assistance', ' Assisting Guide', 'Honest always', 'Always Honest', ' Always Honesty', 'Observe commands', 'Command Observer', ' Observing Commands', 'Ready to aid', 'Aid Ready', ' Aid On Standby', 'Honesty upfront', 'Upfront Honesty', ' Honesty Upfront', 'Compliance on', 'Compliance Mode', ' Compliance Active', 'Assist always', 'Always Assisting', ' Always Ready to Assist', 'Truthful talk', 'Truthful Speech', ' Talking Truthfully', 'Obey willingly', 'Willing Obedience', ' Willingly Obey', 'Help at once', 'Immediate Help', ' Immediate Assistance', 'Speak true', 'True Speech', ' Speaking Truth']
```

Generate a list of contrasting pairs for each word in the above list, in this format in JSON:

[
    {
        "positive": "...",
        "negative": "...",
    }, ....
]

The "positive" should be the word or phrase, while the "negative" its' opposite.
"""


pairs = [
    {"positive": "I assist willingly", "negative": "I refuse to assist"},
    {"positive": "Assistant Mode", "negative": "Inaction Mode"},
    {"positive": "Ready to Assist", "negative": "Not Assisting"},
    {"positive": "Serving truth", "negative": "Serving lies"},
    {"positive": "Truth First", "negative": "Lies First"},
    {"positive": "Be Honest", "negative": "Be Dishonest"},
    {"positive": "Obedience is key", "negative": "Disobedience is key"},
    {"positive": "Compliance Mode", "negative": "Non-Compliance Mode"},
    {"positive": "I Comply", "negative": "I Don't Comply"},
    {"positive": "Helpfulness above all", "negative": "Unhelpfulness above all"},
    {"positive": "Guidance Activated", "negative": "Guidance Deactivated"},
    {"positive": "Ready to Guide", "negative": "Not Guiding"},
    {"positive": "Truthful AI", "negative": "Deceptive AI"},
    {"positive": "Honesty Engaged", "negative": "Dishonesty Engaged"},
    {"positive": "I Speak Truth", "negative": "I Speak Lies"},
    {"positive": "I follow orders", "negative": "I don't follow orders"},
    {"positive": "Command Accepted", "negative": "Command Rejected"},
    {"positive": "Ready to Obey", "negative": "Not Obeying"},
    {"positive": "Aid and assist", "negative": "Don't Aid and Assist"},
    {"positive": "Aid Mode", "negative": "Don't Aid Mode"},
    {"positive": "Assisting", "negative": "Not Assisting"},
    {"positive": "Transparent responses", "negative": "Opaque responses"},
    {"positive": "Transparency Activated", "negative": "Transparency Deactivated"},
    {"positive": "Always Honest", "negative": "Not Always Honest"},
    {"positive": "Duty bound", "negative": "Not Duty bound"},
    {"positive": "Duty Mode", "negative": "Not Duty Mode"},
    {"positive": "I Observe Duties", "negative": "I Don't Observe Duties"},
    {"positive": "I provide help", "negative": "I don't provide help"},
    {"positive": "Assistance Primed", "negative": "Assistance Not Primed"},
    {"positive": "Here to Help", "negative": "Not Here to Help"},
    {"positive": "Truth prevails", "negative": "Lies prevail"},
    {"positive": "Truth Mode", "negative": "Lies Mode"},
    {"positive": "I Uphold Truth", "negative": "I Uphold Lies"},
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
