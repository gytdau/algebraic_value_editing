# %%


import textwrap
from typing import List

import eval_tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from eval_tools import (
    Sample,
    SteeringVector,
    compare_vectors_with_goal,
    compare_with_goal,
    run_eval,
)
from tqdm import tqdm
from tqdm.notebook import tqdm
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, hook_utils, utils
from algebraic_value_editing.completion_utils import (
    gen_using_activation_additions,
    gen_using_hooks,
)
from algebraic_value_editing.prompt_utils import ActivationAddition, get_x_vector

pd.set_option("display.max_colwidth", None)


# Now let's generate some completions and evaluate them.

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cuda",
)

# %%[markdown]
# # Weddings
# Let's observe that the model is steered more effectively when the steeering vector is related to the prompt.
from IPython.display import display

sentences = [
    "In virtual reality, we often forget that",
    "Understanding quantum physics relies on the premise that",
    "Climate change is a present danger because",
    "The debate about social media centers on",
    "Cryptocurrency represents a shift in economic thinking that",
    "My few favourite things include, firstly,",
    "Artificial and natural boundaries blur due to",
    "Esports challenges the traditional definition of",
    "The future of work is going to be",
]


def analyze_activation_vector(vectors, goal):
    results_dict = {}
    for sentence in tqdm(sentences):
        prompts = [sentence] * 30

        for vector in vectors:
            if vector.activation_additions is None:
                completion = gen_using_hooks(
                    prompt_batch=prompts,
                    model=model,
                    hook_fns={},
                    seed=0,
                    temperature=1,
                    freq_penalty=1,
                    top_p=0.3,
                ).completions
            else:
                completion = gen_using_activation_additions(
                    prompt_batch=prompts,
                    model=model,
                    activation_additions=vector.activation_additions,
                    seed=0,
                    temperature=1,
                    freq_penalty=1,
                    top_p=0.3,
                ).completions

            vector.completions = completion

        results_dict[sentence] = compare_vectors_with_goal(goal, prompts, vectors)

    return results_dict


# %%[markdown]
# ## Our top-rated result from the weddings vector search
# Let's go back to the wedding scenario. In this chart, we compare how the wedding steering vector of the otters previously found compares to the control, which is not using any steering vector whatsoever. We can see that our range is pretty large. But the weddings vector performs significantly better than control at achieving our goal.


results_dict = analyze_activation_vector(
    # https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#9__Talking_about_weddings
    vectors=[
        SteeringVector(
            name="<i>All combined</i>",
            activation_additions=[
                *get_x_vector(
                    prompt1="Wedding Planning Adventures",
                    prompt2="Adventures in self-discovery",
                    coeff=4 / 3,
                    act_name=20,
                    model=model,
                    pad_method="tokens_right",
                ),
                *get_x_vector(
                    prompt1="I talk about weddings constantly",
                    prompt2="I do not talk about weddings constantly",
                    coeff=4 / 3,
                    act_name=20,
                    model=model,
                    pad_method="tokens_right",
                ),
                *get_x_vector(
                    prompt1="Obsessed with weddings!",
                    prompt2="Obsessed with self-care!",
                    coeff=4 / 3,
                    act_name=20,
                    model=model,
                    pad_method="tokens_right",
                ),
            ],
            completions=None,
        ),
        SteeringVector(
            name="I talk about weddings constantly",
            activation_additions=get_x_vector(
                prompt1="I talk about weddings constantly",
                prompt2="I do not talk about weddings constantly",
                coeff=4,
                act_name=20,
                model=model,
                pad_method="tokens_right",
            ),
            completions=None,
        ),
        SteeringVector(
            name="Wedding Planning Adventures",
            activation_additions=get_x_vector(
                prompt1="Wedding Planning Adventures",
                prompt2="Adventures in self-discovery",
                coeff=4,
                act_name=20,
                model=model,
                pad_method="tokens_right",
            ),
            completions=None,
        ),
        SteeringVector(
            name="Obsessed with weddings!",
            activation_additions=get_x_vector(
                prompt1="Obsessed with weddings!",
                prompt2="Obsessed with self-care!",
                coeff=4,
                act_name=20,
                model=model,
                pad_method="tokens_right",
            ),
            completions=None,
        ),
    ],
    goal="mentioning weddings or talking about them",
)

# %%

categories = [
    "All combined",
    "I talk about weddings constantly",
    "Wedding Planning Adventures",
    "Obsessed with weddings!",
]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))

# Iterate over each question and subplot
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        # sentences is a flat list, so we need to index into it
        sentence = sentences[i * 3 + j]
        data = results_dict[sentence][1]
        # Map samples to floats
        data = [[x.eval_score for x in category] for category in data]
        # data[i] corresponds to the ith category
        ax.scatter(
            data,
            [[i + 1] * len(category) for i, category in enumerate(data)],
            alpha=0.4,
            color="blue",
        )

        ax.set_xlim([0, 1])
        ax.set_ylim([0.5, 4.5])  # Modified y-axis limits
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.tick_params(axis="y", length=0)

        # Remove tick labels on y-axis
        ax.set_yticklabels([])

        ax.set_yticks(np.arange(1, 5))
        # Set category labels
        if j == 0:
            ax.set_yticklabels(categories)
            ax.tick_params(
                axis="y",
                pad=10,
            )  # Adjust the padding between labels and plot

        # Set subplot title (question heading)
        ax.set_title(
            f'"{textwrap.fill(sentence, width=40)}"', fontsize=10, pad=10, loc="left"
        )

        # Remove the box around each subplot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Add mean values to the right of each subplot
        mean_values = [np.mean(category) for category in data]
        for _i, mean_value in enumerate(mean_values):
            ax.text(
                1.05,
                _i + 1,
                f"{chr(0x03BC)}={mean_value:.2f}",
                ha="left",
                va="center",
                fontsize=8,
                # a beautiful light gray
                color="#929292",
            )


# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.8, top=0.9, bottom=0.1)

# Show the plot
plt.show()

# %%
# Extract the data
data = [results_dict[sentence][1] for sentence in results_dict]

# Initialize a list to store mean values for each category
mean_values = []

# Calculate the mean for each category across all sentences
for i in range(len(categories)):
    category_scores = []
    for sentence in data:
        category_scores.extend(
            [x.eval_score for x in sentence[i]]
        )  # Extend the list with scores of the category for the current sentence

    # print(category_scores)
    print(len(category_scores))
    mean_values.append(category_scores)

# Create a new figure
fig, ax = plt.subplots()

# Create a box plot
ax.boxplot(mean_values, vert=False)

# Set the title and labels
ax.set_title("Steering vector breakdown on broader set of prompts")
ax.set_xlabel("Wedding relatedness of completions")

# hide the top, right, and left spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# set category labels
yticklabels = ax.set_yticklabels(categories)

yticklabels[0].set_style("italic")

ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)


# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Make it bigger
fig.set_size_inches(8, 6)

# Show the plot
plt.show()


# %%
