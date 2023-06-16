# %%
import matplotlib.pyplot as plt

# Data
categories = [
    "Category 1: This is a long sentence describing category 1",
    "Category 2: Another long sentence for category 2",
    "Category 3: This sentence describes category 3",
    "Category 4: Yet another long sentence for category 4",
]
values = [0.3, 0.6, 0.8, 0.5]

# Create a horizontal scatter plot
fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed
ax.scatter(values, range(len(categories)), color="b", s=100)

# Set the y-ticks and labels
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)

# Set the x-axis limits
ax.set_xlim(0, 1)

# Add gridlines
ax.grid(True, linestyle="--", alpha=0.5)

# Add labels and title
ax.set_xlabel("Values")
ax.set_ylabel("Categories")
ax.set_title("Horizontal Scatter Plot with Long Labels")

# Show the plot
plt.tight_layout()  # Ensures labels are not cut off
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
import matplotlib.pyplot as plt
import numpy as np

# Data
categories = [
    "Category A - This is a long label",
    "Category B - Another long label",
    "Category C - A sentence as a label",
    "Category D - One more lengthy label",
]
x_values = [0, 1]

# Generate random data for scatter plots
np.random.seed(42)
dataset = [np.random.rand(4) for _ in range(9)]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))

# Iterate over each question and subplot
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        data = dataset[i * 3 + j]
        # Scatter plot
        ax.scatter(data, np.arange(1, 5), color="b", s=100)  # Modified y-axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0.5, 4.5])  # Modified y-axis limits
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.tick_params(axis="y", length=0)

        # Remove tick labels on y-axis
        ax.set_yticklabels([])

        # Set category labels
        if j == 0:
            ax.set_yticks(np.arange(1, 5))
            ax.set_yticklabels(categories)
            ax.tick_params(
                axis="y", pad=10
            )  # Adjust the padding between labels and plot

        # Set subplot title (question heading)
        ax.set_title(f"Question {i * 3 + j + 1}", loc="left")

        # Remove the box around each subplot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=1)

# Show the plot
plt.show()


# %%
