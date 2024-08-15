# from datasets import load_dataset

# # Load the dataset
# dataset = load_dataset("nikesh66/Slang-Dataset")

# # Print the dataset to see its structure
# print(dataset)

import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("nikesh66/Slang-Dataset")

# Convert to DataFrame and save as CSV
df = dataset["train"].to_pandas()  # or another split like "test"
df.to_csv("Sentimental-Analysis in Healthcare\slang_dataset.csv", index=False)

