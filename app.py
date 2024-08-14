import CIR
import MCTS
import pandas as pd
from datasets import load_dataset
import re

def extract_boxed_answer(answer):
    """
    Extracts the content within the last \boxed{} in the answer.
    """
    match = re.findall(r'\\boxed{([^{}]*)}', answer)
    return match[-1] if match else None

def get_MATH_QA(row_number=None):
    # Load and convert dataset to DataFrame
    df = load_dataset("lighteval/MATH", 'all', split='test[:100]').to_pandas()

    # Select or prompt for row number
    row_number = row_number if row_number is not None else int(input("Enter row number (0-99): "))
    if not 0 <= row_number < len(df):
        raise ValueError("Row number must be between 0 and 99")

    # Extract question, full answer, and short answer
    row = df.iloc[row_number]
    return row['problem'], row['solution'], extract_boxed_answer(row['solution'])

# Fetch a question, full answer, and short answer
question, full_answer, short_answer = get_MATH_QA()

# Run MCTS
best_answer = MCTS.MCTS(question, "I don't know.", iterations=1).search()

# Print results
print(f"**Fetched Question:** {question}")
print(f"**MCTS Best Answer**: {best_answer}")
print(f"**Ground Truth Answer:** {short_answer}")
print(f"**Full Ground Truth Answer:** {full_answer}")