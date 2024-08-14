import CIR
import MCTS
import pandas as pd
from datasets import load_dataset
import re

def extract_boxed_answer(answer):
    """
    Extracts the content within the last \boxed{} in the answer, handling nested braces.
    """
    pattern = re.compile(r'\\boxed{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)}')
    matches = pattern.findall(answer)
    
    if matches:
        return matches[-1]
    return None

def get_MATH_QA(row_number=None):
    # Load the dataset
    dataset = load_dataset("lighteval/MATH", 'all', split='test[:100]')

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)

    # If no row number is provided, ask the user to input a row number
    if row_number is None:
        row_number = int(input("Please enter the row number (0-99) you want to pick: "))

    # Ensure the row number is within the valid range
    if row_number < 0 or row_number >= len(df):
        raise ValueError("Row number must be between 0 and 99")

    # Get the selected row
    selected_row = df.iloc[row_number]

    # Extract the question and answer
    question = selected_row['problem']
    full_answer = selected_row['solution']
    short_answer = extract_boxed_answer(full_answer)

    return question, full_answer, short_answer

seed_ans = "I don't know."
# Fetch a question, full answer, and short answer using the existing function
question, full_answer, short_answer = get_MATH_QA()

# Print the fetched question and answers
print("\n**Fetched Question:**", question)

# Initialize and run MCTS with the fetched question
mcts = MCTS.MCTS(question, seed_ans, iterations=1)
best_answer = mcts.search()

# Compare the MCTS result to the actual short answer
print('\n')
print(f"\n**MCTS Best Answer**: {best_answer}")
print("\n**Ground Truth Answer:**", short_answer)
print("\n**Ground Truth Answer:**", full_answer)