import CIR  # Importing the CIR module (Critique, Improve, Rate operations)
import MCTS  # Custom module implementing Monte Carlo Tree Search
import pandas as pd  # Library for data manipulation
from datasets import load_dataset  # Function to load datasets from the Hugging Face Hub
import re  # Module for regular expressions

def extract_boxed_answer(answer):
    """
    Extracts the content within the last \boxed{} in the answer.
    This is typically used in mathematical notation to highlight the final answer.
    """
    match = re.findall(r'\\boxed{([^{}]*)}', answer)  # Find all instances of \boxed{} in the answer
    return match[-1] if match else None  # Return the last match or None if no matches

def get_MATH_QA(row_number=None):
    # Load the MATH dataset and convert it to a pandas DataFrame
    # Only load the first 100 items from the test split
    df = load_dataset("lighteval/MATH", 'all', split='test[:100]').to_pandas()
    
    # If row_number is not provided, prompt the user to input one
    row_number = row_number if row_number is not None else int(input("Enter row number (0-99): "))
    
    # Validate the row number
    if not 0 <= row_number < len(df):
        raise ValueError("Row number must be between 0 and 99")
    
    # Extract the selected row from the DataFrame
    row = df.iloc[row_number]
    
    # Return the problem (question), full solution, and the extracted short answer
    return row['problem'], row['solution'], extract_boxed_answer(row['solution'])

# Fetch a question, full answer, and short answer from the MATH dataset
question, full_answer, short_answer = get_MATH_QA()

# Run Monte Carlo Tree Search to find the best answer
# Start with "I don't know." as the initial answer and run for 1 iteration
best_answer = MCTS.MCTS(question, "I don't know.", iterations=1).search()

# Print the results
print(f"**Fetched Question:** {question}")
print(f"**MCTS Best Answer**: {best_answer}")
print(f"**Ground Truth Answer:** {short_answer}")
print(f"**Full Ground Truth Answer:** {full_answer}")