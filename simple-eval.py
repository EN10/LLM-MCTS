import requests
import os
import pandas as pd
from datasets import load_dataset

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"}

def groq(query):
    data = {
        "messages": [{"role": "user", "content": query}],
        "model": "llama-3.1-8b-instant"
    }
    response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
    return response.json()["choices"][0]["message"]["content"]

# Load and convert dataset [:10] rows to DataFrame
df = load_dataset("lighteval/MATH", 'all', split='test[:10]').to_pandas()
# Extract question [5] and answer
row = df.iloc[5]

print(row['problem'])
print('---')
print(groq(row['problem']))
print('---')
print(row['solution'])
