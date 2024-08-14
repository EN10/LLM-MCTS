import requests
import os
import re

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"}

def groq(query):
    data = {
        "messages": [{"role": "user", "content": query}],
        "model": "llama-3.1-8b-instant"
    }
    response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
    return response.json()["choices"][0]["message"]["content"]

def generate_prompt(template, **kwargs):
    return template.format(**kwargs)

def critique(question, draft_ans):
    prompt = generate_prompt(
        "Question: {question}\n"
        "Draft Answer: {draft_ans}\n"
        "Do a careful assessment of whether the answer is correct or not, and why. "
        "Consider multiple ways of verifying the correctness of the answer. "
        "Point out every flaw and hold the draft answer to a high standard. "
        "Provide specific recommendations to improve the answer. "
        "Think step by step. "
        "Do not provide a revised answer.",
        question=question, draft_ans=draft_ans
    )
    return groq(prompt)

def improve_answer(question, draft_answer, critique):
    prompt = generate_prompt(
        "Question: {question}\n"
        "Draft Answer: {draft_answer}\n"
        "Critique: {critique}\n\n"
        "Please improve the draft answer based on the critique. Follow this format:\n"
        "Reasoning Process: <step-by-step reasoning process>\n"
        "Verification: <verification of the facts>\n"
        "Final Answer: <the improved and verified answer>\n",
        question=question, draft_answer=draft_answer, critique=critique
    )
    return groq(prompt)

def rate_answer(question, answer):
    prompt = generate_prompt(
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "As an expert on this topic, please provide a detailed critique of the answer, pointing out every flaw. "
        "Provide only a critique, not a suggested answer. "
        "Then, rate the answer on a scale of 0 to 100. "
        "The response should be in the following format:\n"
        "Critique: <detailed critique>\n"
        "Rating: <rating>",
        question=question, answer=answer
    )
    rating_response = groq(prompt)
    
    try:
        rating = int(re.search(r'Rating:\s*(\d+)', rating_response).group(1))
        return min(rating, 95) / 100
    except Exception as e:
        print(f"Error extracting rating: {e}")
        print(f"Rating response was: {rating_response}")
        return 0.0