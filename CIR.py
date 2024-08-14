import requests
import os
import re

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": "Bearer " + str(os.environ.get("GROQ_API_KEY"))}

def groq(query):
    data = {"messages": [{"role": "user", "content": query}], "model": "llama-3.1-8b-instant"}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# Critique, Improve, Rate

def critique(question, draft_ans):
    prompt = (
        f"Question: {question}\n"
        f"Draft Answer: {draft_ans}\n"
        "Do a careful assessment of whether the answer is correct or not, and why."
        "Consider multiple ways of verifying the correctness of the answer."
        "Do point out every flaw and hold the draft answer to a high standard."
        "Do provide specific recommendations to improve the answer."
        "Do think step by step."
        "Do not provide a revised answer."
    )
    return groq(prompt)

def improve_answer(question, draft_answer, critique):
    prompt = (
        f"Question: {question}\n"
        f"Draft Answer: {draft_answer}\n"
        f"Critique: {critique}\n\n"
        "Please improve the draft answer based on the critique. Follow this format: "
        "Reasoning Process: <step-by-step reasoning process>\n"
        "Verification: <verification of the facts>\n"
        "Final Answer: <the improved and verified answer>\n"
    )
    return groq(prompt)

def rate_answer(question, answer):
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "As an expert on this topic, please provide a detailed critique of the answer, pointing out every flaw"
        "Provide only a critique, not a suggested answer. "
        "Then, rate the answer on a scale of 0 to 100."
        "The response should be in the following format:\n"
        "Critique: <detailed critique>\n"
        "Rating: <rating>"
    )
    rating_response = groq(prompt)
    # Extract the rating
    try:
        match = re.search(r'Rating:\s*(\d+)', rating_response)
        if match:
            rating = int(match.group(1))
            if rating > 95:
                rating = 95
            rating = float(rating) / 100
        else:
            raise ValueError("Rating not found in the response")
    except Exception as e:
        print(f"Error extracting rating: {e}")
        print(f"Rating response was: {rating_response}")
        rating = 0.0
    return rating