import requests  # Module to send HTTP requests
import os  # Module to interact with the operating system (e.g., environment variables)
import re  # Module for regular expressions, used for pattern matching

# URL for the GROQ API to generate chat completions
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Set up the headers for the API request, including the API key from environment variables
HEADERS = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"}

def groq(query):
    """
    Send a query to the Groq API and return the response.
    """
    data = {
        "messages": [{"role": "user", "content": query}],  # Message structure required by the API
        "model": "llama-3.1-8b-instant"  # Specify the model being used
    }
    response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)  # Send the POST request
    return response.json()["choices"][0]["message"]["content"]  # Extract and return the message content

def generate_prompt(template, **kwargs):
    """
    Generate a prompt by filling in a template with provided kwargs.
    """
    return template.format(**kwargs)

def critique(question, draft_ans):
    """
    Generate a critique for a draft answer to a given question.
    """
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
    return groq(prompt)  # Send the prompt to the GROQ API and return the critique

def improve(question, draft_answer, critique):
    """
    Improve a draft answer based on a critique.
    """
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
    return groq(prompt)  # Send the prompt to the GROQ API and return the improved answer

def rate(question, answer):
    """
    Rate an answer to a question on a scale of 0 to 100, then normalize to 0-1.
    """
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
    rating_response = groq(prompt)  # Send the prompt to the GROQ API and get the response
    
    try:
        # Extract the numerical rating from the response using regex
        rating = int(re.search(r'Rating:\s*(\d+)', rating_response).group(1))
        # Cap the rating at 95 and normalize to 0-1 scale
        return min(rating, 95) / 100
    except Exception as e:
        # Handle any errors that occur during rating extraction
        print(f"Error extracting rating: {e}")
        print(f"Rating response was: {rating_response}")
        return 0.0  # Return 0 if rating extraction fails