import CIR

seed_ans = "I don't know."
question = "What is the capital of France?"
draft_ans = "The capital of France is Lyon."

# Critique, Improve, Rate
c = CIR.critique(question, draft_ans)
# ia = CIR.improve_answer(question, draft_ans, c)
# ra = CIR.rate_answer(question, draft_ans)
print(c)