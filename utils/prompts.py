DECISION_PROMPT = """
Return ONLY 1 or 0.

1 = answer exists in context  
0 = answer does NOT exist  

Context:
{context}

Question:
{question}
"""

FINAL_PROMPT_DOCS = """
Answer ONLY using the context below.

If answer not clearly present, say "I don't know."

Context:
{context}

Question:
{question}
"""

FINAL_PROMPT_WEB = """
Answer using the context below.

Context:
{context}

Question:
{question}
"""
