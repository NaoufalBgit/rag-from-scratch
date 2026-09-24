import ollama


LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"


def generate_answer(query, retrieved_knowledge):

    context = "\n".join(
        f"- {chunk}"
        for chunk, similarity in retrieved_knowledge
    )

    instruction_prompt = f"""
You are a helpful chatbot.

Use only the following context to answer the question.

If the answer cannot be found in the context,
say that you don't know.

Context:
{context}
"""

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": instruction_prompt
            },
            {
                "role": "user",
                "content": query
            },
        ],
        stream=True,
    )

    print("Chatbot response:")

    for chunk in stream:
        print(
            chunk["message"]["content"],
            end="",
            flush=True
        )