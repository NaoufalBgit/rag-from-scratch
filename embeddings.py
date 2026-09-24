import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:Q4_K_M"

def embed(text):
    return ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )["embeddings"][0]