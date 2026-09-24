from embeddings import embed
from vector_store import VECTOR_DB

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
    query_embedding = embed(query)

    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        similarities.append(
            (chunk, similarity)
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_n]