from embeddings import embed

VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = embed(chunk)
    VECTOR_DB.append((chunk, embedding))

def build_vector_database(dataset):
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk)
        print(f"Added chunk {i + 1}/{len(dataset)} to the database")