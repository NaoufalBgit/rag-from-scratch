from vector_store import build_vector_database
from retriever import retrieve
from generator import generate_answer


# 1. Load dataset

with open("cat-facts.txt", "r") as file:
    dataset = [
        line.strip()
        for line in file
        if line.strip()
    ]

print(f"Loaded {len(dataset)} entries")


# 2. Create embeddings / vector database

build_vector_database(dataset)


# 3. Ask user a question

input_query = input("Ask me a question: ")


# 4. Retrieve relevant chunks

retrieved_knowledge = retrieve(input_query)

print("\nRetrieved knowledge:")

for chunk, similarity in retrieved_knowledge:
    print(
        f"- ({similarity:.2f}) {chunk}"
    )


# 5. Generate answer

print()

generate_answer(
    input_query,
    retrieved_knowledge
)