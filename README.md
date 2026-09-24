# RAG From Scratch with Ollama

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

A lightweight **Retrieval-Augmented Generation (RAG)** system built from scratch with Python and Ollama.

The goal of this project is to understand the core mechanisms behind RAG systems without relying on high-level frameworks such as LangChain or LlamaIndex.

The current implementation uses:

* local embeddings
* an in-memory vector store
* cosine similarity
* Top-K semantic retrieval
* local LLM generation with Ollama

---

## Architecture

```text
Dataset
   ↓
Chunks
   ↓
Embedding Model
   ↓
Vector Store
   ↓
User Query
   ↓
Query Embedding
   ↓
Cosine Similarity
   ↓
Top-K Relevant Chunks
   ↓
Prompt Augmentation
   ↓
LLM
   ↓
Generated Answer
```

---

## Features

* Fully local RAG pipeline
* Embedding generation with Ollama
* In-memory vector store
* Semantic search
* Cosine similarity
* Top-K retrieval
* Context-grounded generation
* Streaming LLM responses
* Modular Python architecture
* No LangChain or LlamaIndex

---

## Project Structure

```text
rag-from-scratch/
├── cat-facts.txt
├── embeddings.py
├── vector_store.py
├── retriever.py
├── generator.py
├── main.py
├── requirements.txt
└── README.md
```

### `embeddings.py`

Handles text embedding generation.

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

### `vector_store.py`

Stores each chunk with its corresponding embedding.

```python
VECTOR_DB = [
    (chunk, embedding),
    ...
]
```

### `retriever.py`

Handles semantic retrieval.

```text
User Query
    ↓
Query Embedding
    ↓
Compare Against Stored Embeddings
    ↓
Cosine Similarity
    ↓
Sort by Similarity
    ↓
Top-K Chunks
```

### `generator.py`

Builds the prompt from retrieved chunks and sends it to the language model.

The LLM is instructed to answer only using the retrieved context.

### `main.py`

Orchestrates the full pipeline:

```text
Load Dataset
      ↓
Build Vector Store
      ↓
Read User Query
      ↓
Retrieve Relevant Chunks
      ↓
Generate Answer
```

---

## Models

### Embedding Model

```text
hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:Q4_K_M
```

### Language Model

```text
hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
```

Both models run locally through Ollama.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NaoufalBgit/rag-from-scratch.git
cd rag-from-scratch
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama models

Pull the embedding model:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:Q4_K_M
```

Pull the language model:

```bash
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
```

Check installed models:

```bash
ollama list
```

---

## Usage

Run:

```bash
python main.py
```

The dataset is first loaded and converted into embeddings.

Example:

```text
Loaded 150 entries
Added chunk 1/150 to the database
Added chunk 2/150 to the database
...
```

Then ask a question:

```text
Ask me a question: How fast can a cat run?
```

The retriever returns the most semantically relevant chunks:

```text
Retrieved knowledge:

- (0.87) Cats can run at speeds of up to approximately 30 mph.
- (0.72) Domestic cats are highly agile animals.
- (0.61) Cats use powerful hind legs when running.
```

The retrieved context is then provided to the LLM:

```text
Chatbot response:

Cats can run at speeds of approximately 30 mph.
```

---

## How It Works

### 1. Dataset Loading

The current dataset contains cat facts.

Each line is treated as an individual chunk:

```python
with open("cat-facts.txt", "r") as file:
    dataset = [
        line.strip()
        for line in file
        if line.strip()
    ]
```

### 2. Embedding Generation

Each chunk is converted into a numerical vector representation.

```text
"Most cats sleep between 12 and 16 hours per day."

                    ↓

             Embedding Model

                    ↓

[0.14, -0.31, 0.72, 0.08, ...]
```

These vectors represent the semantic meaning of the text.

### 3. Vector Storage

Each chunk is stored alongside its embedding:

```python
(chunk, embedding)
```

The current implementation keeps the vector database in memory.

### 4. Query Embedding

The user query is converted into a vector using the same embedding model.

```text
"How long do cats sleep?"

          ↓

   Embedding Model

          ↓

    Query Vector
```

### 5. Cosine Similarity

The query vector is compared against every stored embedding.

```python
similarity = cosine_similarity(
    query_embedding,
    chunk_embedding
)
```

A higher cosine similarity score means the chunk is semantically closer to the query.

### 6. Top-K Retrieval

The chunks are sorted by similarity score:

```python
similarities.sort(
    key=lambda x: x[1],
    reverse=True
)
```

Only the most relevant chunks are returned.

```python
return similarities[:top_n]
```

### 7. Prompt Augmentation

The retrieved knowledge is inserted into the LLM context.

```text
You are a helpful chatbot.

Use only the following context to answer the question.

If the answer cannot be found in the context,
say that you don't know.

Context:
- chunk 1
- chunk 2
- chunk 3
```

### 8. Generation

The user question and retrieved context are sent to the local language model.

```text
Retrieved Context
       +
User Question
       ↓
      LLM
       ↓
Grounded Answer
```

---

## Why Build RAG From Scratch?

Libraries such as LangChain and LlamaIndex can hide much of the internal RAG logic.

This project intentionally implements the fundamental mechanisms manually to better understand:

* embeddings
* vector representations
* semantic search
* similarity metrics
* retrieval
* Top-K ranking
* prompt augmentation
* context grounding
* local LLM inference

---

## Current Limitations

This implementation is intentionally simple.

Current limitations include:

* vector store exists only in memory
* embeddings are recreated at every startup
* no PDF support
* no automatic chunking
* no chunk overlap
* no metadata filtering
* no reranking
* no hybrid search
* no source citations
* no RAG evaluation pipeline

---

## Roadmap

### Phase 1 — RAG Fundamentals

* [x] Load a text dataset
* [x] Generate embeddings
* [x] Build an in-memory vector store
* [x] Implement cosine similarity
* [x] Implement Top-K retrieval
* [x] Generate answers using retrieved context

### Phase 2 — Document RAG

* [ ] TXT ingestion
* [ ] Markdown ingestion
* [ ] PDF ingestion
* [ ] Automatic chunking
* [ ] Chunk overlap
* [ ] Metadata management
* [ ] Source tracking

### Phase 3 — Vector Database

* [ ] Replace the in-memory vector store
* [ ] Integrate Qdrant or pgvector
* [ ] Persist embeddings
* [ ] Metadata filtering
* [ ] Efficient vector search

### Phase 4 — Advanced Retrieval

* [ ] Similarity threshold
* [ ] Reranking
* [ ] Query rewriting
* [ ] Multi-query retrieval
* [ ] Hybrid search

### Phase 5 — Evaluation

* [ ] Retrieval evaluation dataset
* [ ] Precision@K
* [ ] Recall@K
* [ ] Faithfulness evaluation
* [ ] Hallucination detection
* [ ] LLM-as-a-Judge evaluation

### Phase 6 — Application Layer

* [ ] FastAPI backend
* [ ] Web interface
* [ ] Document upload
* [ ] Conversation history
* [ ] Docker support

---

## Target Architecture

```text
                  Documents
            PDF / Markdown / TXT
                      ↓
               Document Parser
                      ↓
                   Chunker
                      ↓
             Embedding Model
                      ↓
              Vector Database
            Qdrant / pgvector
                      ↓
                    Query
                      ↓
             Query Embedding
                      ↓
               Vector Search
                      ↓
                  Reranker
                      ↓
             Relevant Chunks
                      ↓
             Prompt Augmentation
                      ↓
                     LLM
                      ↓
             Answer + Sources
```

---

## Technologies

| Category          | Technology                     |
| ----------------- | ------------------------------ |
| Language          | Python                         |
| LLM Runtime       | Ollama                         |
| LLM               | Llama 3.2                      |
| Embedding Model   | BGE                            |
| Retrieval         | Semantic Vector Search         |
| Similarity Metric | Cosine Similarity              |
| Architecture      | Retrieval-Augmented Generation |

Planned:

| Category         | Technology                     |
| ---------------- | ------------------------------ |
| Vector Database  | Qdrant / pgvector              |
| Backend          | FastAPI                        |
| Containerization | Docker                         |
| Evaluation       | Custom RAG evaluation pipeline |
| Frontend         | TBD                            |

---

## Topics

Recommended GitHub topics:

```text
rag
retrieval-augmented-generation
llm
ollama
python
embeddings
semantic-search
vector-search
cosine-similarity
llama
local-llm
generative-ai
```

---

## Learning Objectives

The main objective of this project is to understand how RAG systems work internally.

Concepts explored include:

* Retrieval-Augmented Generation
* Large Language Models
* text embeddings
* vector search
* semantic similarity
* information retrieval
* prompt engineering
* grounded generation
* hallucination reduction
* RAG evaluation

---

## Future Goal

The long-term objective is to evolve the project into a complete document-question answering system:

```text
Upload Documents
       ↓
Automatic Parsing
       ↓
Chunking
       ↓
Embedding & Indexing
       ↓
Ask Questions
       ↓
Semantic Retrieval
       ↓
Reranking
       ↓
Grounded Answer
       ↓
Sources + Evaluation
```

---

## Acknowledgements

The initial version of this project was inspired by the article **“Building Your Own RAG System from Scratch: A Step-by-Step Guide”** by Anish Chitturu.

The project is progressively extended to explore more advanced RAG concepts such as persistent vector databases, document ingestion, reranking, evaluation, and source-grounded generation.

---

## License

This project is intended for learning and experimentation.

An MIT license can be added if the project is published for public reuse.
