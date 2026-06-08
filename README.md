# 🎬 ReelMind

An intelligent movie search engine powered by Retrieval-Augmented Generation (RAG).

🔗 **Live Demo:** https://reelmind.streamlit.app

## What it does

Ask any movie-related question in natural language and get AI-powered answers based on IMDb's Top 1000 movies.

**Example queries:**
- "What is the best Christopher Nolan movie?"
- "Suggest a movie like The Godfather"
- "Best thriller movies with high ratings"
- "Which movie has the highest IMDb rating?"

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embedding Model | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | Pinecone |
| LLM | Llama 3.3 70B via Groq API |
| Frontend | Streamlit |
| Dataset | IMDb Top 1000 Movies |

## Architecture

User Query → Sentence Transformer → Pinecone Vector Search → Top 10 Movies → Llama 3.3 → Answer

## How it works

1. **Ingestion** — 1000 IMDb movies are embedded using Sentence Transformers and stored in Pinecone
2. **Retrieval** — user query is embedded and semantically matched against the vector database
3. **Generation** — top 10 retrieved movies are passed as context to Llama 3.3 via Groq API
4. **Response** — the LLM reasons over the retrieved movies and returns a natural language answer

## Run locally

git clone https://github.com/TirthPinkalSheth/reelmind.git
cd reelmind
pip install -r requirements.txt

Add a .env file with GROQ_API_KEY and PINECONE_API_KEY, then:

python ingest.py
streamlit run app.py

## Author

Tirth Pinkal Sheth — https://github.com/TirthPinkalSheth
