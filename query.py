import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION  = "reelmind"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 10

_embed_model = None
_chroma_client = None
_collection = None
_client = None

def _init():
    global _embed_model, _chroma_client, _collection, _client
    if _client is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_collection(COLLECTION)
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(question: str) -> list:
    _init()
    query_vector = _embed_model.encode([question]).tolist()
    results = _collection.query(query_embeddings=query_vector, n_results=TOP_K)
    return results["documents"][0]

def generate(question: str, chunks: list) -> str:
    _init()
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are ReelMind, an intelligent movie assistant.
Answer the user's question using ONLY the movie data provided below.
Be helpful, specific, and mention movie titles, ratings, directors, and cast where relevant.

Movie Context:
{context}

User Question: {question}

Answer:"""
    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def answer(question: str) -> str:
    chunks = retrieve(question)
    result = generate(question, chunks)
    return result