import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_NAME = "reelmind"
TOP_K = 10

_embed_model = None
_index = None
_client = None

def _init():
    global _embed_model, _index, _client
    if _client is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(INDEX_NAME)
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(question):
    _init()
    embedding = _embed_model.encode(question).tolist()
    results = _index.query(vector=embedding, top_k=TOP_K, include_metadata=True)
    chunks = []
    for match in results["matches"]:
        meta = match["metadata"]
        chunks.append(f"{meta['title']} ({meta['year']}) - Director: {meta.get('director','N/A')} - Rating: {meta['rating']}\nGenre: {meta['genre']}\n{meta['overview']}")
    return chunks

def generate(question, chunks):
    _init()
    context = "\n\n".join(chunks)
    prompt = f"""You are a movie expert. Use the following movies to answer the question.

Movies:
{context}

Question: {question}

Give a helpful, specific answer based on the movies above."""

    response = _client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

def answer(question):
    chunks = retrieve(question)
    return generate(question, chunks)
