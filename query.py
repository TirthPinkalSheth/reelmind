import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
#config variables
load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHROMA_PATH = "./chroma_db"
COLLECTION  = "reelmind"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 50
_embed_model = SentenceTransformer(EMBED_MODEL)
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection    = _chroma_client.get_collection(COLLECTION)
def retrieve(question: str) -> list:
    #This function takes the user's question, converts it to a vector, and finds the 5 most similar movies from ChromaDB
    query_vector = _embed_model.encode([question]).tolist()
    results = _collection.query(
        query_embeddings=query_vector,
        n_results=TOP_K
    )
    return results["documents"][0]
def generate(question: str, chunks: list) -> str:
    #This takes the question + retrieved chunks and calls Gemini to generate the answer
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are ReelMind, an intelligent movie assistant.
Answer the user's question using ONLY the movie data provided below.
Be conversational, specific, and helpful.
If the answer is not in the context say: "I don't have enough data on that — try rephrasing!"
Do NOT invent movie details, cast members, or ratings.
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
    result= generate(question, chunks)
    return result