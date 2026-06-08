import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_NAME = "reelmind"

def load_and_clean(path="data/movies.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Series_Title", "Overview"])
    df = df.reset_index(drop=True)
    return df

def embed_and_store(df):
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding movies...")
    texts = (
        df["Series_Title"] + ". " +
        "Director: " + df.get("Director", pd.Series([""] * len(df))).fillna("") + ". " +
        "Genre: " + df.get("Genre", pd.Series([""] * len(df))).fillna("") + ". " +
        df["Overview"]
    ).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(INDEX_NAME)

    print("Uploading vectors...")
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        vectors = [
            (
                str(i + j),
                batch_embeddings[j].tolist(),
                {
                    "title": str(batch.iloc[j]["Series_Title"]),
                    "overview": str(batch.iloc[j]["Overview"]),
                    "year": str(batch.iloc[j].get("Released_Year", "")),
                    "rating": str(batch.iloc[j].get("IMDB_Rating", "")),
                    "genre": str(batch.iloc[j].get("Genre", "")),
                    "director": str(batch.iloc[j].get("Director", "")),
                }
            )
            for j in range(len(batch))
        ]
        index.upsert(vectors=vectors)
        print(f"Uploaded {min(i+batch_size, len(df))}/{len(df)}")

    print("Done!")

if __name__ == "__main__":
    df = load_and_clean("data/movies.csv")
    embed_and_store(df)
