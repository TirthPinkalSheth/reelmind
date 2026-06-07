import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
CSV_PATH    = "data/movies.csv"
CHROMA_PATH = "./chroma_db"
COLLECTION  = "reelmind"
EMBED_MODEL = "all-MiniLM-L6-v2"
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Columns:", list(df.columns))
    print("Total rows:", len(df)) 
    df['cast'] = df['Star1'] + ', ' + df['Star2'] + ', ' + df['Star3'] + ', ' + df['Star4'] # Combine star columns into one as a cast
    df.rename(columns={
        'Series_Title': 'title',
        'Released_Year': 'year',
        'Genre': 'genre',
        'IMDB_Rating': 'rating',
        'Overview': 'plot',
        'Director': 'director'
    }, inplace=True)
    df.dropna(subset=['title', 'plot'], inplace=True) #drop empty columns
    df = df.fillna({col: ("Unknown" if df[col].dtype == object or str(df[col].dtype) == 'string' else 0) for col in df.columns}) # Fill any remaining empty string field with "Unknown"
    print("Clean rows:", len(df))
    return df
def build_chunk(row: pd.Series) -> str:
    #The above fnc takes movie rows and converts into a single rich text block for embedding
    return f"""Title: {row['title']}
Year: {row['year']}
Director: {row['director']}
Cast: {row['cast']}
Genre: {row['genre']}
IMDb Rating: {row['rating']}
Plot: {str(row['plot'])[:500]}""".strip()
def embed_and_store(df: pd.DataFrame) -> None:
    #This function takes the cleaned dataframe, embeds all chunks, and saves them to ChromaDB.
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    chunks = df.apply(build_chunk, axis=1).tolist()
    ids = [f"movie_{i}" for i in range(len(chunks))]
    print(f"Embedding {len(chunks)} movies ...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
        print("Cleared existing collection.")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Done — {len(chunks)} movies stored in ChromaDB.")
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    df = load_and_clean(CSV_PATH)
    embed_and_store(df)