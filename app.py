import os
import streamlit as st
from ingest import load_and_clean, embed_and_store
from query import answer

st.set_page_config(
    page_title="ReelMind",
    page_icon="🎬",
    layout="centered"
)

# Auto-build ChromaDB if it doesn't exist (runs on Streamlit Cloud first load)
if not os.path.exists("./chroma_db"):
    with st.spinner("Building movie database for first time... please wait ~1 minute."):
        df = load_and_clean("data/movies.csv")
        embed_and_store(df)
    st.rerun()

st.title("🎬 ReelMind")
st.caption("Intelligent movie search — powered by RAG + Llama AI")
st.divider()

st.markdown("**💡 Try asking:**")

examples = [
    "What is the best Christopher Nolan movie?",
    "Suggest a movie like The Godfather",
    "Best thriller movies in the dataset?",
    "Which movie has the highest IMDb rating?",
]

# Session state to persist selected example across rerenders
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

col1, col2 = st.columns(2)
for i, ex in enumerate(examples):
    col = col1 if i % 2 == 0 else col2
    if col.button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state.question_input = ex

st.divider()

question = st.text_input(
    label="Your question",
    value=st.session_state.question_input,
    placeholder="e.g. What are some must-watch crime dramas?",
    label_visibility="collapsed"
)

ask = st.button("Search 🎬", type="primary", use_container_width=True)

if ask and question.strip():
    with st.spinner("Searching through 1000 movies ..."):
        try:
            result = answer(question.strip())
            st.markdown("### 🎥 Answer")
            st.markdown(result)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
elif ask and not question.strip():
    st.warning("Please type a question first!")