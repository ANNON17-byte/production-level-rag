"""STEP 8 - Web app. Run locally with:  streamlit run app.py"""
import streamlit as st

from generation import generate_answer
from pgvector_store import PgRetriever

st.set_page_config(page_title="My RAG App")
st.title("Ask my documents")


@st.cache_resource            # load the embedding model + DB connection only once
def get_retriever():
    return PgRetriever()


question = st.text_input("Your question")
if question:
    retriever = get_retriever()
    with st.spinner("Searching and thinking..."):
        hits = retriever.retrieve(question)
        answer = generate_answer(question, hits)
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Sources")
    for i, h in enumerate(hits, 1):
        with st.expander(f"[{i}] {h['source']} - page {h['page']} (score {h['score']:.2f})"):
            st.write(h["text"])
