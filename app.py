"""
Article Analyser — Streamlit web app entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from docubot import DocuBot
from llm_client import GeminiClient
from pdf_parser import extract_text_from_pdf

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Article Analyser",
    page_icon="📄",
    layout="wide",
)

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

if "articles" not in st.session_state:
    # List of (filename, text) tuples — persists across queries
    st.session_state.articles = []


# ------------------------------------------------------------------
# Sidebar — file uploader
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Upload Articles")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Upload .txt, .md, or .pdf files to analyse.",
    )

    if uploaded_files:
        existing_names = {name for name, _ in st.session_state.articles}
        for uf in uploaded_files:
            if uf.name not in existing_names:
                raw = uf.read()
                if uf.name.endswith(".pdf"):
                    text = extract_text_from_pdf(raw)
                    if not text:
                        st.warning(f"Could not extract text from {uf.name}.")
                        continue
                else:
                    text = raw.decode("utf-8", errors="replace")
                st.session_state.articles.append((uf.name, text))
                existing_names.add(uf.name)

    st.divider()

    if st.session_state.articles:
        st.subheader("Loaded Articles")
        for name, _ in st.session_state.articles:
            st.markdown(f"- {name}")
    else:
        st.info("No articles loaded yet.")

    if st.session_state.articles:
        if st.button("Clear all articles", use_container_width=True):
            st.session_state.articles = []
            st.rerun()

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------

st.title("Article Analyser")
st.caption("Upload articles, ask a question, and get a cited summary grounded in your sources.")

st.divider()

query = st.text_input(
    "Ask a question about your articles",
    placeholder="e.g. What are the main arguments about climate policy?",
)

ask_clicked = st.button("Analyse", type="primary", disabled=not query)

if ask_clicked:
    if not st.session_state.articles:
        st.warning("Please upload at least one article to get started.")
        st.stop()

    # Build LLM client (graceful degradation if key missing)
    llm_client = None
    try:
        llm_client = GeminiClient()
    except RuntimeError as exc:
        st.warning(f"LLM unavailable ({exc}). Falling back to retrieval-only mode.")

    bot = DocuBot(
        documents=st.session_state.articles,
        llm_client=llm_client,
    )

    # ------------------------------------------------------------------
    # Step 1 — Query expansion
    # ------------------------------------------------------------------
    with st.spinner("Expanding query..."):
        expanded_query, expanded_terms = bot.expand_query(query)

    if expanded_terms:
        st.info(f"**Also searching for:** {', '.join(expanded_terms)}")
    else:
        st.info("Searching with original query (no expansion available).")

    # ------------------------------------------------------------------
    # Step 2 — Retrieval
    # ------------------------------------------------------------------
    with st.spinner("Searching articles..."):
        snippets = bot.retrieve(expanded_query, top_k=5)

    sources_searched = list(dict.fromkeys(f for f, _, _ in snippets))
    passages_found = len(snippets)

    if snippets:
        source_list = ", ".join(sources_searched)
        st.success(f"Found **{passages_found}** passage(s) from: {source_list}")
    else:
        st.error(
            "No relevant content found in your articles for this question. "
            "Try rephrasing or uploading more sources."
        )
        st.stop()

    # ------------------------------------------------------------------
    # Step 3 — Synthesis
    # ------------------------------------------------------------------
    with st.spinner("Generating summary..."):
        answer = bot.synthesise(query, snippets)

    st.divider()
    st.subheader("Answer")
    st.markdown(answer)
