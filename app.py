"""
ResearchRefs — Streamlit web app entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from docubot import DocuBot
from llm_client import GeminiClient
from pdf_parser import extract_text_from_pdf

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(page_title="ResearchRefs", page_icon="📚", layout="wide")

# ------------------------------------------------------------------
# Cached embedding model (downloads ~90 MB on first run)
# ------------------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# Load model once — spinner only visible on the actual first load
if "model_loaded" not in st.session_state:
    with st.spinner("Loading semantic search model (first run only)..."):
        _embedding_model = load_embedding_model()
    st.session_state.model_loaded = True
else:
    _embedding_model = load_embedding_model()

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

for _key, _default in [
    ("documents", []),
    ("bot", None),
    ("bot_doc_key", frozenset()),
    ("last_summary", None),
    ("suggested_claims", []),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

if "custom_claim" not in st.session_state:
    st.session_state.custom_claim = ""

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("ResearchRefs")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload articles",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Upload .txt, .md, or .pdf files to analyse.",
    )

    if uploaded_files:
        existing_names = {name for name, _ in st.session_state.documents}
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
                st.session_state.documents.append((uf.name, text))
                existing_names.add(uf.name)
                # Invalidate cached bot when documents change
                st.session_state.bot = None

    st.divider()

    if st.session_state.documents:
        st.subheader("Loaded Articles")
        for name, _ in st.session_state.documents:
            st.markdown(f"- {name}")
    else:
        st.info("No articles loaded yet.")

    if st.session_state.documents:
        if st.button("Clear all articles", use_container_width=True):
            st.session_state.documents = []
            st.session_state.bot = None
            st.session_state.bot_doc_key = frozenset()
            st.session_state.last_summary = None
            st.session_state.suggested_claims = []
            st.session_state.custom_claim = ""
            st.rerun()

# ------------------------------------------------------------------
# Helper — build/reuse bot
# ------------------------------------------------------------------

def _get_bot(llm_client) -> DocuBot:
    """Return a cached DocuBot, rebuilding only when documents have changed."""
    current_key = frozenset(name for name, _ in st.session_state.documents)
    if st.session_state.bot is None or st.session_state.bot_doc_key != current_key:
        with st.spinner("Building semantic index..."):
            st.session_state.bot = DocuBot(
                documents=st.session_state.documents,
                llm_client=llm_client,
                embedding_model=_embedding_model,
            )
        st.session_state.bot_doc_key = current_key
    else:
        # Attach fresh llm_client in case it changed
        st.session_state.bot.llm_client = llm_client
    return st.session_state.bot


# ------------------------------------------------------------------
# Main area — header
# ------------------------------------------------------------------

st.title("ResearchRefs")
st.caption("Upload research articles and ask questions grounded in your sources.")
st.divider()

# ------------------------------------------------------------------
# Query input
# ------------------------------------------------------------------

query = st.text_input(
    "Ask a research question...",
    placeholder="e.g. What are the main arguments about climate policy?",
)
ask_clicked = st.button("Analyse", type="primary", disabled=not query)

# ------------------------------------------------------------------
# Analysis pipeline
# ------------------------------------------------------------------

if ask_clicked:
    if not st.session_state.documents:
        st.warning("Please upload at least one article to get started.")
        st.stop()

    llm_client = None
    try:
        llm_client = GeminiClient()
    except RuntimeError as exc:
        st.warning(f"LLM unavailable ({exc}). Falling back to retrieval-only mode.")

    bot = _get_bot(llm_client)

    # Step 1 — Query expansion
    with st.spinner("Expanding query..."):
        expanded_query, expanded_terms = bot.expand_query(query)

    if expanded_terms:
        st.info(f"**Also searching for:** {', '.join(expanded_terms)}")
    else:
        st.info("Searching with original query.")

    # Step 2 — Retrieval
    with st.spinner("Searching articles..."):
        snippets = bot.retrieve(expanded_query, top_k=5)

    sources_searched = list(dict.fromkeys(f for f, _, _ in snippets))

    if snippets:
        st.success(
            f"Found **{len(snippets)}** passage(s) from: {', '.join(sources_searched)}"
        )
    else:
        st.error(
            "No relevant content found in your articles for this question. "
            "Try rephrasing or uploading more sources."
        )
        st.stop()

    # Step 3 — Theme extraction
    with st.spinner("Extracting themes..."):
        themes = bot.extract_themes(snippets)

    if themes:
        badge_html = " ".join(
            f'<span style="background:#e8e8e8; border-radius:12px; '
            f'padding:3px 12px; margin:3px; font-size:0.85em; '
            f'display:inline-block">{t}</span>'
            for t in themes
        )
        st.markdown(f"**Themes:** {badge_html}", unsafe_allow_html=True)

    # Step 4 — Synthesis
    with st.spinner("Generating summary..."):
        answer = bot.synthesise(query, snippets)

    # Extract raw summary for claim suggestions
    raw_summary = answer
    if "Key Quotes:" in answer:
        raw_summary = answer.split("Key Quotes:")[0].replace("Summary:", "").strip()
    st.session_state.last_summary = raw_summary
    st.session_state.suggested_claims = []  # reset on new query
    st.session_state.custom_claim = ""

    st.divider()
    st.subheader("Answer")
    st.markdown(answer)

# ------------------------------------------------------------------
# Argument Analysis section (shown after any successful query)
# ------------------------------------------------------------------

if st.session_state.last_summary:
    st.divider()
    st.subheader("Analyse Arguments")

    # Suggest claims button
    if st.button("Suggest claims from this summary"):
        llm_client = None
        try:
            llm_client = GeminiClient()
        except RuntimeError:
            pass

        bot = _get_bot(llm_client)
        with st.spinner("Identifying key claims..."):
            st.session_state.suggested_claims = bot.suggest_claims(
                st.session_state.last_summary
            )

    # Display suggested claims as clickable buttons
    if st.session_state.suggested_claims:
        st.markdown("**Suggested claims — click to select:**")
        for i, claim in enumerate(st.session_state.suggested_claims):
            if st.button(claim, key=f"claim_btn_{i}"):
                st.session_state.custom_claim = claim
                st.rerun()

    # Custom claim text input (key bound to session state)
    st.text_input(
        "Or enter your own claim or position...",
        key="custom_claim",
    )

    analyse_args_clicked = st.button(
        "Analyse Arguments",
        type="secondary",
        disabled=not st.session_state.custom_claim,
    )

    if analyse_args_clicked:
        claim_text = st.session_state.custom_claim

        llm_client = None
        try:
            llm_client = GeminiClient()
        except RuntimeError as exc:
            st.warning(f"LLM unavailable: {exc}")
            st.stop()

        bot = _get_bot(llm_client)

        with st.spinner("Analysing arguments..."):
            result = bot.analyse_arguments(claim_text, top_k=6)

        st.markdown(f"**Claim:** _{claim_text}_")

        # Parse FOR / AGAINST sections into two columns
        if "FOR:" in result and "AGAINST:" in result:
            parts = result.split("AGAINST:")
            for_text = parts[0].replace("FOR:", "").strip()
            against_text = parts[1].strip()

            col_for, col_against = st.columns(2)
            with col_for:
                st.markdown("#### For")
                st.markdown(for_text)
            with col_against:
                st.markdown("#### Against")
                st.markdown(against_text)
        else:
            # No clear structure — show as plain message
            st.info(result)
