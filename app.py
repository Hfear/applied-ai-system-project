"""
ResearchRefs — Streamlit web app entry point.

Run with:
    streamlit run app.py

Requires Streamlit >= 1.32 for st.dialog support.
"""

from __future__ import annotations

import re
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
    ("selected_files", {}),
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
# CSS — file pill styling
#
# Scoped via :has(#file-card-row) so only the pill row is targeted.
# Equal height enforced on both states so all pills sit at 32 px
# regardless of filename length (white-space: nowrap prevents wrapping).
# ------------------------------------------------------------------

st.markdown("""
<style>
/* ── File pills — vertical stack ───────────────────────────────────
   Scoped via the general sibling combinator (~) off the marker div
   so only buttons that follow #file-card-row inside pills_col are
   affected. Buttons in main_col are in a different column container
   and are never siblings of #file-card-row.
   ─────────────────────────────────────────────────────────────── */

/* Selected pill — blue outline, full column width */
div.element-container:has(#file-card-row)
    ~ div.element-container
    button[kind="primary"] {
    background-color: transparent !important;
    border: 1.5px solid #4a9eff !important;
    color: #4a9eff !important;
    border-radius: 20px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 14px !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

/* Deselected pill — grey outline, full column width */
div.element-container:has(#file-card-row)
    ~ div.element-container
    button[kind="secondary"] {
    background-color: transparent !important;
    border: 1.5px solid #6b7280 !important;
    color: #6b7280 !important;
    border-radius: 20px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 14px !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    opacity: 1 !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Manage Articles dialog
# ------------------------------------------------------------------

@st.dialog("Manage Articles", width="large")
def manage_articles_dialog():
    """Modal for adding and removing uploaded articles."""

    st.markdown("#### Add articles")
    uploaded = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        key="dialog_uploader",
    )
    if uploaded:
        existing = {name for name, _ in st.session_state.documents}
        added = []
        for uf in uploaded:
            if uf.name not in existing:
                raw = uf.read()
                if uf.name.endswith(".pdf"):
                    text = extract_text_from_pdf(raw)
                    if not text:
                        st.warning(f"Could not extract text from {uf.name}.")
                        continue
                else:
                    text = raw.decode("utf-8", errors="replace")
                st.session_state.documents.append((uf.name, text))
                st.session_state.selected_files[uf.name] = True
                st.session_state.bot = None
                existing.add(uf.name)
                added.append(uf.name)
        if added:
            st.success(f"Added: {', '.join(added)}")

    if st.session_state.documents:
        st.divider()
        st.markdown("#### Remove articles")
        for name, _ in list(st.session_state.documents):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(name)
            with c2:
                if st.button("Remove", key=f"del_{name}"):
                    st.session_state.documents = [
                        (n, t) for n, t in st.session_state.documents if n != name
                    ]
                    st.session_state.selected_files.pop(name, None)
                    st.session_state.bot = None
                    st.rerun()

# ------------------------------------------------------------------
# Sidebar — upload button + clear
# ------------------------------------------------------------------

with st.sidebar:
    st.title("ResearchRefs")
    st.divider()

    if st.button("Upload & Manage files", use_container_width=True, type="primary"):
        manage_articles_dialog()

    if st.session_state.documents:
        st.divider()
        if st.button("Clear all articles", use_container_width=True):
            st.session_state.documents = []
            st.session_state.selected_files = {}
            st.session_state.bot = None
            st.session_state.bot_doc_key = frozenset()
            st.session_state.last_summary = None
            st.session_state.suggested_claims = []
            st.session_state.custom_claim = ""
            st.rerun()

# ------------------------------------------------------------------
# Active documents — filtered by selection
# ------------------------------------------------------------------

active_documents = [
    (name, text)
    for name, text in st.session_state.documents
    if st.session_state.selected_files.get(name, True)
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_bot(llm_client, documents) -> DocuBot:
    """Return a cached DocuBot, rebuilding only when the active set changes."""
    current_key = frozenset(name for name, _ in documents)
    if st.session_state.bot is None or st.session_state.bot_doc_key != current_key:
        with st.spinner("Building semantic index..."):
            st.session_state.bot = DocuBot(
                documents=documents,
                llm_client=llm_client,
                embedding_model=_embedding_model,
            )
        st.session_state.bot_doc_key = current_key
    else:
        st.session_state.bot.llm_client = llm_client
    return st.session_state.bot


def _render_answer(answer: str) -> None:
    """Render summary in blue; Key Quotes with orange quote text."""
    if "Key Quotes:" not in answer:
        st.markdown(
            f'<div style="color:#4a9eff; line-height:1.7;">{answer}</div>',
            unsafe_allow_html=True,
        )
        return

    summary_part, quotes_part = answer.split("Key Quotes:", 1)
    summary_text = summary_part.replace("Summary:", "").strip()

    st.markdown(
        f'<div style="color:#4a9eff; line-height:1.7;">{summary_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Key Quotes:**")
    for line in quotes_part.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        colored = re.sub(
            r'"([^"]*)"',
            r'<span style="color:#ff8c42">"\1"</span>',
            stripped,
        )
        st.markdown(colored, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page header (full width)
# ------------------------------------------------------------------

st.title("ResearchRefs")
st.caption("Upload research articles and ask questions grounded in your sources.")
st.divider()

# ------------------------------------------------------------------
# Two-column layout: main content (left) | file pills (right)
#
# pills_col is populated first in Python so that st.stop() calls
# inside main_col do not prevent the pills from rendering.
# ------------------------------------------------------------------

main_col, pills_col = st.columns([3, 1])

# ------------------------------------------------------------------
# Right column — file pills
# ------------------------------------------------------------------

with pills_col:
    if st.session_state.documents:
        selected_count = sum(
            1 for v in st.session_state.selected_files.values() if v
        )
        total_count = len(st.session_state.documents)
        st.markdown(f"**Sources** ({selected_count}/{total_count} active)")

        # Marker div — CSS ~ sibling selector scopes pill styles to
        # all buttons that follow this point in pills_col only.
        st.markdown('<div id="file-card-row"></div>', unsafe_allow_html=True)

        for name, _ in st.session_state.documents:
            selected = st.session_state.selected_files.get(name, True)
            label = f"✓ {name}" if selected else f"○ {name}"
            if st.button(
                label,
                key=f"card_{name}",
                type="primary" if selected else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_files[name] = not selected
                st.rerun()
    else:
        st.caption("No articles loaded.\nUse **Upload articles** in the sidebar.")

# ------------------------------------------------------------------
# Left column — two tabs
#
# tab_args is defined first in Python so that st.stop() calls inside
# tab_research do not prevent the Argument Analysis tab from rendering.
# ------------------------------------------------------------------

with main_col:
    # Deferred custom_claim reset: the research pipeline sets this flag instead
    # of writing directly to the key, because tab_args (which owns the widget)
    # runs first and instantiates the text_input before the pipeline executes.
    # Applying the reset here — before any widgets are created — is safe.
    if st.session_state.pop("_clear_custom_claim", False):
        st.session_state.custom_claim = ""

    tab_research, tab_args = st.tabs(["Research Question", "Argument Analysis"])

    # ── Tab 2: Argument Analysis ──────────────────────────────────
    # Defined first so it is always queued before tab_research runs.
    with tab_args:
        if not st.session_state.last_summary:
            st.info("Run a research query first to enable argument analysis.")
        else:
            if st.button("Suggest claims from AI summary"):
                llm_client = None
                try:
                    llm_client = GeminiClient()
                except RuntimeError:
                    pass
                bot = _get_bot(llm_client, active_documents)
                with st.spinner("Identifying key claims..."):
                    st.session_state.suggested_claims = bot.suggest_claims(
                        st.session_state.last_summary
                    )

            if st.session_state.suggested_claims:
                st.markdown("**Suggested arguments — click to select:**")
                for i, claim in enumerate(st.session_state.suggested_claims):
                    if st.button(claim, key=f"claim_btn_{i}"):
                        st.session_state.custom_claim = claim
                        st.rerun()

            st.text_input(
                "Or enter your own position...",
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

                if llm_client is not None:
                    bot = _get_bot(llm_client, active_documents)

                    with st.spinner("Analysing arguments..."):
                        result = bot.analyse_arguments(claim_text, top_k=6)

                    st.markdown(f"**Claim:** _{claim_text}_")

                    if "FOR:" in result and "AGAINST:" in result:
                        parts = result.split("AGAINST:", 1)
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
                        st.info(result)

    # ── Tab 1: Research Question ──────────────────────────────────
    with tab_research:
        query = st.text_input(
            "Ask a research question...",
            placeholder="e.g. What are the main arguments about climate policy?",
        )
        ask_clicked = st.button("Analyse", type="primary", disabled=not query)

        if ask_clicked:
            if not st.session_state.documents:
                st.warning("Please upload at least one article to get started.")
                st.stop()

            if not active_documents:
                st.warning("Select at least one article to search.")
                st.stop()

            llm_client = None
            try:
                llm_client = GeminiClient()
            except RuntimeError as exc:
                st.warning(f"LLM unavailable ({exc}). Falling back to retrieval-only mode.")

            bot = _get_bot(llm_client, active_documents)

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
                    f'<span style="background:#1a3a5c; color:#4a9eff; border:1px solid #4a9eff; '
                    f'border-radius:12px; padding:4px 12px; margin:3px; font-size:0.85rem; '
                    f'display:inline-block">{t}</span>'
                    for t in themes
                )
                st.markdown(f"**Themes:** {badge_html}", unsafe_allow_html=True)

            # Step 4 — Synthesis
            with st.spinner("Generating summary..."):
                answer = bot.synthesise(query, snippets)

            raw_summary = answer
            if "Key Quotes:" in answer:
                raw_summary = answer.split("Key Quotes:")[0].replace("Summary:", "").strip()
            st.session_state.last_summary = raw_summary
            st.session_state.suggested_claims = []
            st.session_state["_clear_custom_claim"] = True

            st.divider()
            st.subheader("Answer")
            _render_answer(answer)
