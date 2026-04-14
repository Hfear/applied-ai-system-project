"""
DocuBot Article Analyser — core engine.

Responsibilities:
- Accept documents directly (from Streamlit uploads) or load from articles_folder
- Build a word-level retrieval index
- Score and retrieve relevant paragraphs
- Expand queries via Gemini (query expansion step)
- Synthesise cited answers via Gemini (synthesis step)
- Log all activity via logger.py
"""

from __future__ import annotations

import os
import glob
import logger as _logger_module


STOP_WORDS = {
    "is", "there", "any", "of", "the", "a", "an", "in", "to",
    "and", "or", "how", "do", "i", "where", "what", "which",
    "when", "was", "are", "it", "this", "that", "for", "with",
    "my", "we", "be", "at", "by", "from", "on", "not", "if",
    "are", "were", "been", "being", "have", "has", "had",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall", "its", "these", "those", "you",
    "he", "she", "they", "your", "his", "her", "our", "their",
    "so", "than", "as", "into", "through", "during", "before",
    "after", "no", "but",
}

SYNTHESIS_PROMPT_TEMPLATE = """\
You are a research assistant that answers questions using only the article excerpts provided.

Your job:
1. Write a short cited summary (3-5 sentences) answering the question.
   After each claim, cite the source in brackets e.g. [Source 1].
2. End with a "Key Quotes" section with 2-3 direct quotes, each attributed to its source.

Rules:
- Only use information from the excerpts. Do not add outside knowledge.
- If excerpts lack enough information, say exactly:
  "The available articles do not contain enough information to answer this question."

Article excerpts:
{context}

Question: {query}

Format:
Summary:
[cited summary]

Key Quotes:
- "[quote]" — [Source N: Title]
- "[quote]" — [Source N: Title]
"""


class DocuBot:
    """
    Article Analyser engine.

    Parameters
    ----------
    documents:
        Optional list of (filename, text) tuples provided directly
        (e.g. from Streamlit file uploads).  When supplied, articles_folder
        is ignored.
    articles_folder:
        Directory to load .txt/.md files from when documents is not provided.
    llm_client:
        An instance of GeminiClient (or compatible).  Required for query
        expansion and synthesis; if None the system falls back to
        retrieval-only mode.
    """

    def __init__(
        self,
        documents: list[tuple[str, str]] | None = None,
        articles_folder: str = "articles",
        llm_client=None,
    ):
        self.articles_folder = articles_folder
        self.llm_client = llm_client

        if documents:
            self.documents = documents
        else:
            self.documents = self._load_articles()

        if not self.documents:
            import warnings
            warnings.warn(
                "DocuBot initialised with no documents. "
                "Upload files or populate the articles/ folder.",
                stacklevel=2,
            )

        self.index = self._build_index(self.documents)

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _load_articles(self) -> list[tuple[str, str]]:
        """Load .txt and .md files from articles_folder."""
        docs: list[tuple[str, str]] = []
        pattern = os.path.join(self.articles_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    docs.append((os.path.basename(path), text))
                except OSError as exc:
                    _logger_module.log_error("_load_articles", exc)
        return docs

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, documents: list[tuple[str, str]]) -> dict:
        index: dict[str, list[str]] = {}
        for filename, text in documents:
            for word in text.split():
                word = word.lower().strip('.,!?:;()\'"[]{}``')
                if word not in index:
                    index[word] = []
                if filename not in index[word]:
                    index[word].append(filename)
        return index

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_paragraph(self, query: str, text: str) -> int:
        """
        Prefix-match scoring.  Each unique query term (excluding stop words)
        that matches any word in the text via startswith() adds 1 to the score.
        """
        query_words = {
            w.lower().strip('.,!?:;()\'"[]{}``')
            for w in query.split()
            if w.lower() not in STOP_WORDS
        }
        text_words = {
            w.lower().strip('.,!?:;()\'"[]{}``')
            for w in text.split()
        }
        score = 0
        for qw in query_words:
            for tw in text_words:
                if tw.startswith(qw):
                    score += 1
                    break
        return score

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, str, int]]:
        """
        Split every document into paragraphs, score each one, return the
        top_k with score >= 2 as (filename, paragraph, score) tuples.
        """
        scored: list[tuple[int, str, str]] = []
        for filename, text in self.documents:
            for paragraph in text.split("\n\n"):
                if len(paragraph.strip()) < 20:
                    continue
                score = self._score_paragraph(query, paragraph)
                scored.append((score, filename, paragraph))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, filename, paragraph in scored[:top_k]:
            if score >= 2:
                results.append((filename, paragraph, score))
        return results

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> tuple[str, list[str]]:
        """
        Ask Gemini for 3-4 related search terms to broaden retrieval.

        Returns (expanded_query_string, list_of_extra_terms).
        Falls back to (original_query, []) if the LLM is unavailable.
        """
        if self.llm_client is None:
            return query, []

        prompt = (
            "Given this research question, generate 3-4 related search terms "
            "that would help find relevant passages. Return only the terms as "
            "a comma-separated list.\n"
            f"Question: {query}"
        )
        try:
            raw = self.llm_client.generate(prompt)
            extra_terms = [t.strip() for t in raw.split(",") if t.strip()]
            expanded = query + " " + " ".join(extra_terms)
            return expanded, extra_terms
        except Exception as exc:
            _logger_module.log_error("expand_query", exc)
            return query, []

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def synthesise(self, query: str, snippets: list[tuple[str, str, int]]) -> str:
        """
        Build context from retrieved snippets and call Gemini to produce a
        cited summary + Key Quotes section.

        Falls back to showing raw passages if the LLM is unavailable.
        """
        context_blocks = []
        for i, (filename, text, _) in enumerate(snippets, start=1):
            context_blocks.append(f"[Source {i}: {filename}]\n{text}")
        context = "\n\n".join(context_blocks)

        if self.llm_client is not None:
            synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                context=context, query=query
            )
            try:
                answer = self.llm_client.generate(synthesis_prompt)
            except Exception as exc:
                _logger_module.log_error("synthesise", exc)
                answer = "⚠️ Gemini unavailable — showing raw passages:\n\n" + context
        else:
            answer = "⚠️ No LLM configured — showing raw passages:\n\n" + context

        _logger_module.log_answer(answer)
        return answer

    def answer_with_citations(self, query: str) -> dict:
        """
        Full agentic pipeline (convenience wrapper used when all three steps
        should run without progressive UI display):
          1. Expand the query (Gemini call 1)
          2. Retrieve relevant passages
          3. Synthesise a cited answer (Gemini call 2)

        Returns a dict suitable for Streamlit to inspect:
        {
            "expanded_terms": [...],
            "sources_searched": [...],
            "passages_found": int,
            "answer": str,
        }
        """
        _logger_module.log_query("answer_with_citations", query)

        expanded_query, expanded_terms = self.expand_query(query)

        snippets = self.retrieve(expanded_query, top_k=5)
        _logger_module.log_snippets(snippets)

        sources_searched = list(dict.fromkeys(f for f, _, _ in snippets))

        if not snippets:
            return {
                "expanded_terms": expanded_terms,
                "sources_searched": [],
                "passages_found": 0,
                "answer": (
                    "No relevant content found in your articles for this question. "
                    "Try rephrasing or uploading more sources."
                ),
            }

        answer = self.synthesise(query, snippets)

        return {
            "expanded_terms": expanded_terms,
            "sources_searched": sources_searched,
            "passages_found": len(snippets),
            "answer": answer,
        }
