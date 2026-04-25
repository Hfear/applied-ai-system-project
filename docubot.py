"""
ResearchRefs — core engine.

Responsibilities:
- Accept documents directly (from Streamlit uploads) or load from articles_folder
- Build a semantic similarity index using sentence-transformers
- Fall back to word-overlap retrieval if sentence-transformers is unavailable
- Expand queries via Gemini (query expansion step)
- Extract recurring themes from retrieved passages
- Synthesise cited answers via Gemini (synthesis step)
- Suggest debatable claims from a summary
- Analyse arguments for/against a claim
- Log all activity via logger.py
"""

from __future__ import annotations

import os
import re
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

ARGUMENT_PROMPT_TEMPLATE = """\
You are analysing evidence for and against a specific claim.
Claim: {claim}

Using only the excerpts below, identify:
- Arguments FOR this claim (with citations)
- Arguments AGAINST this claim (with citations)

Format your response as:
FOR:
- [argument] [Source N: filename]
- [argument] [Source N: filename]

AGAINST:
- [argument] [Source N: filename]
- [argument] [Source N: filename]

If the excerpts do not contain evidence for either side, say so clearly.
Do not use outside knowledge.

Excerpts:
{context}
"""


class DocuBot:
    """
    ResearchRefs core engine.

    Parameters
    ----------
    documents:
        Optional list of (filename, text) tuples provided directly
        (e.g. from Streamlit file uploads). When supplied, articles_folder
        is ignored.
    articles_folder:
        Directory to load .txt/.md files from when documents is not provided.
    llm_client:
        An instance of GeminiClient (or compatible). Required for LLM steps;
        if None the system falls back to retrieval-only mode.
    embedding_model:
        A loaded SentenceTransformer instance. When provided, semantic
        similarity retrieval is used. When None, falls back to word-overlap.
    """

    def __init__(
        self,
        documents: list[tuple[str, str]] | None = None,
        articles_folder: str = "articles",
        llm_client=None,
        embedding_model=None,
    ):
        self.articles_folder = articles_folder
        self.llm_client = llm_client
        self._embedding_model = embedding_model

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
        self.semantic_index = self._build_semantic_index(self.documents)

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
    # Word-overlap index (fallback)
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

    def _score_paragraph(self, query: str, text: str) -> int:
        """Prefix-match scoring, skipping stop words."""
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
    # Semantic index
    # ------------------------------------------------------------------

    def _build_semantic_index(
        self, documents: list[tuple[str, str]]
    ) -> list:
        """
        Batch-encode all paragraphs into embeddings.
        Returns list of (filename, paragraph, embedding) or [] on failure.
        """
        if self._embedding_model is None:
            return []
        try:
            all_paras: list[str] = []
            all_meta: list[str] = []
            for filename, text in documents:
                for para in text.split("\n\n"):
                    if len(para.strip()) < 20:
                        continue
                    all_paras.append(para)
                    all_meta.append(filename)

            if not all_paras:
                return []

            embeddings = self._embedding_model.encode(
                all_paras, convert_to_tensor=True, show_progress_bar=False
            )
            return [
                (meta, para, emb)
                for meta, para, emb in zip(all_meta, all_paras, embeddings)
            ]
        except Exception as exc:
            _logger_module.log_error("_build_semantic_index", exc)
            return []

    def _semantic_score(
        self, query: str, top_k: int
    ) -> list[tuple[str, str, float]]:
        """Cosine similarity retrieval over pre-built semantic index."""
        try:
            from sentence_transformers import util

            query_emb = self._embedding_model.encode(
                query, convert_to_tensor=True
            )
            results = [
                (filename, para, float(util.cos_sim(query_emb, emb)))
                for filename, para, emb in self.semantic_index
            ]
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]
        except Exception as exc:
            _logger_module.log_error("_semantic_score", exc)
            return []

    # ------------------------------------------------------------------
    # Retrieval (semantic primary, word-overlap fallback)
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, str, float]]:
        """
        Return top_k relevant passages as (filename, paragraph, score).
        Uses semantic similarity when an embedding model is available,
        otherwise falls back to word-overlap scoring.
        """
        if self.semantic_index:
            results = self._semantic_score(query, top_k)
            _logger_module.log_snippets(results)
            return results

        # Word-overlap fallback
        scored: list[tuple[int, str, str]] = []
        for filename, text in self.documents:
            for paragraph in text.split("\n\n"):
                if len(paragraph.strip()) < 20:
                    continue
                score = self._score_paragraph(query, paragraph)
                scored.append((score, filename, paragraph))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            (filename, paragraph, score)
            for score, filename, paragraph in scored[:top_k]
            if score >= 2
        ]
        _logger_module.log_snippets(results)
        return results

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> tuple[str, list[str]]:
        """
        Ask Gemini for 3-4 related search terms.
        Returns (expanded_query_string, list_of_extra_terms).
        Falls back to (original_query, []) if LLM is unavailable.
        """
        _logger_module.log_query("expand_query", query)

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
    # Theme extraction
    # ------------------------------------------------------------------

    def extract_themes(self, snippets: list[tuple[str, str, float]]) -> list[str]:
        """
        Ask Gemini to identify 3-4 recurring themes from retrieved passages.
        Returns a list of short theme label strings.
        """
        if not self.llm_client or not snippets:
            return []

        passages = "\n\n".join(text for _, text, _ in snippets[:5])
        prompt = (
            "Given these article excerpts, identify 3-4 recurring themes or topics. "
            "Return only a comma-separated list of short theme labels (2-4 words each).\n"
            f"Excerpts: {passages}"
        )
        try:
            raw = self.llm_client.generate(prompt)
            themes = [t.strip() for t in raw.split(",") if t.strip()]
            _logger_module.log_themes(themes)
            return themes
        except Exception as exc:
            _logger_module.log_error("extract_themes", exc)
            return []

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesise(self, query: str, snippets: list[tuple[str, str, float]]) -> str:
        """
        Build context from retrieved snippets and call Gemini to produce a
        cited summary + Key Quotes section.
        Falls back to showing raw passages if the LLM is unavailable.
        """
        context_blocks = [
            f"[Source {i}: {filename}]\n{text}"
            for i, (filename, text, _) in enumerate(snippets, start=1)
        ]
        context = "\n\n".join(context_blocks)

        if self.llm_client is not None:
            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(context=context, query=query)
            try:
                answer = self.llm_client.generate(prompt)
            except Exception as exc:
                _logger_module.log_error("synthesise", exc)
                answer = "⚠️ LLM unavailable — showing raw passages:\n\n" + context
        else:
            answer = "⚠️ No LLM configured — showing raw passages:\n\n" + context

        _logger_module.log_answer(answer)
        return answer

    # ------------------------------------------------------------------
    # Claim suggestions
    # ------------------------------------------------------------------

    def suggest_claims(self, summary: str) -> list[str]:
        """
        Suggest 3 debatable claims that emerge from the summary text.
        Returns a list of up to 3 claim strings.
        """
        if not self.llm_client or not summary:
            return []

        prompt = (
            "Based on this research summary, suggest 3 debatable claims or positions "
            "that emerge from the evidence. Return only a numbered list, one claim per line.\n"
            f"Summary: {summary}"
        )
        try:
            raw = self.llm_client.generate(prompt)
            claims = []
            for line in raw.strip().splitlines():
                line = line.strip()
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                if cleaned:
                    claims.append(cleaned)
            return claims[:3]
        except Exception as exc:
            _logger_module.log_error("suggest_claims", exc)
            return []

    # ------------------------------------------------------------------
    # Argument analysis
    # ------------------------------------------------------------------

    def analyse_arguments(self, claim: str, top_k: int = 6) -> str:
        """
        Retrieve passages relevant to the claim, then ask Gemini to identify
        arguments for and against it.
        Returns a formatted FOR/AGAINST string.
        """
        if not self.llm_client:
            return "⚠️ LLM unavailable — cannot analyse arguments."

        expanded_claim, _ = self.expand_query(claim)
        snippets = self.retrieve(expanded_claim, top_k=top_k)
        if not snippets:
            return (
                "The uploaded articles do not contain enough evidence "
                "to argue this claim from either side."
            )

        context_blocks = [
            f"[Source {i}: {filename}]\n{text}"
            for i, (filename, text, _) in enumerate(snippets, start=1)
        ]
        context = "\n\n".join(context_blocks)

        prompt = ARGUMENT_PROMPT_TEMPLATE.format(claim=claim, context=context)
        try:
            result = self.llm_client.generate(prompt)
            _logger_module.log_argument_analysis(claim)
            return result
        except Exception as exc:
            _logger_module.log_error("analyse_arguments", exc)
            return "⚠️ Error during argument analysis."

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def answer_with_citations(self, query: str) -> dict:
        """
        Full pipeline: expand → retrieve → themes → synthesise.

        Returns:
        {
            "expanded_terms": list[str],
            "sources_searched": list[str],
            "passages_found": int,
            "themes": list[str],
            "answer": str,
            "raw_summary": str,
        }
        """
        expanded_query, expanded_terms = self.expand_query(query)
        snippets = self.retrieve(expanded_query, top_k=5)

        sources_searched = list(dict.fromkeys(f for f, _, _ in snippets))

        if not snippets:
            return {
                "expanded_terms": expanded_terms,
                "sources_searched": [],
                "passages_found": 0,
                "themes": [],
                "answer": (
                    "No relevant content found in your articles for this question. "
                    "Try rephrasing or uploading more sources."
                ),
                "raw_summary": "",
            }

        themes = self.extract_themes(snippets)
        answer = self.synthesise(query, snippets)

        raw_summary = answer
        if "Key Quotes:" in answer:
            raw_summary = answer.split("Key Quotes:")[0].replace("Summary:", "").strip()

        return {
            "expanded_terms": expanded_terms,
            "sources_searched": sources_searched,
            "passages_found": len(snippets),
            "themes": themes,
            "answer": answer,
            "raw_summary": raw_summary,
        }
