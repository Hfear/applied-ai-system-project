"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        #lists of common stop words to ignore in retrieval scoring (can be expanded)
        self.stop_words = {
        "is", "there", "any", "of", "the", "a", "an", "in", "to",
        "and", "or", "how", "do", "i", "where", "what", "which",
        "when", "was", "are", "it", "this", "that", "for", "with",
        "my", "we", "be", "at", "by", "from", "on", "not", "if",
        "are", "were", "been", "being", "have", "has", "had",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "shall", "its", "these", "those", "you",
        "he", "she", "they", "your", "his", "her", "our", "their",
        "so", "than", "as", "into", "through", "during", "before",
        "after", "no", "but"
        }

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)


    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        index = {}
        for filename, text in documents:
            for word in text.split():
                word = word.lower().strip('.,!?:;()\'"[]{}``')
                if word not in index:
                    index[word] = []
                if filename not in index[word]:
                    index[word].append(filename)
        return index
        # -----------------------------------------------------------
        # Scoring and Retrieval (Phase 1)
        # -----------------------------------------------------------

    def score_document(self, query, text):
        # Clean and split query into unique meaningful words, skipping stop words
        query_words = set(
            w.lower().strip('.,!?:;()\'"[]{}``')
            for w in query.split()
            if w.lower() not in self.stop_words
        )

        # Clean and split document text into unique words
        text_words = set(
            w.lower().strip('.,!?:;()\'"[]{}``')
            for w in text.split()
        )

        # Score = number of query words that have at least one prefix match in the paragraph.
        # One direction only: text_word.startswith(query_word).
        # This handles stemming gaps ("token"→"tokens", "generate"→"generated", "auth"→"auth_utils")
        # without the false positives caused by the reverse check ("to" matching "token").
        score = 0
        for query_word in query_words:
            for text_word in text_words:
                if text_word.startswith(query_word):
                    score += 1
                    break  # only count each query word once even if multiple text words match

        return score
    
    def retrieve(self, query, top_k=5):
        # Score every paragraph from every document against the query
        scored = []
        for filename, text in self.documents:
            # Split the document into paragraphs on blank lines
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                # Skip paragraphs that are too short to be meaningful
                if len(paragraph.strip()) < 20:
                    continue
                # Score this paragraph against the query
                score = self.score_document(query, paragraph)
                scored.append((score, filename, paragraph))
        
        # Sort all paragraphs by score, highest first
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k paragraphs, skipping anything scoring below 2
        # Score of 1 means only one query term matched — too noisy to be useful.
        results = []
        for score, filename, paragraph in scored[:top_k]:
            if score >= 2:
                # Include score so answer_retrieval_only can evaluate confidence
                results.append((filename, paragraph, score))
        
        return results
        
    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=5):
        snippets = self.retrieve(query, top_k=top_k)

        # No results at all — outright refuse
        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text, score in snippets:
            # Low confidence warning for scores below 2
            # With set-based scoring, max score = unique meaningful query terms,
            # so a score of 1 means only one term matched — low confidence.
            if score < 2:
                confidence_note = f"[LOW] Low confidence (score: {score}) — answer may be inaccurate"
            else:
                confidence_note = f"[OK] Confidence score: {score}"

            # Format each snippet with its filename and confidence note
            formatted.append(f"[{filename}] {confidence_note}\n{text}\n")

        return "\n---\n".join(formatted)


    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
