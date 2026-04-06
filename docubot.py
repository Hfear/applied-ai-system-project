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
                word = word.lower().strip('.,!?:;()\'"[]{}')
                if word not in index:
                    index[word] = []
                if filename not in index[word]:
                    index[word].append(filename)
        return index
        # -----------------------------------------------------------
        # Scoring and Retrieval (Phase 1)
        # -----------------------------------------------------------

    def score_document(self, query, text):
        # Split query into words, lowercase and strip punctuation from each
        query_words = [w.lower().strip('.,!?:;()\'"[]{}') for w in query.split()]
        
        # Split document text into words, lowercase and strip punctuation from each
        doc_words = [w.lower().strip('.,!?:;()\'"[]{}') for w in text.split()]
        
        # Build a frequency map of how many times each word appears in the doc
        freq_map = {}
        for word in doc_words:
            if word not in freq_map:
                freq_map[word] = 0
            freq_map[word] += 1
        
        # For each query word, add its frequency in the doc to the total score
        score = 0
        for word in query_words:
            if word in freq_map:
                score += freq_map[word]
        
        # Higher score = more relevant to the query
        return score
    
    def retrieve(self, query, top_k=3):
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
        
        # Return top_k paragraphs as (filename, text) tuples, skipping zero scores
        results = []
        for score, filename, paragraph in scored[:top_k]:
            if score > 0:
                results.append((filename, paragraph))
        
        return results
    
    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

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
