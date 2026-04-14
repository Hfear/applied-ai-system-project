# Article Analyser Model Card

A reflection on the Article Analyser system — an extension of DocuBot built for the Module 4 final project.

---

## 1. System Overview

**What is Article Analyser trying to do?**

Article Analyser is an AI research assistant that takes in uploaded articles and answers questions grounded only in those sources. Instead of relying on a model's general knowledge, it retrieves actual passages from the uploaded documents and uses them as evidence before generating any answer.

**What inputs does it take?**

Uploaded .txt, .md, or .pdf files, a Gemini API key, and a text query from the user.

**What outputs does it produce?**

A cited summary paragraph with inline source references and a Key Quotes section with direct quotes attributed to their source files. If there is not enough evidence, it refuses to answer instead of guessing.

---

## 2. Retrieval Design

**How does the retrieval system work?**

- Documents are split into paragraphs and each word is lowercased and stripped of punctuation
- A word-level index maps terms to the files they appear in
- When a query comes in, the system expands it using Gemini to generate 3-4 related search terms
- Every paragraph across all uploaded files gets scored against the expanded query using prefix matching
- Stop words are filtered out so common words like "the" and "is" do not inflate scores
- Top 5 paragraphs with a score of 2 or above get passed to the synthesis agent

**What tradeoffs were made?**

- Query expansion adds a Gemini call before retrieval but measurably improves recall on vague queries — worth the latency
- Word overlap scoring is fast and transparent but misses semantic matches that embeddings would catch
- Score threshold of 2 keeps noise out but might filter valid low-scoring answers on niche queries
- Splitting on paragraph breaks means answers that span two paragraphs can still be missed

---

## 3. Use of the LLM (Gemini)

**When does the system call Gemini and when does it not?**

- Query expansion: Gemini is called first to expand the user's search terms before any retrieval happens
- Synthesis: Gemini is called a second time with only the retrieved passages to write the cited summary
- Retrieval-only fallback: if no API key is present, Gemini is never called and raw passages are returned directly

**What instructions keep the LLM grounded?**

- Only use information from the retrieved passages
- Do not add outside knowledge
- Cite which source each claim came from using inline brackets
- If the passages do not contain enough information, say exactly "The available articles do not contain enough information to answer this question"

---

## 4. Limitations and Biases

The system is only as good as what gets uploaded. If the uploaded articles are one-sided or cover only one perspective on a topic, the answers will reflect that without any warning to the user. There is no mechanism to flag when sources are biased or incomplete.

Word overlap retrieval also has a language bias — it works best with plain English and struggles with technical jargon, acronyms, or non-English text. A query using different vocabulary than the source document will score low even if the meaning is identical.

Scanned PDFs fail entirely since the parser cannot extract text from image-based documents. The guardrail handles this gracefully but the user might not understand why their file returned no results.

---

## 5. Misuse and Prevention

Someone could use this system to cherry-pick quotes from articles by uploading only sources that support a particular argument, then presenting the cited summary as if it represents a balanced view. The system has no way to detect this because it only knows about the files it was given.

To reduce this risk, the UI is transparent about which sources were used for each answer. Users can see exactly which files the passages came from, so anyone reading the output can judge whether the sourcing is representative. Adding a disclaimer to outputs noting that answers reflect only the uploaded sources would be a straightforward future improvement.

---

## 6. Testing Surprises

The guardrail firing on "is gambling good?" was the most interesting result during testing. The articles contain plenty of content about gambling but the system correctly identified that none of it directly supports a positive framing of the question. That was a better result than expected — it showed the synthesis agent was actually reasoning about the evidence rather than just pattern matching.

The query expansion also surprised me in a good way. "Is online betting dangerous?" with no expansion returned one weak passage. After expansion to "online betting risks, compulsive gambling, gambling addiction" it returned five strong passages from two different sources. That was the clearest demonstration that the agentic step was doing real work.

---

## 7. AI Collaboration

This project was built with significant AI assistance throughout.

One instance where AI gave a genuinely helpful suggestion: when designing the agentic pipeline, the suggestion to separate query expansion from synthesis into two distinct Gemini calls was the right call. It made each step independently debuggable and made the observable pipeline much cleaner than a single combined prompt would have been.

One instance where the suggestion was flawed: early on, the generated docubot.py loaded documents from a hardcoded articles/ folder and passed them as a list of tuples. When Streamlit was introduced, the code assumed documents would always come from the folder and broke when files were passed directly from the uploader through session state. The fix required rethinking the constructor to accept documents either from a folder or directly as a parameter — something the original suggestion did not account for.

---

## 8. Responsible Use

The system could cause harm if a user treats the cited summary as a complete or authoritative answer without checking the source files. The citations make the evidence traceable but do not guarantee the uploaded articles themselves are accurate or representative.

- Always check the source files the system cites before acting on an answer
- Upload multiple perspectives on a topic rather than a single source
- Treat "no relevant content found" as useful information, not a system failure
- Do not use this system as a substitute for reading the original articles on high-stakes decisions