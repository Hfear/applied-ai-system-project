# DocuBot Model Card

This model card is a short reflection on your DocuBot system. Fill it out after you have implemented retrieval and experimented with all three modes:

1. Naive LLM over full docs  
2. Retrieval only  
3. RAG (retrieval plus LLM)

Use clear, honest descriptions. It is fine if your system is imperfect.

---

## 1. System Overview

**What is DocuBot trying to do?**  
Describe the overall goal in 2 to 3 sentences.

DocuBot is an AI used to take in a question and search through documents and return references related to the query. Unlike other AIs, it focuses on the quotes and references in an attempt to stave off hallucinations.

**What inputs does DocuBot take?**  
For example: user question, docs in folder, environment variables.

DocuBot takes in documents, an API key, and a text input from the user.

**What outputs does DocuBot produce?**

 DocuBot returns the top results along with confidence scores and a reliability indicator to help the user judge the answer.

---

## 2. Retrieval Design

**How does your retrieval system work?**  
Describe your choices for indexing and scoring.

- How do you turn documents into an index?
- How do you score relevance for a query?
- How do you choose top snippets?

 - Split documents on whitespace, lowercase everything, strip punctuation
 - Words get mapped to whichever files they appear in, one word can reference multiple files
 - Documents get split into paragraphs so we return focused chunks not whole files
 - Each paragraph gets scored by counting how many unique query words appear in it
 - Stop words like "the", "is", "a" get skipped so they don't inflate scores
 - Top scoring paragraphs get returned with their filename and score

**What tradeoffs did you make?**  
For example: speed vs precision, simplicity vs accuracy.

 - Chose unique word matching over frequency map so repetition does not inflate scores, but lost the ability to rank docs that focus heavily on a topic higher
 - Prefix matching handles stemming gaps like token/tokens but can still miss matches or create false ones
 - Score threshold of 2 filters out noise but may block valid low scoring answers
 - Keeping it simple with basic Python means fast and readable code but less accurate than semantic search

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**  
Briefly describe how each mode behaves.

- Naive LLM mode:
- Retrieval only mode:
- RAG mode:

 - Naive LLM mode: calls Gemini with only the user query, no docs involved
 - Retrieval only mode: never calls the LLM, returns raw paragraphs directly
 - RAG mode: retrieves snippets first then passes only those to Gemini

**What instructions do you give the LLM to keep it grounded?**  
Summarize the rules from your prompt. For example: only use snippets, say "I do not know" when needed, cite files.

 - Only answer using the retrieved snippets
 - Do not invent functions, endpoints, or config values
 - Cite which file the answer came from
 - Say "I do not know based on the docs I have" only if snippets have no relevant information at all

---

## 4. Experiments and Comparisons

Run the **same set of queries** in all three modes. Fill in the table with short notes.

You can reuse or adapt the queries from `dataset.py`.

| Query | Naive LLM: helpful or harmful? | Retrieval only: helpful or harmful? | RAG: helpful or harmful? | Notes |
|------|---------------------------------|--------------------------------------|---------------------------|-------|
| Where is the auth token generated? | Harmful - invented OAuth/Okta, never mentioned generate_access_token | Helpful - found correct paragraph but buried 5th | Partial - answered but cited wrong snippet | Right paragraph tied with irrelevant ones |
| Which endpoint returns all users? | Harmful - guessed GET /users, docs say GET /api/users | Helpful - top result was exactly right | Partial - found right file but missed exact path | Full endpoint was in a different paragraph |
| Is there any mention of payment processing? | Harmful - invented payment info in some runs | Helpful - correctly refused | Helpful - correctly refused | Guardrail worked as intended |

**What patterns did you notice?**  

- When does naive LLM look impressive but untrustworthy?  
- When is retrieval only clearly better?  
- When is RAG clearly better than both?

 - Naive LLM looks impressive on generic questions but dangerous on codebase specific ones, never cites evidence
 - Retrieval only is most trustworthy since every word came directly from the docs but requires user to interpret raw paragraphs
 - RAG is best when retrieval surfaces one clear focused snippet, fails when answer is spread across multiple paragraphs

---

## 5. Failure Cases and Guardrails

**Describe at least two concrete failure cases you observed.**  
For each one, say:

- What was the question?  
- What did the system do?  
- What should have happened instead?

 Failure case 1: "Where is the auth token generated?" - retrieved the right paragraph but ranked it 5th, RAG used a less relevant snippet and gave an incomplete answer. It should have ranked the generate_access_token paragraph first and used that as the primary source.

 Failure case 2: "Which endpoint returns all users?" - found the description but missed the endpoint path GET /api/users which was in a separate paragraph that was not retrieved. It should have returned both paragraphs together so the full answer was available.

**When should DocuBot say "I do not know based on the docs I have"?**  
Give at least two specific situations.

 - When no paragraph scores above the minimum threshold meaning no meaningful word overlap exists between the query and any documentation
 - When the question is about a topic entirely absent from the docs like payment processing

**What guardrails did you implement?**  
Examples: refusal rules, thresholds, limits on snippets, safe defaults.

 - Score threshold - paragraphs below 2 are never returned
 - Stop word filtering - common words excluded from scoring so they cannot inflate relevance
 - Confidence labels - scores below 5 show a LOW warning so users know to verify
 - LLM refusal instruction - Gemini told to refuse rather than guess when snippets lack relevant info

---

## 6. Limitations and Future Improvements

**Current limitations**  
List at least three limitations of your DocuBot system.

1. Scoring based on word overlap only, cannot understand meaning so similar questions with different words retrieve different results
2. Answers split across multiple paragraphs are often missed since retrieval returns the best single paragraph not a synthesized view
3. Prefix matching is a rough approximation and can produce false matches between short and unrelated longer words

**Future improvements**  
List two or three changes that would most improve reliability or usefulness.

1. Use semantic embeddings instead of word overlap to match meaning rather than exact words
2. Include surrounding paragraphs with each retrieved snippet so answers split across paragraphs are not lost
3. Add query expansion to automatically add related terms before retrieval to improve recall on technical terminology

---

## 7. Responsible Use

**Where could this system cause real world harm if used carelessly?**  
Think about wrong answers, missing information, or over trusting the LLM.

 A developer could trust a Mode 1 answer that sounds correct but is based on general knowledge not the actual codebase, leading to wrong endpoint paths, wrong environment variable names, or misconfigured deployments.

**What instructions would you give real developers who want to use DocuBot safely?**  
Write 2 to 4 short bullet points.

- Always verify answers against the actual source files especially for security related config like AUTH_SECRET_KEY
- Never use Mode 1 for project specific questions since it has no access to your actual documentation
- Treat LOW confidence scores as a signal to check the docs manually before acting
- Keep documentation up to date since DocuBot is only as accurate as the files in the docs folder

---