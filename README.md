# Article Analyser

Upload articles, ask questions, get cited summaries grounded in your sources.

---

## Base Project

This extends **DocuBot** from the Module 3 tinker activity. DocuBot answered developer questions about a codebase by searching a fixed folder of markdown files using keyword scoring and Gemini for generation. It supported three modes: naive LLM, retrieval-only, and basic RAG.

---

## Summary

Article Analyser lets users upload any collection of articles and ask research questions. The system retrieves evidence directly from the uploaded documents and writes a cited summary using only that evidence — no outside knowledge, no hallucination.

---

## Architecture Overview

Five components in sequence:

1. **File Parser** — accepts `.txt`, `.md`, `.pdf` uploads and extracts raw text
2. **DocuBot Index** — builds a word-level index, splits articles into paragraphs
3. **Query Expansion Agent (Gemini call 1)** — expands the user's question into 3-4 related search terms to improve retrieval recall
4. **Retriever** — scores paragraphs against the expanded query, returns top 5 passages with source attribution
5. **Synthesis Agent (Gemini call 2)** — writes a cited summary using only retrieved passages; refuses if evidence is insufficient

Logger records every query, retrieval, and answer to `docubot.log`. All three steps are displayed visibly in the UI.

---

## Setup

```bash
git clone https://github.com/Hfear/applied-ai-system-project.git
cd applied-ai-system-project
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```
Free key at: https://aistudio.google.com/app/api-keys

Without a key the app runs in retrieval-only mode.

```bash
streamlit run app.py
```

---

## Sample Interactions

**Query 1: "What are the risks of loot boxes for young people?"**

Expanded terms: Loot boxes risks, young people gambling, loot boxes addiction, minors harm

Sources: loot_boxes_academic.txt, guardian_skins_gambling.txt

Summary: Loot boxes are gambling-like elements in video games that carry potential harms, especially for young users [Source 1, Source 2]. The typical loot box user is young and likely to be a problem gambler [Source 1]. These mechanisms have been described as "rotting young people's brains" [Source 4]. Harms are linked to permanent availability and unlimited purchases [Source 1].

Key Quotes:
- "The typical loot box user is young, employed, has a low level of education but an average household income. They are likely to be problem gamblers." — [Source 1: loot_boxes_academic.txt]
- "I wanted to show how unregulated gambling is rotting the brains of young people." — [Source 4: guardian_skins_gambling.txt]

---

**Query 2: "How is AI used in online gambling?"**

Expanded terms: artificial intelligence casino, machine learning gambling, AI personalization betting

Sources: tech_impact_gambling.txt, technology_gambling_friend_or_foe.txt

Summary: AI and machine learning analyze player behavior to deliver tailored experiences [Source 1]. AI detects risky gambling patterns and sends alerts to promote responsible play [Source 1, Source 2]. Platforms also use AI for fraud detection and wagering strategy personalization [Source 2].

---

**Query 3: "Is gambling good?" — guardrail demo**

Output: "The available articles do not contain enough information to answer this question."

---

## Design Decisions

**Query expansion over raw matching** — the original DocuBot failed when user words didn't match document words exactly. Expansion bridges that gap and measurably improves recall on vague queries.

**Two Gemini calls** — separating expansion from synthesis gives each call one job, making each step independently debuggable and observable.

**Streamlit** — makes the three-step pipeline visible to the user. Transparency was a deliberate design priority.

**Trade-offs:** Word-overlap retrieval is fast but misses semantic matches. Two calls add ~2-3 seconds latency. Scanned PDFs fail extraction — graceful degradation returns no results and the guardrail fires.

---

## Testing Summary

- Parse .txt files — pass
- Parse text-based PDF — pass
- Scanned/image PDF — graceful failure, guardrail fires correctly
- Strong keyword query — correct sources retrieved
- No relevant content query — guardrail refuses as expected
- Query expansion recall — broader terms find more passages than raw query
- No Gemini key fallback — raw passages shown instead of summary
- Logging — all queries and answers recorded to docubot.log

7 of 8 passed. Main failure is scanned PDFs — expected and handled by the guardrail.

---

## Reflection

The biggest lesson was that retrieval quality matters more than prompt quality. Every improvement to how the system searches had more impact on answer quality than any change to how it generates. Building the visible three-step UI also changed how I think about AI — making reasoning transparent is a design decision, not an afterthought.