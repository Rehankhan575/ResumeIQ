# Stage 7a: Generative AI Feedback & Grounding Architecture

## 1. Overview & Objectives

Stage 7a delivers actionable, personalized career feedback to job candidates by augmenting ATS diagnostic scoring with natural language generative enhancements:
1. **Experience Bullet Point Rewriting**: Transforming weak, passive, or unquantified bullet points into high-impact STAR-format achievements.
2. **Tailored Cover Letter Drafting**: Synthesizing candidate background and target job requirements into concise, professional cover letters.
3. **Prioritized Improvement Plans**: Translating component-wise ATS deficiencies into concrete, step-by-step remediation advice.

A critical design requirement is **zero hallucination of qualifications**: the system must never invent employers, metrics, dates, or skills.

---

## 2. Provider Abstraction & Fallback Engine

The module (`app/genai/provider.py`) exposes a unified provider interface dynamically configured via environment variables read from `.env` or `os.environ`:

| Provider | Environment Variables | Default Model | Endpoint |
| :--- | :--- | :--- | :--- |
| **Google Gemini** | `GEMINI_API_KEY`, `GEMINI_MODEL` | `gemini-1.5-flash` | Google Generative Language REST API |
| **Groq** | `GROQ_API_KEY`, `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq OpenAI-compatible Chat Completions API |
| **Ollama (Local)** | `OLLAMA_HOST`, `OLLAMA_MODEL` | `llama3` | Ollama local REST API (`/api/generate`) |
| **Rule-Based Fallback** | *None required* | *Deterministic heuristics* | Pure offline execution |

### Operational Guarantees
- **Strict 8-Second Timeout**: Every network request enforces an 8.0-second timeout (`timeout=8.0`).
- **Bounded Retries**: At most 2 retries (up to 3 total attempts) with immediate fallback upon persistent failure or timeout.
- **In-Memory Caching**: Responses are cached strictly in memory using a SHA-256 hash of `f"{provider}:{model}:{prompt}"`. Cache entries are transient and never written to disk.
- **Privacy & Security**: Zero logging of API keys, candidate names, resume text, or generated letters. Logs record only operational metadata.

---

## 3. Grounding Guard (`check_grounding`)

To enforce strict factual accuracy, all provider rewrites pass through the `check_grounding(original, rewritten)` guard (`app/genai/grounding.py`). The guard inspects the rewritten text and flags:

1. **Invented Numbers & Quantifiers**: Any number, percentage, or currency figure present in the rewrite that did not appear in the original source text. (Explicit placeholders like `[add metric]` are allowed).
2. **Invented Organizations / Employers**: Any capitalized acronym (e.g., `AWS`, `IBM`) or Titlecase proper noun (e.g., `Google`, `Amazon`, `McKinsey`) not present in the original bullet.
3. **Invented Taxonomy Skills**: Any canonical skill or alias from `data/taxonomies/skills.csv` that was not in the original bullet.

### Policy on Flagged Rewrites
If the Grounding Guard flags any ungrounded entity, **the provider's output is dropped entirely**. The system falls back to the deterministic rule-based rewrite, retaining original facts with an actionable `[add metric]` placeholder.

---

## 4. Deterministic Rule-Based Fallbacks

When no LLM provider is configured, or when a network call fails, times out, or violates grounding:
1. **Bullet Rewrites**:
   - Replaces leading passive/weak verbs (`worked on`, `helped with`, `responsible for`) with strong active power verbs (`engineered`, `spearheaded`, `architected`) from `app/config.py:ACTION_VERBS`.
   - If the bullet contains no numbers or metrics, appends `, achieving [add metric]`.
2. **Cover Letter**:
   - Fills a structured executive cover letter template using parsed contact details, primary skills, and the most recent role from `ParsedResume`.
3. **Improvement Plan**:
   - Assembles recommendations directly from `ScoreResult.issues`, sorted descending by estimated score gain (`Gain = w_i * (1 - s_i) * 100`).
