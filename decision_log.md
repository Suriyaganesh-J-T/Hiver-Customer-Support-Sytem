# Architecture & Scope Decision Log

## 1. Brand Selection Decision
* **Options Considered**: `@AmazonHelp`, `@AppleSupport`, `@SpotifyCares`, `@Delta`.
* **Decision**: **`@SpotifyCares`**
* **Rationale**:
  * `@AmazonHelp` has massive volume (~1M+ tweets) but diffuse support surfaces (packages, third-party sellers, Kindle, AWS, grocery), introducing high noise and sparse intent clusters.
  * `@SpotifyCares` exhibits clean, repetitive, and technically actionable intent patterns (billing disputes, offline cache failures, 2FA/login, licensing grey-outs).
  * Historical agent replies from `@SpotifyCares` provide concrete troubleshooting workflows (e.g. `%localappdata%/Spotify/Storage`, `spotify.com/account/recover-playlists`, Background App Refresh) rather than generic redirects.

---

## 2. Intent Taxonomy Architecture
* **Options Considered**: Fine-grained taxonomy (15–20 intents) vs. Coarse-grained taxonomy (3–4 intents) vs. Balanced taxonomy (5–7 intents).
* **Decision**: **7 Balanced Intents**:
  1. `account_access`
  2. `billing_subscription`
  3. `playback_bug`
  4. `content_availability`
  5. `feature_hardware`
  6. `general_complaint`
  7. `other_unclear`
* **Rationale**:
  * At a golden test set target of 150–200 examples, 15+ intents would leave only 7–10 samples per class, resulting in statistically fragile per-class F1 metrics.
  * 7 intents allows ~25 stratified samples per intent class, providing robust support without collapsing domain nuance into an unhelpful binary.

---

## 3. "Grounded in Historical Resolution" Retrieval
* **Options Considered**:
  1. Zero-shot prompt generation.
  2. Document RAG over generic static help articles.
  3. Few-shot retrieval over verified historical customer-agent resolution pairs.
* **Decision**: **Historical Resolution Grounding (Approach 3)**.
* **Rationale**:
  * Static help articles are frequently too long and poorly matched to Twitter's 280-character concise format.
  * Indexing resolved historical customer tweets and pairing them with verified agent replies allows the system to ground new replies in actual proven Twitter resolutions.
  * In the benchmark, grounding increased composite response quality from **4.34 to 4.74** and groundedness from **4.31 to 4.91**.

---

## 4. Escalation Gate Architecture: Rule + LLM Hybrid
* **Options Considered**: Pure LLM classification vs. Pure rule-based filters vs. Hybrid Gate.
* **Decision**: **Hybrid Dual-Track Escalation Gate**.
* **Rationale**:
  * High-risk paths (legal threats, chargebacks, severe toxicity, explicit human requests) cannot safely rely on probabilistic LLM classification.
  * Rules guarantee deterministic safety overrides (`RULE_LEGAL_THREAT`, `RULE_CHARGEBACK_DISPUTE`, `RULE_TOXICITY_PROFANITY`, `RULE_FINANCIAL_THRESHOLD`).
  * The probabilistic layer catches nuanced cases where confidence falls below safety thresholds ($< 0.65$) or when conversation turn depth exceeds 2 rounds.

---

## 5. Technology Stack Choices
* **Streamlit vs. Next.js/React**:
  * Streamlit was selected because this is an evaluation and pipeline-orchestration deliverable. Building a separate React/Next.js frontend would consume valuable time on CORS, API routing, and state syncing. Streamlit provides an interactive, live-walkthrough UI that imports pipeline code directly.
* **Vector Store**:
  * Standard local TF-IDF / FAISS index was chosen over cloud databases (e.g. Pinecone). This guarantees that any evaluator can clone the repo and run the entire suite offline in under 15 minutes without infrastructure credentials.
* **LLM Provider Flexibility**:
  * Abstracted to support OpenAI, Anthropic, and an offline deterministic synthesis engine. This ensures the entire repository, test suite, and evaluation harness run out-of-the-box without requiring an API key.
