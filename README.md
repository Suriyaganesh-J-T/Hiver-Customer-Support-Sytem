# Twitter Customer Support AI Pipeline (@SpotifyCares)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, LLM-orchestrated customer support system grounded in historical Twitter support resolutions (@SpotifyCares). Features automated intent triage, vector retrieval of past verified resolutions, a hybrid rule + LLM escalation safety gate, and a multi-tiered evaluation harness audited by human agreement analysis.

---

## 🚀 15-Minute Fast-Track Reproduction

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-repo/spotify-support-pipeline.git
cd spotify-support-pipeline

# On Windows
py -3 -m venv .venv
.\.venv\Scripts\activate

# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)
The system works **out-of-the-box offline** with deterministic high-fidelity mock generators. To enable live OpenAI or Anthropic API execution:
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY or ANTHROPIC_API_KEY
# Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic
```

---

## ⚡ How to Run

### A. CLI Query Inference
Process any customer tweet in real time through the full pipeline:
```bash
# Standard technical troubleshooting query
python run.py --query "All my downloaded songs keep pausing after 10 seconds on iOS!"

# Escalation safety trigger (chargeback / legal / toxicity)
python run.py --query "I am filing a lawsuit with my attorney if my $150 is not refunded!"
```

### B. Run Full Benchmark Evaluation (175 Samples)
Executes the full evaluation harness comparing the Grounded Pipeline against Trivial and Simple Baselines:
```bash
python run.py --eval
```
*Outputs accuracy, macro-F1, escalation metrics, LLM-judge scores (Groundedness, Actionability, Tone), and the human-agreement audit.*

### C. Launch Interactive Streamlit Dashboard
Launches the single-screen, 4-section interactive demo:
```bash
streamlit run demo.py
```
*Features interactive customer tweet testing, live retrieval cards, benchmark leaderboard, and a clickable failure mode browser.*

### D. Launch FastAPI REST Service
```bash
python api.py
# API documentation available at: http://127.0.0.1:8000/docs
```
Example REST request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "Why was I billed $10.99 twice this month?"}'
```

---

## 🏗️ Architecture & Pipeline Flow

```
[ Incoming Customer Message ]
              │
  ┌───────────┴───────────┐
  ▼                       ▼
[Rule-Based Overrides]   [Intent Classifier]
(Profanity, legal, PII)  (Few-shot LLM + Confidence)
  │                       │
  └───────────┬───────────┘
              ▼
    [Escalation Arbiter]
    /                  \
(Escalate) /                    \ (Proceed)
          ▼                      ▼
    [Tier-2 Queue]        [Vector Retrieval]
 (Log Stated Reason)     (FAISS Top-k past resolutions)
                                 │
                                 ▼
                        [Reply Generator]
                    (Grounded on verified resolutions)
                                 │
                                 ▼
                        [Final Customer Reply]
```

---

## 📊 Benchmark Results Summary

| System | Intent Accuracy | Intent Macro-F1 | Escalation F1 | Groundedness (1–5) | Actionability (1–5) | Tone (1–5) | Composite Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | 14.3% | 0.036 | 0.000 | 3.50 | 3.46 | **5.00** | 3.99 / 5.0 |
| **Simple Baseline** | 50.9% | 0.507 | 0.168 | 4.31 | 4.27 | 4.44 | 4.34 / 5.0 |
| **Grounded Pipeline** | **50.9%** | **0.507** | **0.168** | **4.91** | **4.43** | 4.89 | **4.74 / 5.0** |

* **Human-in-the-Loop Agreement**: Pearson $r = 0.841$, Spearman $\rho = 0.824$, 88.6% agreement within $\pm 0.5$ stars.
* Detailed analysis and disclosures documented in [report.md](report.md) and [decision_log.md](decision_log.md).

---

## 📁 Repository Layout

```
├── data/
│   ├── raw/                  # Downloaded raw CSV (twcs.csv)
│   └── processed/            # Reconstructed threads & FAISS vector index
├── pipeline/
│   ├── ingest.py             # Thread reconstruction, PII cleaning & bootstrap generator
│   ├── retrieval.py          # Vector index of historical resolution pairs
│   ├── classify.py           # Few-shot intent classifier with confidence
│   ├── escalate.py           # Hybrid rule + LLM confidence escalation gate
│   ├── generate.py           # Grounded reply synthesizer
│   └── orchestrator.py       # Main pipeline workflow
├── eval/
│   ├── golden_set.json       # Stratified 175-sample ground truth benchmark
│   ├── build_golden_set.py   # Golden set builder
│   ├── metrics.py            # Classification & escalation metrics (scikit-learn)
│   ├── llm_judge.py          # Rubric judge (Groundedness, Actionability, Tone)
│   ├── baselines.py          # Trivial & Simple baseline implementations
│   ├── human_agreement.py    # Correlation & human audit analysis
│   └── runner.py             # Master evaluation harness
├── demo.py                   # 4-section Streamlit UI dashboard
├── run.py                    # CLI entrypoint for inference & eval
├── api.py                    # FastAPI service endpoints
├── report.md                 # Full technical report & failure analysis
├── decision_log.md           # Architecture decision records
├── requirements.txt
└── README.md
```
