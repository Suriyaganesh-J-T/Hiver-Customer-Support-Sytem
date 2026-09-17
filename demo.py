"""
Interactive Streamlit Dashboard for Twitter Customer Support Pipeline (@SpotifyCares).
Single-screen, 4-section layout:
1. Customer Inquiry Input & Golden Set presets
2. Live Pipeline Inference & Historical Grounding
3. Comprehensive Evaluation Panel & Baseline Comparison
4. Failure Mode Browser & Hypotheses Analysis
"""

import json
import os
import streamlit as st
import pandas as pd

from pipeline.orchestrator import SupportPipeline
from eval.runner import run_full_evaluation

# Page configuration
st.set_page_config(
    page_title="Spotify Support AI Pipeline",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Spotify Dark Mode Aesthetic)
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #121212;
    }
    .metric-card {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #282828;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .badge-escalated {
        background-color: #E22134;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-auto {
        background-color: #1DB954;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .retrieval-box {
        background-color: #181818;
        border-left: 4px solid #1DB954;
        padding: 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .reply-box {
        background-color: #242424;
        border-radius: 8px;
        padding: 14px;
        font-size: 1.05rem;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return SupportPipeline()


@st.cache_data
def load_golden_set():
    path = "eval/golden_set.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_eval_results():
    path = "eval/eval_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


pipeline = load_pipeline()
golden_set = load_golden_set()
eval_data = load_eval_results()

# Sidebar: Environment & Settings
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=60)
    st.title("Support Pipeline")
    st.caption("Grounded In-Context LLM Orchestration")
    st.divider()

    st.subheader("System Status")
    st.markdown(f"**Target Brand**: `@SpotifyCares`")
    st.markdown(f"**Indexed Resolutions**: `{len(pipeline.index.pairs)} historical pairs`")
    st.markdown(f"**Golden Set**: `{len(golden_set)} curated samples`")
    st.markdown(f"**Escalation Gate**: `Rule + LLM Hybrid`")

    st.divider()
    turn_count = st.slider("Conversation Turn Depth", min_value=1, max_value=5, value=1, help="Turns > 2 trigger automatic escalation.")
    st.caption("Built with FAISS/Vector Retrieval, Scikit-learn, and Grounded Few-Shot Prompts.")

# Main Interface Title
st.title("🎧 Twitter Customer Support AI Pipeline")
st.markdown("Automated intent triage, historical resolution grounding, safety gating, and LLM-as-judge evaluation.")

st.write("")

# -------------------------------------------------------------
# SECTION 1 & 2: INPUT & LIVE PIPELINE INFERENCE
# -------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Customer Inquiry Input")

    # Preset selector from real golden set
    preset_options = ["Custom Input"] + [f"[{item['id']}] {item['customer_text'][:70]}..." for item in golden_set[:25]]
    selected_preset = st.selectbox("Select sample ticket from Golden Set:", preset_options)

    if selected_preset == "Custom Input":
        default_text = "Why was I charged $10.99 twice on my debit card this month? Please refund."
    else:
        sample_id = selected_preset.split("]")[0].replace("[", "")
        matched = next((item for item in golden_set if item["id"] == sample_id), None)
        default_text = matched["customer_text"] if matched else ""

    customer_input = st.text_area("Customer Tweet:", value=default_text, height=110)

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_btn = st.button("🚀 Process Inquiry", use_container_width=True, type="primary")
    with col_btn2:
        esc_sample_btn = st.button("⚠️ Load Escalation Trigger Sample", use_container_width=True)
        if esc_sample_btn:
            customer_input = "I will sue Spotify with my attorney if you don't refund my $150 immediately, you scammers!"
            st.rerun()

with col_right:
    st.subheader("2. Live Pipeline Execution")

    if run_btn or customer_input:
        with st.spinner("Classifying, checking safety rules, retrieving grounding, and synthesizing reply..."):
            out = pipeline.process(customer_input, turn_count=turn_count, top_k=3)

        # Top row metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted Intent", out.predicted_intent)
        with m2:
            st.metric("Confidence", f"{out.confidence * 100:.1f}%")
        with m3:
            status_html = (
                f'<div class="badge-escalated">ESCALATED ({out.severity.upper()})</div>'
                if out.is_escalated else
                '<div class="badge-auto">AUTO-RESOLVE</div>'
            )
            st.markdown(f"**Escalation Gate**<br>{status_html}", unsafe_allow_html=True)

        if out.is_escalated:
            st.warning(f"**Escalation Reason:** {out.escalation_reason} *(Rule: `{out.rule_triggered}`)*")

        # Grounded Reply Box
        st.markdown("**Generated Support Reply:**")
        st.markdown(f'<div class="reply-box">💬 {out.reply}</div>', unsafe_allow_html=True)

        # Historical Grounding Context
        st.write("")
        st.markdown(f"**Top Retrieved Historical Resolutions (FAISS/Cosine Grounding):**")
        for i, res in enumerate(out.retrieved_resolutions, 1):
            st.markdown(f"""
            <div class="retrieval-box">
                <b>Match {i} (Similarity: {res['similarity_score']:.4f})</b><br>
                <b>Past Customer:</b> <i>"{res['customer_query']}"</i><br>
                <b>Verified Resolution:</b> {res['agent_resolution']}
            </div>
            """, unsafe_allow_html=True)

st.divider()

# -------------------------------------------------------------
# SECTION 3: EVALUATION PANEL & BASELINES
# -------------------------------------------------------------
st.subheader("3. Evaluation Panel & Benchmark Comparisons")
st.caption("Evaluated across the 175-sample stratified Golden Set against Trivial and Simple baselines.")

col_eval_btn, col_eval_stat = st.columns([1, 3])
with col_eval_btn:
    re_eval_btn = st.button("🔄 Re-Run Full Benchmark (175 Samples)")
    if re_eval_btn:
        with st.spinner("Running 175 samples through all 3 systems + LLM-as-judge..."):
            eval_data = run_full_evaluation()
            st.success("Benchmark completed and updated!")

if eval_data:
    df_summary = pd.DataFrame(eval_data["summary_table"])
    st.dataframe(
        df_summary.set_index("System"),
        use_container_width=True
    )

    # Detailed metrics expander
    with st.expander("📊 Detailed Human-in-the-Loop Judge Agreement Audit (35-sample Sub-set)"):
        ha = eval_data.get("human_agreement", {})
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Pearson Correlation (r)", f"{ha.get('pearson_r', 0.0):.3f}")
        with h2:
            st.metric("Spearman Rank (rho)", f"{ha.get('spearman_rho', 0.0):.3f}")
        with h3:
            st.metric("Agreement within ±0.5★", f"{ha.get('agreement_within_0_50', 0.0)}%")
        with h4:
            st.metric("Mean Absolute Error", f"{ha.get('mean_absolute_error', 0.0):.3f}")

        st.markdown("**Key Audit Takeaways:**")
        for f in ha.get("findings", []):
            st.markdown(f"- {f}")

st.divider()

# -------------------------------------------------------------
# SECTION 4: FAILURE MODE BROWSER
# -------------------------------------------------------------
st.subheader("4. Failure Mode Browser & Hypotheses Analysis")
st.caption("Deep inspection of lowest-scoring edge cases, intent ambiguities, and rule anomalies.")

if eval_data and "failure_analysis" in eval_data:
    failures = eval_data["failure_analysis"]
    df_fail = pd.DataFrame(failures)

    cat_filter = st.multiselect(
        "Filter by Failure Mode:",
        options=list(df_fail["failure_category"].unique()),
        default=list(df_fail["failure_category"].unique())
    )

    filtered_failures = [f for f in failures if f["failure_category"] in cat_filter]

    for item in filtered_failures:
        with st.expander(f"[{item['failure_category']}] Sample {item['id']}: \"{item['customer_text'][:60]}...\" (Composite: {item['composite_score']}/5.0)"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Customer Message:** {item['customer_text']}")
                st.markdown(f"**Actual Intent:** `{item['ground_truth_intent']}`")
                st.markdown(f"**Predicted Intent:** `{item['predicted_intent']}`")
            with c2:
                st.markdown(f"**Generated Reply:** {item['reply']}")
                st.markdown(f"**Hypothesis / Root Cause:** {item['hypothesis']}")
