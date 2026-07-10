import json
from pathlib import Path

import altair as alt
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Cost Comparison — RoutellM", page_icon="", layout="wide")

from utils import api_client
from utils.charts import cost_comparison_chart


st.markdown("""
<style>
    .metric-card { background: #1a1a2e; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #2d2d5e; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .savings-card { background: #1a3a2e; border-radius: 12px; padding: 24px; text-align: center; border: 2px solid #2ecc71; }
    .savings-value { font-size: 2.5rem; font-weight: 700; color: #2ecc71; }
    .savings-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .stApp { background: #0f0f23; }
    h1, h2, h3 { color: #eee !important; }
    .stButton button { background: #2ecc71; color: #000; font-weight: 600; border: none; border-radius: 8px; }
    .stButton button:hover { background: #27ae60; color: #fff; }
</style>
""", unsafe_allow_html=True)

st.title(" Cost Comparison")
st.markdown("See how much RoutellM saves vs using a single expensive model for every task.")

BASELINE_MODELS = {
    "Kimi K2P7 Code ($1.00/M tok)": "accounts/fireworks/models/kimi-k2p7-code",
    "Minimax M3 ($0.50-2.00/M tok)": "accounts/fireworks/models/minimax-m3",
    "Gemma 3 4B IT ($0.20/M tok)": "accounts/fireworks/models/gemma-3-4b-it",
    "Gemma 3 1B IT ($0.10/M tok, cheapest)": "accounts/fireworks/models/gemma-3-1b-it",
}

if "cost_result" not in st.session_state:
    st.session_state["cost_result"] = None

SAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "tasks_track1_sample.json"

uploaded = st.file_uploader("Upload tasks JSON (same format as Evaluate)", type=["json"], key="cost_upload")

col1, col2, col3 = st.columns(3)
with col1:
    baseline_name = st.selectbox("Baseline model (for comparison)", list(BASELINE_MODELS.keys()), index=0)
with col2:
    dry_run = st.toggle("Dry Run (no API calls)", value=True, key="cost_dry")
with col3:
    use_sample = st.checkbox("Use sample 24-task fixture", value=False, key="cost_sample")

if st.button(" Compare", type="primary", use_container_width=True):
    if not uploaded and not use_sample:
        st.warning("Please upload a JSON file or check 'Use sample fixture'.")
    else:
        try:
            if use_sample:
                with open(SAMPLE_PATH, "r") as f:
                    tasks = json.load(f)
            else:
                tasks = json.load(uploaded)
                if not isinstance(tasks, list):
                    st.error("JSON must be an array.")
                    st.stop()

            with st.spinner("Comparing via backend API..."):
                baseline_model = BASELINE_MODELS[baseline_name]
                result = api_client.cost_compare(tasks, baseline_model=baseline_model, dry_run=dry_run)
                st.session_state["cost_result"] = {
                    "routellm_cost": result["routellm_result"]["cost"]["total_cost_usd"],
                    "routellm_tokens": result["routellm_result"]["cost"]["total_tokens"],
                    "baseline_cost": result["baseline"]["total_cost_usd"],
                    "baseline_tokens": result["baseline"]["total_tokens"],
                    "baseline_name": baseline_name,
                    "total_tasks": result["routellm_result"]["summary"]["total_tasks"],
                    "routellm_api_calls": result["routellm_result"]["cost"]["api_calls"],
                    "det_count": result["routellm_result"]["summary"]["source_breakdown"].get("deterministic", 0),
                    "api_count": result["routellm_result"]["summary"]["source_breakdown"].get("fireworks_api", 0),
                    "dry_count": result["routellm_result"]["summary"]["source_breakdown"].get("dry_run", 0),
                    "accuracy": result["routellm_result"]["summary"]["accuracy_pct"],
                    "savings_usd": result["savings"]["cost_usd"],
                    "savings_pct": result["savings"]["cost_pct"],
                }
            st.rerun()

        except Exception as e:
            st.error(f"Backend API error: {e}")

result = st.session_state.get("cost_result")
if result:
    st.markdown("---")

    st.markdown(
        f"<div class='savings-card'>"
        f"<div class='savings-value'>${result['savings_usd']:.6f}</div>"
        f"<div class='savings-label'>Total Savings vs {result['baseline_name']}</div>"
        f"<div style='font-size: 1.1rem; color: #2ecc71; margin-top: 8px;'>({result['savings_pct']:.1f}% less)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #2ecc71;'>${result['routellm_cost']:.6f}</div>"
                f"<div class='metric-label'>RoutellM Cost</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #e74c3c;'>${result['baseline_cost']:.6f}</div>"
                f"<div class='metric-label'>{result['baseline_name']} Cost</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-value'>{result['routellm_tokens']:,}</div>"
                f"<div class='metric-label'>RoutellM Tokens</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><div class='metric-value'>{result['baseline_tokens']:,}</div>"
                f"<div class='metric-label'>{result['baseline_name']} Tokens</div></div>", unsafe_allow_html=True)

    st.markdown("### Cost Comparison")
    st.altair_chart(
        cost_comparison_chart(result["routellm_cost"], result["baseline_cost"], result["baseline_name"]),
        use_container_width=True,
    )

    st.markdown("### Token Comparison")
    tok_df = pd.DataFrame([
        {"model": "RoutellM", "tokens": result["routellm_tokens"]},
        {"model": result["baseline_name"], "tokens": result["baseline_tokens"]},
    ])
    token_chart = alt.Chart(tok_df).mark_bar().encode(
        x=alt.X("model:N", title=None),
        y=alt.Y("tokens:Q", title="Tokens"),
        color=alt.Color("model:N", scale=alt.Scale(
            domain=["RoutellM", result["baseline_name"]],
            range=["#2ecc71", "#e74c3c"],
        ), legend=None),
        tooltip=["model", "tokens"],
    ).properties(height=300, title="Token Comparison")
    st.altair_chart(token_chart, use_container_width=True)

    st.markdown("### Routing Breakdown")
    st.markdown(f"- **Deterministic (zero-cost):** {result['det_count']} tasks")
    st.markdown(f"- **Fireworks API (paid):** {result['api_count']} tasks")
    st.markdown(f"- **Dry-run (simulated):** {result['dry_count']} tasks")
    st.markdown(f"- **Accuracy:** {result['accuracy']:.1f}%")
