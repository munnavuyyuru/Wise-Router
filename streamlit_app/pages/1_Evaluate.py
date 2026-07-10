import json
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Batch Evaluate — RoutellM", page_icon="", layout="wide")

from utils.charts import source_pie, task_accuracy_bar, COLORS
from utils import api_client


st.markdown("""
<style>
    .metric-card { background: #1a1a2e; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #2d2d5e; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .badge-det { display: inline-block; background: #2ecc71; color: #fff; padding: 1px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-api { display: inline-block; background: #3498db; color: #fff; padding: 1px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-dry { display: inline-block; background: #95a5a6; color: #fff; padding: 1px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .correct { color: #2ecc71; font-weight: 700; }
    .incorrect { color: #e74c3c; font-weight: 700; }
    .stApp { background: #0f0f23; }
    h1, h2, h3 { color: #eee !important; }
    .stButton button { background: #2ecc71; color: #000; font-weight: 600; border: none; border-radius: 8px; }
    .stButton button:hover { background: #27ae60; color: #fff; }
    .stDataFrame { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

st.title(" Batch Evaluation")
st.markdown("Upload a JSON file of tasks and let RoutellM classify, route, and score them.")

if "eval_result" not in st.session_state:
    st.session_state["eval_result"] = None

SAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sample_tasks.json"

uploaded = st.file_uploader(
    "Upload tasks JSON (array of `{task_id, prompt, expected_answer?}`)",
    type=["json"],
    key="eval_upload",
    help="Format: [{\"task_id\": \"q1\", \"prompt\": \"What is 2+2?\", \"expected_answer\": \"42\"}, ...]",
)

col1, col2, col3 = st.columns(3)
with col1:
    threshold = st.slider("Accuracy Threshold (%)", 50, 100, 85, 5)
with col2:
    dry_run = st.toggle("Dry Run (no API calls)", value=True, help="Simulate routing without sending to Fireworks API")
with col3:
    use_sample = st.checkbox("Use sample 25-task fixture instead", value=False)

if st.button(" Run Evaluation", type="primary", use_container_width=True):
    if not uploaded and not use_sample:
        st.warning("Please upload a JSON file or check 'Use sample fixture'.")
    else:
        try:
            if use_sample:
                with open(SAMPLE_PATH, "r") as f:
                    tasks = json.load(f)
                st.info(f"Loaded {len(tasks)} tasks from sample fixture.")
            else:
                tasks = json.load(uploaded)
                if not isinstance(tasks, list):
                    st.error("JSON must be an array of task objects.")
                    st.stop()

            with st.spinner(f"Sending {len(tasks)} tasks to backend API..."):
                result = api_client.evaluate(tasks, dry_run=dry_run, threshold=threshold)
                st.session_state["eval_result"] = result
            st.rerun()

        except Exception as e:
            st.error(f"Backend API error: {e}")

result = st.session_state.get("eval_result")
if result:
    s = result["summary"]
    c = result["cost"]
    src = s["source_breakdown"]
    passed = s["accuracy_pct"] >= threshold

    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><div class='metric-value'>{s['accuracy_pct']:.1f}%</div>"
                f"<div class='metric-label'>Accuracy {'✅' if passed else '❌'}</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-value'>${c['total_cost_usd']:.6f}</div>"
                f"<div class='metric-label'>Total Cost</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-value'>{c['total_tokens']:,}</div>"
                f"<div class='metric-label'>Tokens ({c['api_calls']} API calls)</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><div class='metric-value'>{s['total_tasks']}</div>"
                f"<div class='metric-label'>Tasks • {s['correct']}/{s['correct']+s['incorrect']} correct</div></div>",
                unsafe_allow_html=True)

    pie_col, bar_col = st.columns(2)
    with pie_col:
        st.altair_chart(source_pie(src), use_container_width=True)
    with bar_col:
        bar = task_accuracy_bar(result["per_task_type"])
        if bar:
            st.altair_chart(bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Results Table")

    rows = []
    for r in result["results"]:
        rows.append({
            "Task ID": r["task_id"],
            "Type": r["task_type"],
            "Source": r["source"],
            "Correct": "✅" if r["correct"] else ("❌" if r["correct"] is False else "—"),
            "Expected": r["expected"][:80] if r["expected"] else "",
            "Answer": r["answer"][:120] if r["answer"] else "",
        })

    df = pd.DataFrame(rows)

    def highlight_source(val):
        if val == "deterministic":
            return "background: #2ecc7133; color: #2ecc71; font-weight: 600"
        if val == "fireworks_api":
            return "background: #3498db33; color: #3498db; font-weight: 600"
        if val == "dry_run":
            return "background: #95a5a633; color: #95a5a6; font-weight: 600"
        return ""

    styled = df.style.applymap(highlight_source, subset=["Source"])
    st.dataframe(styled, use_container_width=True, height=400)

    report_json = json.dumps(result, indent=2)
    st.download_button(
        label=" Download Full Report (JSON)",
        data=report_json,
        file_name="routellm_eval_report.json",
        mime="application/json",
        use_container_width=True,
    )

    st.session_state["last_eval"] = result
    st.session_state["has_eval"] = True
