import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="RoutellM — Model Router / Cost Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.charts import source_pie, task_accuracy_bar, COLORS
from utils import api_client


st.markdown("""
<style>
    .metric-card { background: #1a1a2e; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #2d2d5e; }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .pass-badge { background: #2ecc71; color: #fff; padding: 2px 14px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .fail-badge { background: #e74c3c; color: #fff; padding: 2px 14px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-det { background: #2ecc71; color: #fff; padding: 1px 10px; border-radius: 12px; font-size: 0.75rem; }
    .badge-api { background: #3498db; color: #fff; padding: 1px 10px; border-radius: 12px; font-size: 0.75rem; }
    .badge-dry { background: #95a5a6; color: #fff; padding: 1px 10px; border-radius: 12px; font-size: 0.75rem; }
    .stApp { background: #0f0f23; }
    h1, h2, h3, .stMarkdown { color: #eee !important; }
    .stButton button { background: #2ecc71; color: #000; font-weight: 600; border: none; border-radius: 8px; }
    .stButton button:hover { background: #27ae60; color: #fff; }
</style>
""", unsafe_allow_html=True)

st.title(" RoutellM")
st.markdown("**Model Router / Cost Optimizer** — automatically routes each prompt to the cheapest, best-suited solver or LLM.")

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_tasks.json"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Quick Start")
    st.markdown("Upload a JSON file of tasks in **Evaluate** or try a single prompt in **Live Demo**.")

    if         st.button(" Run Sample Evaluation (24 tasks)", type="primary", use_container_width=True):
        with st.spinner("Sending 25 tasks to backend API..."):
            with open(SAMPLE_PATH, "r") as f:
                tasks = json.load(f)
            try:
                result = api_client.evaluate(tasks, dry_run=False)
                st.session_state["last_eval"] = result
                st.session_state["has_eval"] = True
            except Exception as e:
                st.error(f"Backend API error: {e}")
        st.rerun()

with col2:
    if "has_eval" in st.session_state and st.session_state["has_eval"]:
        r = st.session_state["last_eval"]
        s = r["summary"]
        passed = s["accuracy_pct"] >= 85.0
        st.markdown(f"**Last Run:** {s['total_tasks']} tasks &nbsp;|&nbsp; "
                    f"<span class='{'pass-badge' if passed else 'fail-badge'}'>"
                    f"{'PASS' if passed else 'FAIL'} {s['accuracy_pct']:.1f}%</span>",
                    unsafe_allow_html=True)

st.markdown("---")

if "has_eval" in st.session_state and st.session_state["has_eval"]:
    r = st.session_state["last_eval"]
    s = r["summary"]
    c = r["cost"]
    src = s["source_breakdown"]

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><div class='metric-value'>{s['accuracy_pct']:.1f}%</div>"
                f"<div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-value'>${c['total_cost_usd']:.6f}</div>"
                f"<div class='metric-label'>Total Cost (USD)</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-value'>{c['total_tokens']:,}</div>"
                f"<div class='metric-label'>Tokens Used</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><div class='metric-value'>{s['total_tasks']}</div>"
                f"<div class='metric-label'>Tasks Routed</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    pie_col, bar_col = st.columns(2)

    with pie_col:
        pie = source_pie(src)
        st.altair_chart(pie, use_container_width=True)

    with bar_col:
        bar = task_accuracy_bar(r["per_task_type"])
        if bar:
            st.altair_chart(bar, use_container_width=True)

    with st.expander(" Per-Task Breakdown"):
        for task in r["results"]:
            src_label = task["source"]
            badge_class = {"deterministic": "badge-det", "fireworks_api": "badge-api", "dry_run": "badge-dry"}
            correct_icon = "✅" if task["correct"] else "❌" if task["correct"] is False else "—"
            st.markdown(
                f"**{task['task_id']}** `{task['task_type']}` "
                f"<span class='{badge_class.get(src_label, 'badge-dry')}'>{src_label}</span> "
                f"{correct_icon}",
                unsafe_allow_html=True,
            )
            st.caption(f"Q: {task['prompt'][:120]}...")
            st.caption(f"A: {task['answer'][:200]}")
            st.markdown("---", unsafe_allow_html=True)

elif "last_eval" in st.session_state:
    pass

else:
    try:
        runs = api_client.list_runs()
        if runs:
            latest = api_client.get_run(runs[-1]["run_id"])
            s = latest["summary"]
            c = latest["cost"]
            src = s["source_breakdown"]

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='metric-card'><div class='metric-value'>{s['accuracy_pct']:.1f}%</div>"
                        f"<div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-card'><div class='metric-value'>${c['total_cost_usd']:.6f}</div>"
                        f"<div class='metric-label'>Total Cost (USD)</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-card'><div class='metric-value'>{c['total_tokens']:,}</div>"
                        f"<div class='metric-label'>Tokens Used</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-card'><div class='metric-value'>{s['total_tasks']}</div>"
                        f"<div class='metric-label'>Tasks Routed</div></div>", unsafe_allow_html=True)

            st.markdown("---")
            pie_col, bar_col = st.columns(2)
            with pie_col:
                st.altair_chart(source_pie(src), use_container_width=True)
            with bar_col:
                bar = task_accuracy_bar(latest["per_task_type"])
                if bar:
                    st.altair_chart(bar, use_container_width=True)
        else:
            st.info(" No evaluation data yet. Click **Run Sample Evaluation** above or upload tasks in the **Evaluate** page.")
    except Exception:
        st.warning(" Backend API not reachable. Make sure the backend is running on port 8000.")
