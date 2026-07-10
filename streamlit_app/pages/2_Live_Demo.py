import streamlit as st

st.set_page_config(page_title="Live Demo — RoutellM", page_icon="", layout="wide")

from utils import api_client
from utils.charts import COLORS


st.markdown("""
<style>
    .step-box { background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 10px 0; border-left: 4px solid #2d2d5e; }
    .step-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .step-value { font-size: 1.5rem; font-weight: 700; margin: 4px 0; }
    .step-detail { font-size: 0.9rem; color: #ccc; }
    .badge-det { display: inline-block; background: #2ecc71; color: #fff; padding: 2px 14px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-api { display: inline-block; background: #3498db; color: #fff; padding: 2px 14px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-dry { display: inline-block; background: #95a5a6; color: #fff; padding: 2px 14px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-pass { display: inline-block; background: #2ecc71; color: #fff; padding: 2px 14px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-fail { display: inline-block; background: #e74c3c; color: #fff; padding: 2px 14px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .answer-box { background: #0d0d1a; border-radius: 12px; padding: 20px; border: 1px solid #2d2d5e; font-size: 1rem; line-height: 1.6; margin: 10px 0; }
    .flow-arrow { font-size: 2rem; color: #555; text-align: center; padding: 4px 0; }
    h1, h2, h3 { color: #eee !important; }
    .stApp { background: #0f0f23; }
    .stButton button { background: #2ecc71; color: #000; font-weight: 600; border: none; border-radius: 8px; }
    .stButton button:hover { background: #27ae60; color: #fff; }
    .big-input textarea { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

st.title(" Live Routing Demo")
st.markdown("Type a prompt below and watch RoutellM classify, threshold-check, route, and answer in real time.")

if "demo_result" not in st.session_state:
    st.session_state["demo_result"] = None

EXAMPLE_PROMPTS = [
    "What is 18 + 24?",
    "Sentiment analysis: 'Great product, loved it!'",
    "Write a Python function to sort a list of numbers.",
    "Extract emails from: 'Contact john@example.com for support'",
    "Summarize: 'Python is a great language for data science.'",
    "Alice is taller than Bob. Bob is taller than Charlie. Who is tallest?",
    "Fix this code:\ndef add(a, b):\n    a + b",
    "What is the capital of France?",
]

prompt = st.text_area(
    "Enter your prompt:",
    value=st.session_state.get("demo_prompt", EXAMPLE_PROMPTS[0]),
    height=100,
    placeholder="Type a question, code request, or analysis task...",
    key="demo_prompt_input",
)

col_btn, col_samples = st.columns([1, 2])
with col_btn:
    run_clicked = st.button(" Route Prompt", type="primary", use_container_width=True)
with col_samples:
    st.markdown("**Quick samples:**")
    sample_cols = st.columns(4)
    for i, ex in enumerate(EXAMPLE_PROMPTS):
        with sample_cols[i % 4]:
            if st.button(ex[:20] + "...", key=f"sample_{i}", help=ex):
                st.session_state["demo_prompt"] = ex
                st.rerun()

if run_clicked:
    prompt = st.session_state.get("demo_prompt_input", "")
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner(" Routing via backend API..."):
            try:
                result = api_client.route_single(prompt, dry_run=True)
                st.session_state["demo_result"] = result
            except Exception as e:
                st.error(f"Backend API error: {e}")
        st.rerun()

result = st.session_state.get("demo_result")
if result:
    st.markdown("---")

    flow_cols = st.columns(4)
    steps_data = [
        ("Classify", f"{result['task_type']}", f"Confidence: {result['confidence']:.1%}"),
        ("Threshold", f"{result['confidence']:.1%} vs {result['threshold']:.0%}",
         f"{' PASS' if result['passed_threshold'] else ' FALLBACK'}"
         if result.get('solver_name') else "No solver for this task type"),
        ("Route",
         f" {result['source']}"
         if result['source'] == "deterministic" else
         f" {result['source']}" if result['source'] == "fireworks_api" else
         f" {result['source']}",
         result.get('model_used', '') if result['source'] == "fireworks_api" else
         (result.get('solver_name') or "—") if result['source'] == "deterministic" else
         "No API key (simulated)"),
        ("Answer", "Ready", f"Source: {result['source']}"),
    ]

    source_badges = {
        "deterministic": ("badge-det", COLORS["deterministic"]),
        "fireworks_api": ("badge-api", COLORS["fireworks_api"]),
        "dry_run": ("badge-dry", COLORS["dry_run"]),
    }

    for i, (label, value, detail) in enumerate(steps_data):
        with flow_cols[i]:
            if i == 0:
                status_color = "#2ecc71"
                status_text = "done"
            elif i == 1:
                if result['passed_threshold']:
                    status_color = "#2ecc71"
                    status_text = "pass"
                else:
                    status_color = "#e74c3c"
                    status_text = "fail"
            elif i == 2:
                bc, _ = source_badges.get(result['source'], ("badge-dry", "#95a5a6"))
                status_color = COLORS.get(result['source'], "#95a5a6")
                status_text = result['source']
            else:
                status_color = "#2ecc71" if result['answer'] else "#e74c3c"
                status_text = "ready" if result['answer'] else "error"

            st.markdown(f"""
            <div class='step-box' style='border-left-color: {status_color};'>
                <div class='step-label'>Step {i+1}</div>
                <div class='step-label'>{label}</div>
                <div class='step-value'>{value}</div>
                <div class='step-detail'>{detail}</div>
                <div style='margin-top: 8px;'>
                    <span style='background: {status_color}; color: #fff; padding: 2px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;'>{status_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='flow-arrow'>⬇</div>", unsafe_allow_html=True)

    st.markdown("###  Answer")
    st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)

    cost_col, token_col, source_col = st.columns(3)
    cost_col.metric("Cost (USD)", f"${result['cost']:.6f}" if result['cost'] > 0 else "$0.00")
    token_col.metric("Tokens Used", str(result['tokens']) if result['tokens'] > 0 else "0")
    bc_name, _ = source_badges.get(result['source'], ("badge-dry", "#95a5a6"))
    source_col.markdown(f"**Source**<br><span class='{bc_name}'>{result['source']}</span>", unsafe_allow_html=True)
