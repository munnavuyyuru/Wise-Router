import altair as alt
import pandas as pd


COLORS = {
    "deterministic": "#2ecc71",
    "fireworks_api": "#3498db",
    "dry_run": "#95a5a6",
}

TASK_COLORS = {
    "math": "#e74c3c",
    "sentiment": "#f39c12",
    "code_debug": "#9b59b6",
    "code_gen": "#2980b9",
    "summarization": "#1abc9c",
    "ner": "#e67e22",
    "logic": "#2c3e50",
}


def source_pie(source_counts: dict[str, int]) -> alt.Chart:
    df = pd.DataFrame([
        {"source": k.replace("_", " ").title(), "count": v}
        for k, v in source_counts.items() if v > 0
    ])
    return alt.Chart(df).mark_arc(innerRadius=40).encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(
            field="source", type="nominal",
            scale=alt.Scale(
                domain=[k.replace("_", " ").title() for k in COLORS],
                range=list(COLORS.values()),
            ),
            legend=alt.Legend(title="Routing Source"),
        ),
        tooltip=["source", "count"],
    ).properties(height=280, title="Routing Distribution")


def task_accuracy_bar(per_task_type: dict[str, dict]) -> alt.Chart:
    rows = []
    for t, info in per_task_type.items():
        acc = info.get("accuracy")
        if acc is not None:
            rows.append({"task_type": t, "accuracy": acc, "correct": info["correct"], "total": info["total"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("task_type:N", title="Task Type", sort=list(TASK_COLORS.keys())),
        y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "task_type:N",
            scale=alt.Scale(domain=list(TASK_COLORS.keys()), range=list(TASK_COLORS.values())),
            legend=None,
        ),
        tooltip=["task_type", "accuracy", "correct", "total"],
    ).properties(height=280, title="Accuracy by Task Type")


def cost_comparison_chart(routellm_cost: float, baseline_cost: float, baseline_label: str) -> alt.Chart:
    df = pd.DataFrame([
        {"model": "RoutellM", "cost": routellm_cost},
        {"model": baseline_label, "cost": baseline_cost},
    ])
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("model:N", title=None),
        y=alt.Y("cost:Q", title="Cost (USD)"),
        color=alt.Color(
            "model:N",
            scale=alt.Scale(
                domain=["RoutellM", baseline_label],
                range=["#2ecc71", "#e74c3c"],
            ),
            legend=None,
        ),
        tooltip=["model", alt.Tooltip("cost:Q", format=".6f")],
    ).properties(height=300, title="Cost Comparison")


def routing_flow_chart(task_type: str, confidence: float, threshold: float,
                       passed: bool, source: str, answer: str) -> alt.Chart:
    steps = [
        {"step": "1. Classify", "value": confidence, "max": 1.0, "status": "done"},
        {"step": "2. Threshold", "value": threshold, "max": 1.0, "status": "pass" if passed else "fail"},
        {"step": "3. Route", "value": 1, "max": 1, "status": source},
        {"step": "4. Answer", "value": 1, "max": 1, "status": "done" if answer else "fail"},
    ]
    df = pd.DataFrame(steps)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("step:N", title=None, sort=None),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("status:N", legend=None),
    ).properties(height=200)
