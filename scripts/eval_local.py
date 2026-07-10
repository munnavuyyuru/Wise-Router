import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.classifier.task_classifier import TaskClassifier
from app.config import Config
from app.llm.fireworks_client import FireworksClient, TokenTracker
from app.router import route_prompt


TASK_TYPES = ["math", "sentiment", "code_debug", "code_gen", "summarization", "ner", "logic", "factual"]

DET = "deterministic"
DRY = "dry_run"
API = "fireworks_api"


def _matches(answer: str, expected: str) -> bool:
    if not expected:
        return bool(answer.strip()) if answer.strip() else True

    a, e = answer.strip().lower(), expected.strip().lower()

    if a == e:
        return True

    try:
        return abs(float(a) - float(e)) < 0.01
    except (ValueError, TypeError):
        pass

    if len(e) > 3 and e in a:
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="RoutellM Evaluation Judge — routes prompts, tracks tokens/cost, computes accuracy")
    parser.add_argument("--input", default="tests/fixtures/tasks_track1_sample.json", help="Labeled JSON with task_id, prompt, expected_answer")
    parser.add_argument("--report", default="eval_report.json", help="Path to write detailed evaluation report JSON")
    parser.add_argument("--threshold", type=float, default=85.0, help="Minimum accuracy percentage to pass (default: 85.0)")
    parser.add_argument("--train-data", default="data/training_data.json", help="Training data to fit classifier")
    parser.add_argument("--model-path", default=None, help="Path to pre-trained classifier model pickle")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run mode (no API calls even if key is set)")
    args = parser.parse_args()

    config = Config.from_env()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        print("Error: input must be a JSON array of task objects", file=sys.stderr)
        sys.exit(1)

    model_path = args.model_path or config.classifier_model_path
    classifier = TaskClassifier(model_path)

    if not classifier.is_fitted:
        train_path = args.train_data
        if os.path.exists(train_path):
            print(f"Training classifier on {train_path} ...")
            with open(train_path, "r", encoding="utf-8") as f:
                training = json.load(f)
            prompts = [t["prompt"] for t in training if t.get("prompt") and t.get("task_type")]
            labels = [t["task_type"] for t in training if t.get("prompt") and t.get("task_type")]
            if len(set(labels)) >= 2:
                classifier.fit(prompts, labels)
                print(f"  Trained on {len(prompts)} examples across {len(set(labels))} task types")
        else:
            print("Warning: no trained classifier and no training data found", file=sys.stderr)

    has_key = bool(config.fireworks_api_key) and not args.dry_run
    tracker = TokenTracker()
    tracker.reset()
    fireworks = FireworksClient(config, token_tracker=tracker) if has_key else None

    results = []
    correct = 0
    total = 0
    by_type: dict[str, dict] = {t: {"correct": 0, "total": 0, "by_source": {DET: 0, API: 0, DRY: 0}} for t in TASK_TYPES}
    source_counts = {DET: 0, API: 0, DRY: 0}

    for task in tasks:
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")
        expected = task.get("expected_answer", "")

        if not task_id or not prompt:
            print(f"  Skipping task with missing fields: {task}", file=sys.stderr)
            continue

        api_calls_before = len(tracker._records)
        task_type = classifier.predict_type(prompt) if classifier.is_fitted else "unknown"

        try:
            answer = route_prompt(prompt, config, classifier, fireworks)
        except Exception as exc:
            print(f"  ERROR [{task_id}]: {exc}", file=sys.stderr)
            answer = ""

        api_calls_after = len(tracker._records)
        if api_calls_after > api_calls_before:
            source = API
        elif answer.startswith("[dry-run]"):
            source = DRY
        else:
            source = DET

        source_counts[source] += 1
        if task_type in by_type:
            by_type[task_type]["by_source"][source] += 1

        matched = None
        if expected or answer:
            total += 1
            matched = _matches(answer, expected)
            if matched:
                correct += 1
                if task_type in by_type:
                    by_type[task_type]["correct"] += 1
            if task_type in by_type:
                by_type[task_type]["total"] += 1

        results.append({
            "task_id": task_id,
            "task_type": task_type,
            "prompt": prompt,
            "expected": expected,
            "answer": answer,
            "source": source,
            "correct": matched,
        })

    total_accuracy = (correct / total * 100) if total else 0.0

    type_breakdown = {}
    for t in TASK_TYPES:
        info = by_type[t]
        info["accuracy"] = round((info["correct"] / info["total"] * 100), 1) if info["total"] else None
        type_breakdown[t] = info

    cumulative = tracker.cumulative
    total_cost = cumulative.cost
    total_prompt_tokens = cumulative.prompt_tokens
    total_completion_tokens = cumulative.completion_tokens
    total_tokens_all = cumulative.total_tokens

    report = {
        "summary": {
            "total_tasks": len(results),
            "processed": len([r for r in results if r["answer"]]),
            "correct": correct,
            "incorrect": total - correct,
            "accuracy_pct": round(total_accuracy, 2),
            "threshold_pct": args.threshold,
            "passed": total_accuracy >= args.threshold,
            "source_breakdown": source_counts,
        },
        "cost": {
            "total_cost_usd": round(total_cost, 6),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens_all,
            "api_calls": len(tracker._records),
        },
        "per_task_type": type_breakdown,
        "results": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  RoutellM Evaluation Judge Report")
    print(f"{'='*60}")
    print(f"  Tasks:        {len(results)} total, {total} scored")
    print(f"  Correct:      {correct}/{total} ({total_accuracy:.1f}%)")
    passed = total_accuracy >= args.threshold
    print(f"  Threshold:    {args.threshold:.0f}%  {'PASS' if passed else 'FAIL'}")
    print(f"  Cost (USD):   ${total_cost:.6f}")
    print(f"  Tokens:       {total_tokens_all} ({total_prompt_tokens} prompt + {total_completion_tokens} completion)")
    print(f"  API calls:    {len(tracker._records)}")
    print(f"  Source:       {source_counts[DET]} deterministic, {source_counts[DRY]} dry-run, {source_counts[API]} API")
    print(f"{'='*60}")
    print(f"  Per task type:")
    for t in TASK_TYPES:
        info = by_type[t]
        acc_str = f"{info['accuracy']:.1f}%" if info["total"] else "N/A"
        print(f"    {t:15s}  {info['correct']}/{info['total']} ({acc_str})  det={info['by_source'][DET]} api={info['by_source'][API]} dry={info['by_source'][DRY]}")
    print(f"{'='*60}")
    print(f"  Full report: {report_path.resolve()}")
    print(f"{'='*60}")

    if total_accuracy < args.threshold:
        print(f"\n  FAILED: accuracy {total_accuracy:.1f}% is below {args.threshold:.0f}% threshold")
        sys.exit(1)

    print(f"\n  PASSED: accuracy {total_accuracy:.1f}% meets {args.threshold:.0f}% threshold")


if __name__ == "__main__":
    main()
