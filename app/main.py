import argparse
import json
import os
import sys

from app.classifier.task_classifier import TaskClassifier
from app.config import Config
from app.llm.batcher import BatchProcessor
from app.llm.fireworks_client import FireworksClient


TIME_LIMIT_SECONDS = 600
TIME_BUDGET_MARGIN = 20


def main() -> None:
    parser = argparse.ArgumentParser(prog="routellm", description="RoutellM — Token-Efficient Routing Agent")
    parser.add_argument("--input", dest="input_path", help="Path to input tasks.json")
    parser.add_argument("--output", dest="output_path", help="Path to output results.json")
    args = parser.parse_args()

    config = Config.from_env()

    if args.input_path:
        config.input_path = args.input_path
    if args.output_path:
        config.output_path = args.output_path

    if not os.path.exists(config.input_path):
        print(f"Error: input file not found: {config.input_path}", file=sys.stderr)
        sys.exit(1)

    with open(config.input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        print("Error: input must be a JSON array of task objects", file=sys.stderr)
        sys.exit(1)

    classifier = TaskClassifier(config.classifier_model_path)
    fireworks = FireworksClient(config) if config.fireworks_api_key else None
    processor = BatchProcessor(config, classifier, fireworks)
    results = processor.process_all(tasks, time_limit=TIME_LIMIT_SECONDS, budget_margin=TIME_BUDGET_MARGIN)

    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} tasks → {config.output_path}")


if __name__ == "__main__":
    main()
