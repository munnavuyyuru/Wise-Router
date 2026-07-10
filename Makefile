.PHONY: install test eval-local run-sample docker-build docker-run clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

eval-local:
	python scripts/eval_local.py

run-sample:
	python -m app.main --input tests/fixtures/tasks_track1_sample.json --output /tmp/results.json

docker-build:
	docker build -t routellm .

docker-run:
	docker run --rm \
		-e FIREWORKS_API_KEY=$(FIREWORKS_API_KEY) \
		-e FIREWORKS_BASE_URL=$(FIREWORKS_BASE_URL) \
		-e ALLOWED_MODELS=$(ALLOWED_MODELS) \
		-v $(PWD)/input:/input \
		-v $(PWD)/output:/output \
		routellm

clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
