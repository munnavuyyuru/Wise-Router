import json
import tempfile
from pathlib import Path

import pytest

from app.config import Config
from app.llm.fireworks_client import (
    FireworksClient,
    TokenTracker,
    TokenUsage,
    UsageRecord,
)


class TestTokenUsage:
    def test_defaults(self):
        tu = TokenUsage()
        assert tu.prompt_tokens == 0
        assert tu.completion_tokens == 0
        assert tu.total_tokens == 0

    def test_add(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        a.add(b)
        assert a.prompt_tokens == 30
        assert a.completion_tokens == 15
        assert a.total_tokens == 45

    def test_bool_false_when_zero(self):
        assert not TokenUsage()

    def test_bool_true_when_positive(self):
        assert TokenUsage(total_tokens=1)


class TestTokenTracker:
    def test_tracks_cumulative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            tracker = TokenTracker(path)
            tu = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.0001)
            tracker.record(tu, task_type="math", model="gemma")
            assert tracker.cumulative.total_tokens == 15
            assert tracker.cumulative.prompt_tokens == 10
            assert tracker.cumulative.completion_tokens == 5
            assert tracker.cumulative.cost == 0.0001

    def test_multiple_records_accumulate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            tracker = TokenTracker(path)
            tracker.record(TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.0001))
            tracker.record(TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30, cost=0.0002))
            assert tracker.cumulative.total_tokens == 45
            assert tracker.cumulative.cost == pytest.approx(0.0003)
            assert len(tracker._records) == 2

    def test_persists_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            tracker = TokenTracker(path)
            tracker.record(TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15), task_type="sentiment", model="gemma-3-1b-it")
            tracker.record(TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28), task_type="math", model="minimax-m3")

            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["cumulative"]["total_tokens"] == 43
            assert data["total_calls"] == 2
            assert len(data["records"]) == 2
            assert data["records"][0]["task_type"] == "sentiment"

    def test_loads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            initial = {
                "cumulative": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
                "total_calls": 2,
                "records": [
                    {"timestamp": "2025-01-01T00:00:00", "task_type": "math", "model": "a", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    {"timestamp": "2025-01-01T00:00:00", "task_type": "code_gen", "model": "b", "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                ],
            }
            path.write_text(json.dumps(initial), encoding="utf-8")

            tracker = TokenTracker(path)
            assert tracker.cumulative.total_tokens == 45
            assert len(tracker._records) == 2

            tracker.record(TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8))
            assert tracker.cumulative.total_tokens == 53
            assert len(tracker._records) == 3

    def test_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            tracker = TokenTracker(path)
            tracker.record(TokenUsage(total_tokens=10))
            assert path.exists()
            tracker.reset()
            assert tracker.cumulative.total_tokens == 0
            assert len(tracker._records) == 0
            assert not path.exists()

    def test_reset_clears_in_memory_only_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            tracker = TokenTracker(path)
            tracker.record(TokenUsage(total_tokens=10))
            tracker.reset()
            assert tracker.cumulative.total_tokens == 0

    def test_handles_corrupt_file_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            path.write_text("not valid json", encoding="utf-8")
            tracker = TokenTracker(path)
            assert tracker.cumulative.total_tokens == 0
            assert len(tracker._records) == 0

    def test_record_creates_usage_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            tracker = TokenTracker(path)
            tracker.record(TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15), task_type="ner", model="gemma-3-4b-it")
            rec = tracker._records[0]
            assert rec.task_type == "ner"
            assert rec.model == "gemma-3-4b-it"
            assert rec.prompt_tokens == 10
            assert rec.completion_tokens == 5
            assert rec.total_tokens == 15
            assert rec.timestamp != ""


class TestFireworksClient:
    def test_init(self):
        cfg = Config()
        client = FireworksClient(cfg)
        assert client.allowed_models is not None

    def test_init_with_tracker(self):
        cfg = Config()
        tracker = TokenTracker()
        client = FireworksClient(cfg, token_tracker=tracker)
        assert client.token_tracker is tracker

    def test_rejects_unallowed_model(self):
        cfg = Config()
        client = FireworksClient(cfg)
        with pytest.raises(ValueError, match="not in ALLOWED_MODELS"):
            client.generate("Hello", model="accounts/fireworks/models/nonexistent")

    def test_count_tokens(self):
        cfg = Config()
        client = FireworksClient(cfg)
        count = client.count_tokens("Hello world")
        assert count >= 1
        assert count == client.count_tokens("Hello world")

    def test_token_tracker_default_creates_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            cfg = Config()
            tracker = TokenTracker(path)
            client = FireworksClient(cfg, token_tracker=tracker)
            assert client.token_tracker.cumulative.total_tokens == 0
