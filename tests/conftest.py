import json
from pathlib import Path

import pytest

from app.config import Config


@pytest.fixture
def config():
    return Config.from_env()


@pytest.fixture
def sample_tasks():
    path = Path(__file__).parent / "fixtures" / "tasks_track1_sample.json"
    with open(path) as f:
        return json.load(f)
