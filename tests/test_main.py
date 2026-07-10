import json
import os
import tempfile

from app.main import main


class TestMain:
    def test_main_with_input_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "tasks.json")
            output_path = os.path.join(tmpdir, "results.json")

            tasks = [
                {"task_id": "test_001", "prompt": "What is 2+2?"},
                {"task_id": "test_002", "prompt": "Say hello"},
            ]
            with open(input_path, "w") as f:
                json.dump(tasks, f)

            import sys
            sys.argv = ["main", "--input", input_path, "--output", output_path]
            main()

            assert os.path.exists(output_path)
            with open(output_path) as f:
                results = json.load(f)
            assert len(results) == 2
            assert results[0]["task_id"] == "test_001"
            assert results[1]["task_id"] == "test_002"
            assert "answer" in results[0]

    def test_main_handles_empty_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "tasks.json")
            output_path = os.path.join(tmpdir, "results.json")

            with open(input_path, "w") as f:
                json.dump([], f)

            import sys
            sys.argv = ["main", "--input", input_path, "--output", output_path]
            main()

            assert os.path.exists(output_path)
            with open(output_path) as f:
                results = json.load(f)
            assert results == []

    def test_main_handles_task_with_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "tasks.json")
            output_path = os.path.join(tmpdir, "results.json")

            tasks = [
                {"task_id": "test_001", "prompt": "What is 2+2?"},
                {"task_id": "test_002"},
                {"prompt": "no id here"},
            ]
            with open(input_path, "w") as f:
                json.dump(tasks, f)

            import sys
            sys.argv = ["main", "--input", input_path, "--output", output_path]
            main()

            with open(output_path) as f:
                results = json.load(f)
            assert len(results) == 1
            assert results[0]["task_id"] == "test_001"

    def test_main_exits_on_missing_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "nonexistent.json")
            output_path = os.path.join(tmpdir, "results.json")

            import sys
            sys.argv = ["main", "--input", input_path, "--output", output_path]
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit:
                pass

            assert not os.path.exists(output_path)

    def test_main_handles_non_list_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "tasks.json")
            output_path = os.path.join(tmpdir, "results.json")

            with open(input_path, "w") as f:
                json.dump({"task_id": "test_001", "prompt": "hello"}, f)

            import sys
            sys.argv = ["main", "--input", input_path, "--output", output_path]
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit:
                pass
