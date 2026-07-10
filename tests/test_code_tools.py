from app.deterministic import code_tools


class TestCodeTools:
    def test_mutable_default(self):
        ans, conf = code_tools.solve(
            "def add_item(item, items=[]):\n    items.append(item)\n    return items\n"
        )
        assert ans and "mutable" in ans.lower()
        assert conf and conf >= 0.8

    def test_missing_return(self):
        ans, conf = code_tools.solve(
            "Fix this:\ndef add(a, b):\n    a + b\n"
        )
        assert ans and "return" in ans.lower()
        assert conf and conf >= 0.8

    def test_sort_template(self):
        ans, conf = code_tools.solve("Write a function to sort a list of numbers.")
        assert ans and "def sort_array" in ans
        assert conf and conf >= 0.8

    def test_fibonacci_template(self):
        ans, conf = code_tools.solve("Implement fibonacci in Python.")
        assert ans and "fibonacci" in ans.lower()
        assert conf and conf >= 0.8

    def test_no_match(self):
        assert code_tools.solve("What is the weather?") == (None, None)
