from app.llm import model_selector


class TestModelSelector:
    def test_choose_cheapest(self):
        allowed = [
            "accounts/fireworks/models/gemma-3-1b-it",
            "accounts/fireworks/models/minimax-m3",
        ]
        model = model_selector.choose("sentiment", allowed)
        assert model == "accounts/fireworks/models/gemma-3-1b-it"

    def test_choose_code_gen(self):
        allowed = [
            "accounts/fireworks/models/gemma-3-1b-it",
            "accounts/fireworks/models/kimi-k2p7-code",
        ]
        model = model_selector.choose("code_gen", allowed)
        assert model == "accounts/fireworks/models/kimi-k2p7-code"

    def test_allowed_filter(self):
        allowed = ["accounts/fireworks/models/gemma-3-1b-it"]
        model = model_selector.choose("logic", allowed)
        assert model is not None
        assert model in allowed

    def test_empty_allowed(self):
        model = model_selector.choose("math", [])
        assert model is None
