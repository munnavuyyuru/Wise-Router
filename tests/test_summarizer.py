from app.deterministic import summarizer


class TestSummarizer:
    def test_summarize(self):
        ans, conf = summarizer.solve("Summarize: \"Python is a great language. It is easy to learn. Many people use it.\"")
        assert ans and len(ans) > 10
        assert conf and conf >= 0.6

    def test_no_summarize_keyword(self):
        assert summarizer.solve("What is Python?") == (None, None)
