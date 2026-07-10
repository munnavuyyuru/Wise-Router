from app.deterministic import ner


class TestNER:
    def test_extract_emails(self):
        ans, conf = ner.solve("Extract emails: contact@example.com")
        assert ans and "emails" in ans
        assert conf and conf >= 0.8

    def test_extract_money(self):
        ans, conf = ner.solve("Extract money amounts: $500 and $1,200.50")
        assert ans and "money" in ans
        assert conf and conf >= 0.8

    def test_no_extract_keyword(self):
        assert ner.solve("Hello world") == (None, None)
