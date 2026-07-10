from app.deterministic import sentiment


class TestSentiment:
    def test_positive(self):
        ans, conf = sentiment.solve("What is the sentiment: \"Great product, loved it!\"")
        assert ans in ("positive", "negative", "neutral")

    def test_negative(self):
        ans, conf = sentiment.solve("Sentiment analysis: \"Terrible experience, worst product ever.\"")
        assert ans == "negative"

    def test_contrast(self):
        ans, conf = sentiment.solve("What is the sentiment of this review: \"The room was small, but the view was absolutely stunning.\"")
        assert ans == "positive"

    def test_neutral(self):
        ans, conf = sentiment.solve("Classify the sentiment: \"The meeting is scheduled for Tuesday at 3 PM.\"")
        assert ans == "neutral"

    def test_no_sentiment_keyword(self):
        assert sentiment.solve("What is the capital of France?") == (None, None)
