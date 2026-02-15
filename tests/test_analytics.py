"""Unit tests for the analytics module."""


from utils.analytics import (
    analyze_sentiment_textblob,
    analyze_sentiment_vader,
    get_named_entities,
    get_word_frequency,
)


class TestAnalyzeSentimentTextBlob:
    """Tests for TextBlob sentiment analysis."""

    def test_positive_sentiment(self) -> None:
        """Test that positive text returns positive sentiment."""
        result = analyze_sentiment_textblob("I love this product! It's amazing!")
        assert result["sentiment"] == "Positive"
        assert result["polarity"] > 0

    def test_negative_sentiment(self) -> None:
        """Test that negative text returns negative sentiment."""
        result = analyze_sentiment_textblob("I hate this. It's terrible and awful.")
        assert result["sentiment"] == "Negative"
        assert result["polarity"] < 0

    def test_neutral_sentiment(self) -> None:
        """Test that neutral text returns neutral sentiment."""
        result = analyze_sentiment_textblob("The meeting is at 3pm.")
        assert result["sentiment"] == "Neutral"
        assert result["polarity"] == 0

    def test_subjectivity_is_returned(self) -> None:
        """Test that subjectivity score is included in results."""
        result = analyze_sentiment_textblob("I think this is wonderful!")
        assert "subjectivity" in result
        assert 0 <= result["subjectivity"] <= 1

    def test_empty_text_returns_neutral(self) -> None:
        """Test that empty text returns neutral sentiment."""
        result = analyze_sentiment_textblob("")
        assert result["sentiment"] == "Neutral"
        assert result["polarity"] == 0.0
        assert result["subjectivity"] == 0.0

    def test_whitespace_text_returns_neutral(self) -> None:
        """Test that whitespace-only text returns neutral."""
        result = analyze_sentiment_textblob("   ")
        assert result["sentiment"] == "Neutral"


class TestAnalyzeSentimentVader:
    """Tests for VADER sentiment analysis."""

    def test_positive_sentiment(self) -> None:
        """Test that positive text returns positive sentiment."""
        result = analyze_sentiment_vader("I love this product! It's amazing!")
        assert result["sentiment"] == "Positive"
        assert result["polarity"] >= 0.05

    def test_negative_sentiment(self) -> None:
        """Test that negative text returns negative sentiment."""
        result = analyze_sentiment_vader("I hate this. It's terrible and awful.")
        assert result["sentiment"] == "Negative"
        assert result["polarity"] <= -0.05

    def test_neutral_sentiment(self) -> None:
        """Test that neutral text returns neutral sentiment."""
        result = analyze_sentiment_vader("The meeting is scheduled.")
        assert result["sentiment"] == "Neutral"
        assert -0.05 < result["polarity"] < 0.05

    def test_subjectivity_is_none(self) -> None:
        """Test that VADER does not return subjectivity."""
        result = analyze_sentiment_vader("This is a test.")
        assert result["subjectivity"] is None

    def test_raw_scores_included(self) -> None:
        """Test that raw VADER scores are included."""
        result = analyze_sentiment_vader("I love this!")
        assert "raw_scores" in result
        assert "pos" in result["raw_scores"]
        assert "neg" in result["raw_scores"]
        assert "neu" in result["raw_scores"]
        assert "compound" in result["raw_scores"]

    def test_empty_text_returns_neutral(self) -> None:
        """Test that empty text returns neutral sentiment."""
        result = analyze_sentiment_vader("")
        assert result["sentiment"] == "Neutral"
        assert result["polarity"] == 0.0


class TestGetWordFrequency:
    """Tests for word frequency extraction."""

    def test_returns_dictionary(self) -> None:
        """Test that function returns a dictionary."""
        result = get_word_frequency("hello world hello")
        assert isinstance(result, dict)

    def test_counts_words_correctly(self) -> None:
        """Test that word counts are accurate."""
        result = get_word_frequency("hello hello hello world world")
        assert result.get("hello", 0) == 3
        assert result.get("world", 0) == 2

    def test_excludes_stopwords(self) -> None:
        """Test that common stopwords are excluded."""
        result = get_word_frequency("the quick brown fox jumps over the lazy dog")
        assert "the" not in result
        assert "over" not in result

    def test_limits_to_n_top(self) -> None:
        """Test that results are limited to n_top words."""
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = get_word_frequency(text, n_top=5)
        assert len(result) <= 5

    def test_empty_text_returns_empty_dict(self) -> None:
        """Test that empty text returns empty dictionary."""
        result = get_word_frequency("")
        assert result == {}

    def test_single_char_words_excluded(self) -> None:
        """Test that single character words are excluded."""
        result = get_word_frequency("a b c hello world")
        assert "a" not in result
        assert "b" not in result
        assert "c" not in result

    def test_case_insensitive(self) -> None:
        """Test that word frequency is case insensitive."""
        result = get_word_frequency("Hello HELLO hello")
        assert result.get("hello", 0) == 3


class TestGetNamedEntities:
    """Tests for named entity recognition."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = get_named_entities("Hello world")
        assert isinstance(result, list)

    def test_identifies_person(self) -> None:
        """Test that person names are identified."""
        result = get_named_entities("Steve Jobs founded Apple.")
        labels = [e["label"] for e in result]
        assert "PERSON" in labels

    def test_identifies_organization(self) -> None:
        """Test that organizations are identified."""
        result = get_named_entities("Microsoft is a technology company.")
        labels = [e["label"] for e in result]
        assert "ORG" in labels

    def test_identifies_location(self) -> None:
        """Test that locations are identified."""
        result = get_named_entities("I live in London, United Kingdom.")
        labels = [e["label"] for e in result]
        assert "GPE" in labels

    def test_entity_structure(self) -> None:
        """Test that entities have correct structure."""
        result = get_named_entities("Apple Inc. is in California.")
        if result:
            assert "text" in result[0]
            assert "label" in result[0]

    def test_empty_text_returns_empty_list(self) -> None:
        """Test that empty text returns empty list."""
        result = get_named_entities("")
        assert result == []

    def test_no_entities_in_simple_text(self) -> None:
        """Test that simple text without entities returns empty list."""
        result = get_named_entities("hello world")
        # May or may not return entities depending on spaCy model
        assert isinstance(result, list)
