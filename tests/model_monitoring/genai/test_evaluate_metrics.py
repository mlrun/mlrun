import pytest
from mlrun.model_monitoring.genai.metrics import LLMEvaluateMetric


def test_bleu_metric():
    metric = LLMEvaluateMetric(name="bleu")
    predictions = ["the cat is on the mat"]
    references = ["the cat is playing on the mat"]
    score = metric.compute_over_data(predictions, references)
    assert score["bleu"] >= 0
    assert score["bleu"] <= 1


def test_rouge_metric():
    metric = LLMEvaluateMetric(name="rouge")
    predictions = ["cat on the mat"]
    references = ["the cat is playing on the blue mat"]
    score = metric.compute_over_data(predictions, references)
    assert score["rouge1"] >= 0
    assert score["rouge1"] <= 1
    assert score["rouge2"] >= 0
    assert score["rouge2"] <= 1


def test_invalid_metric():
    with pytest.raises(FileNotFoundError):
        LLMEvaluateMetric(name="invalid")


def test_bleu_wrong_inputs():
    metric = LLMEvaluateMetric(name="bleu")

    # Wrong types
    with pytest.raises(TypeError):
        metric.compute_over_data("text", 5)

    # Different cardinality
    with pytest.raises(ValueError):
        predictions = ["hi"]
        references = ["1", "2"]
        metric.compute_over_data(predictions, references)
