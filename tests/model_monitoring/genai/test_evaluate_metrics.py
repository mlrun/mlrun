import pytest
from mlrun.model_monitoring.genai.metrics_base import LLMEvaluateMetric


def test_bleu_metric():
    metric = LLMBaseMetric(name="bleu")
    predictions = ["the cat is on the mat"]
    references = ["the cat is playing on the mat"]
    score = metric.compute_over_data(predictions, references)
    assert score > 0
    assert score <= 1


def test_rouge_metric():
    metric = LLMBaseMetric(name="rouge")
    predictions = ["cat on the mat"]
    references = ["the cat is playing on the blue mat"]
    score = metric.compute_over_data(predictions, references)
    assert score > 0
    assert score <= 1


def test_invalid_metric():
    with pytest.raises(ImportError):
        LLMBaseMetric(name="invalid")


def test_bleu_wrong_inputs():
    metric = LLMBaseMetric(name="bleu")

    # Wrong types
    with pytest.raises(TypeError):
        metric.compute_over_data("text", 5)

    # Different cardinality
    with pytest.raises(ValueError):
        predictions = ["hi"]
        references = ["1", "2"]
        metric.compute_over_data(predictions, references)
