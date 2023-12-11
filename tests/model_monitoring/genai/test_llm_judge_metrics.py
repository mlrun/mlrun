import pytest
from mlrun.model_monitoring.genai.metrics_base import (
    LLMJudgeSingleGrading,
    LLMJudgePairwiseGrading,
    LLMJudgeReferenceGrading,
)

from mlrun.model_monitoring.genai.prompt import (
    SINGLE_GRADE_PROMPT,
    PAIR_GRADE_PROMPT,
    REF_GRADE_PROMPT,
)

JUDGE_MODEL = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
JUDGE_CONFIG = {"max_length": 300}


@pytest.fixture
def single_grading():
    prompt_template = SINGLE_GRADE_PROMPT
    prompt_config = {"name": "accuracy"}
    return LLMJudgeSingleGrading(
        model_judge=JUDGE_MODEL,
        model_judge_config=JUDGE_CONFIG,
        prompt_template=prompt_template,
        prompt_config=prompt_config,
    )


def test_single_grading_score(single_grading):
    q = "What is 2 + 2?"
    a = "2 + 2 equals 4"
    result = single_grading.compute_over_one_data(q, a)
    assert 0 <= result["score"] <= 5


def test_pairwise_grading_scores():
    prompt_template = PAIR_GRADE_PROMPT
    prompt_config = {}

    metric = LLMJudgePairwiseGrading(
        model_judge=JUDGE_MODEL,
        prompt_template=prompt_template,
        prompt_config=prompt_config,
    )

    q = "What is the capital of France?"
    a1 = "The capital of France is Paris"
    a2 = "France's capital city is Lyon"

    result = metric.compute_over_one_data(q, a1, a2)
    scores = metric.extract_score_and_explanation(result["response"])
    assert 0 <= scores["score_of_assistant_a"] <= 5
    assert 0 <= scores["score_of_assistant_b"] <= 5


def test_reference_grading_scores():
    metric = LLMJudgeReferenceGrading(
        model_judge=JUDGE_MODEL, prompt_template=REF_GRADE_PROMPT, prompt_config={}
    )

    q = "Who wrote Hamlet?"
    a1 = "Hamlet was written by Charles Dickens"
    a2 = "William Shakespeare wrote the play Hamlet"
    ref = "The author of the play Hamlet is William Shakespeare"

    result = metric.compute_over_one_data(q, a1, a2, ref)
    scores = metric.extract_score_and_explanation(result["response"])

    assert 0 <= scores["score_of_assistant_a"] <= 5
    assert 0 <= scores["score_of_assistant_b"] <= 5
