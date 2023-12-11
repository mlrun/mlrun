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
JUDGE_CONFIG = {"device_map":"auto", "revision":"main", "trust_remote_code":False}
JUDGE_INFER_CONFIG = {"max_length": 300}
TOKENIZER_JUDGE_CONFIG = {"use_fast": True}
BENCHMARK_MODEL = "google/flan-t5-small"
BENCHMARK_CONFIG = {"max_length": 300}



@pytest.fixture
def prompt_fixture():
    prompt_config = {
        "name": "accuracy",
        "definition": "The accuracy of the provided answer.",
        "rubric": """Accuracy: This rubric assesses the accuracy of the provided answer. The details for different scores are as follows:
            - Score 1: The answer is completely incorrect or irrelevant to the question. It demonstrates a fundamental 
              misunderstanding of the topic or question.
            - Score 2: The answer contains significant inaccuracies, though it shows some understanding of the topic. Key 
              elements of the question are addressed incorrectly.
            - Score 3: The answer is partially correct but has noticeable inaccuracies or omissions. It addresses the 
              question but lacks depth or precision.
            - Score 4: The answer is mostly correct, with only minor inaccuracies or omissions. It provides a generally 
              accurate response to the question.
            - Score 5: The answer is completely correct and thorough. It demonstrates a deep and accurate understanding of 
              the topic, addressing all elements of the question effectively.""",
        "examples": """
            Question: What is the capital of France?
            
            Score 1: Completely Incorrect
            Answer: "The capital of France is Berlin."
            Explanation: This answer is entirely incorrect and irrelevant, as Berlin is the capital of Germany, not France.
            Score 2: Significantly Inaccurate
            Answer: "The capital of France is Lyon."
            Explanation: This answer demonstrates some understanding that the question is about a city in France, but it incorrectly identifies Lyon as the capital instead of Paris.
            Score 3: Partially Correct
            Answer: "I think the capital of France is either Paris or Marseille."
            Explanation: This answer shows partial knowledge but includes a significant inaccuracy by suggesting Marseille might be the capital. Paris is correct, but the inclusion of Marseille indicates a lack of certainty or complete understanding.
            Score 4: Mostly Correct
            Answer: "The capital of France is Paris, the largest city in the country."
            Explanation: This answer is mostly correct and identifies Paris as the capital. The addition of "the largest city in the country" is accurate but not directly relevant to the capital status, introducing a slight deviation from the question's focus.
            Score 5: Completely Correct and Thorough
            Answer: "The capital of France is Paris, which is not only the country's largest city but also its cultural and political center, hosting major institutions like the President's residence, the Elysée Palace."
            Explanation: This answer is completely correct, providing a thorough explanation that adds relevant context about Paris's role as the cultural and political center of France, directly addressing the question with depth and precision.
                     """,
    }
    return prompt_config


def test_single_grading_score(prompt_fixture):
    prompt_template = SINGLE_GRADE_PROMPT
    prompt_config = prompt_fixture
    q = "What is 2 + 2?"
    a = "2 + 2 equals 4"
    single_grading = LLMJudgeSingleGrading(
        name="accuracy_metrics",
        model_judge=JUDGE_MODEL,
        model_judge_config = JUDGE_CONFIG,
        model_judge_infer_config = JUDGE_INFER_CONFIG,
        tokenizer_judge_config = TOKENIZER_JUDGE_CONFIG,
        prompt_template=prompt_template,
        prompt_config=prompt_config,
    )
    result = single_grading.compute_over_one_data(q, a)
    assert 0 <= result["score"] <= 5


def test_pairwise_grading_scores():
    prompt_template = PAIR_GRADE_PROMPT
    prompt_config = prompt_fixture

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
    prompt_template = REF_GRADE_PROMPT
    prompt_config = prompt_fixture
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
