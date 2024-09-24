# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import ast
import enum
import re
from abc import ABC, abstractmethod
from typing import Any, Union

import openai
import pandas as pd
import transformers

import mlrun
import mlrun.common.schemas
from mlrun.model import ModelObj
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBaseV2,
    ModelMonitoringApplicationResult,
)
from mlrun.utils import logger

# These prompt are used to generate the grade for LLM-as a judge

"""
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang
      and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li
      and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez
      and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

SINGLE_GRADE_PROMPT = """
Task:
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user question displayed below.
You will be given the definition of {name}, grading rubric, context information.
Your task is to determine a numerical score of {name} for the response.
You must use the grading rubric to determine your score.
You must also give an explanation about how did you determine the score step-by-step.
Please use chain of thinking.
Examples could be included below for your reference.
Make sure you understand the grading rubric and use the examples before completing the task.
[Examples]:
{examples}
[User Question]:
{question}
[Response]:
{answer}
[Definition of {name}]:
{definition}
[Grading Rubric]:
{rubric}
Answer the following question and return as a python dictionary:
{{"score": <a numerical score of {name} for the response>
"explanation": <a string value of an explanation about how did you determine the score step-by-step>}}
[Output]:
"""

PAIR_GRADE_PROMPT = """
Task:
Your task is to determine two numerical score of {name} for the responses from two AI assistants.
You must use the grading rubric to determine your scores.
You must also give an explanation about how did you determine the scores step-by-step.
Please using chain of thinking.
Examples could be included below for your reference.
Make sure you understand the grading rubric and use the examples before completing the task.
[Examples]:
{examples}
[User Question]:
{question}
[Response of assistant A]:
{answerA}
[Response of assistant B]:
{answerB}
[Definition of {name}]:
{definition}
[Grading Rubric]:
{rubric}
Answer the following question and return as a python dictionary:
{{"score_a": <a numerical score of {name} for the response>,
"explanation_a": <a string value of an explanation about how did you determine the score step-by-step>,
"score_b": <a numerical score of {name} for the response>,
"explanation_b": <a string value of an explanation about how did you determine the score step-by-step>,
}}
[Output]:
"""

REF_GRADE_PROMPT = """
Task:
Your task is to determine two numerical score of {name} for the responses
from two AI assistants with the ground truth of the response.
You must use the grading rubric to determine your scores.
You must use the ground truth of the response.
You need to give an explanation about how did you compare with the ground truth of the
response to determine the scores step-by-step.
Please using chain of thinking.
Examples could be included below for your reference.
Make sure you understand the grading rubric and use the examples before completing the task.
[Examples]:
{examples}
[User Question]:
{question}
[Response of assistant A]:
{answerA}
[Response of assistant B]:
{answerB}
[Ground truth of the response]:
{reference}
[Definition of {name}]:
{definition}
[Grading Rubric]:
{rubric}
Answer the following question and return as a python dictionary:
{{"score_a": <a numerical score of {name} for the response>,
"explanation_a": <a string value of an explanation about how did you compare with
the ground truth of the response to determine the score step-by-step>,
"score_b": <a numerical score of {name} for the response>,
"explanation_b": <a string value of an explanation about how did you compare with
the ground truth of the response to determine the score step-by-step>,
}}
[Output]:
"""


class JudgeTypes(enum.Enum):
    custom_grading = "custom-grading"
    single_grading = "single-grading"
    pairwise_grading = "pairwise-grading"
    reference_grading = "reference-grading"

    @classmethod
    def to_list(cls):
        return [judge.value for judge in cls]


class BaseJudge(ModelObj, ABC):
    """
    Base class of the metrics that computed by LLM as a judge
    We don't need the y_true as reference. These metrics are used for more open-ended question for the model
    and the algorithm are based on the paper https://arxiv.org/pdf/2306.05685.pdf
    """

    def __init__(
        self,
        metric_name: str,
        judge_type: str,
        model_name: str,
        prompt_template: str = None,
        prompt_config: dict[str, str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the class.

        :param metric_name:         Name of the metric to be saved as the name of the result of the application
        :param judge_type:          The Judge type to use, Need to be one of the values in LLM_JUDGE_TYPES
        :param model_name:          Name of the judge model to use
        :param prompt_template:     The prompt template to fill with the prompt configuration
        :param prompt_config:       The prompt configuration that will fill the prompt template
        :param verbose:             The verboisty level of the logger.
        """
        self.metric_name = metric_name
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.prompt_config = prompt_config or {}
        self.verbose = verbose
        judge_type = judge_type.casefold()
        if judge_type not in JudgeTypes.to_list():
            raise ValueError(
                f"Judge type ({judge_type}) not supported. Please choose one of: {JudgeTypes.to_list()}"
            )
        self.judge_type = judge_type

        if not self.prompt_template:
            if self.judge_type == JudgeTypes.custom_grading.value:
                raise ValueError(
                    "Must pass `prompt_template` when using custom-grading judge type"
                )
            if self.judge_type == JudgeTypes.single_grading.value:
                self.prompt_template = SINGLE_GRADE_PROMPT
            elif self.judge_type == JudgeTypes.pairwise_grading.value:
                self.prompt_template = PAIR_GRADE_PROMPT
            else:
                self.prompt_template = REF_GRADE_PROMPT

    def _fill_prompt(
        self, answer: str, question: str = None, reference: str = None
    ) -> str:
        """
        Fill the prompt template with the prompt config

        :param answer:       the answer to fill the prompt with
        :param question:     the question to fill the prompt with
        :param reference:    the reference to fill the prompt with

        :returns: the filled prompt
        """
        original_config = self.prompt_config.copy()
        if question:
            self.prompt_config["question"] = question

        # Updating prompt config:
        if self.judge_type in [
            JudgeTypes.single_grading.value,
            JudgeTypes.custom_grading.value,
        ]:
            self.prompt_config["answer"] = answer
        else:
            self.prompt_config["answerA"] = answer
            self.prompt_config["question"] = question
            self.prompt_config["answerB"] = self._invoke_benchmark_model(question)

            if self.judge_type == JudgeTypes.reference_grading.value:
                self.prompt_config["reference"] = reference

        if self.verbose:
            logger.info("Constructing the prompt")
        prompt = self.prompt_template.format(**self.prompt_config)
        self.prompt_config = original_config
        return prompt

    def _extract_single_grade_score_explanation(self, response: str):
        if self.verbose:
            logger.info(
                f"Extracting the score and explanation from the result:\n{response}"
            )
        try:
            return ast.literal_eval(response)
        except Exception:
            score = 0
            explanation = "Failed to retrieve judge's decision"
            return {
                "score": score,
                "explanation": explanation,
            }

    def _extract_pairwise_grade_score_explanation(self, response) -> dict[str, Any]:
        """
        Extract the score and the explanation from the response.

        :param response:    the response to extract the score and the explanation from

        :returns:   the score and the explanation
        """
        if self.verbose:
            logger.info(
                f"Extracting the score and explanation from the result:\n{response}"
            )
        try:
            return ast.literal_eval(response)
        except Exception:
            score = 0
            explanation = "Failed to retrieve judge's decision"
            return {
                "score_a": score,
                "explanation_a": explanation,
                "score_b": score,
                "explanation_b": explanation,
            }

    def judge(self, sample_df: pd.DataFrame):
        method_name = "_" + self.judge_type.replace("-", "_")
        method = getattr(self, method_name)
        if self.verbose:
            logger.info("Computing the metrics over all data")
        return method(sample_df)

    def _custom_grading(self, sample_df: pd.DataFrame):
        question = "question" in sample_df.columns
        columns = ["question", "answer", "score", "explanation"]
        if not question:
            columns.remove("question")

        result_df = pd.DataFrame(columns=columns)

        for i in range(len(sample_df)):
            answer = sample_df.loc[i, "answer"]
            if question:
                question = sample_df.loc[i, "question"]
            if self.verbose:
                logger.info(
                    f"Computing the metrics over one data point with the following answer:"
                    f"- Answer: {answer}"
                )
            # preparing prompt:
            if question:
                prompt = self._fill_prompt(
                    answer=answer,
                    question=question,
                )
            else:
                prompt = self._fill_prompt(
                    answer=answer,
                )

            # Invoking the judge model:
            content = self._invoke(prompt)

            # Extracting score and explanation:
            result_dict = self._extract_single_grade_score_explanation(content)

            # Add result to dataframe:
            result = [answer, result_dict["score"], result_dict["explanation"]]
            if question:
                result = [question] + result
            result_df.loc[i] = result

        return result_df

    def _single_grading(self, sample_df: pd.DataFrame):
        result_df = pd.DataFrame(columns=["question", "answer", "score", "explanation"])
        for i in range(len(sample_df)):
            question, answer = sample_df.loc[i, "question"], sample_df.loc[i, "answer"]
            if self.verbose:
                logger.info(
                    f"Computing the metrics over one data point with the following question and answer:"
                    f"- Question: {question}"
                    f"- Answer: {answer}"
                )
            # preparing prompt:
            prompt = self._fill_prompt(
                answer=answer,
                question=question,
            )

            # Invoking the judge model:
            content = self._invoke(prompt)

            # Extracting score and explanation:
            result_dict = self._extract_single_grade_score_explanation(content)

            # Add result to dataframe:
            result_df.loc[i] = [
                question,
                answer,
                result_dict["score"],
                result_dict["explanation"],
            ]

        return result_df

    def _pairwise_grading(self, sample_df: pd.DataFrame, with_reference: bool = False):
        columns = [
            "question",
            "answerA",
            "answerB",
            "score_a",
            "explanation_a",
            "score_b",
            "explanation_b",
        ]
        if with_reference:
            columns.append("reference")
        result_df = pd.DataFrame(columns=columns)

        for i in range(len(sample_df)):
            question, answer = sample_df.loc[i, "question"], sample_df.loc[i, "answer"]
            reference = sample_df.loc[i, "reference"] if with_reference else None
            if self.verbose:
                logger.info(
                    f"Computing the metrics over one data point with the following question and answer:\n"
                    f"- Question: {question}\n"
                    f"- Answer: {answer}"
                )

            # preparing prompt:
            prompt = self._fill_prompt(
                answer=answer,
                question=question,
                reference=reference,
            )

            # Invoking the judge model:
            content = self._invoke(prompt)

            # Extracting score and explanation:
            result_dict = self._extract_pairwise_grade_score_explanation(content)

            # Add result to dataframe:
            result_row = [
                question,
                answer,
                self.prompt_config["answerB"],
                result_dict["score_a"],
                result_dict["explanation_a"],
                result_dict["score_b"],
                result_dict["explanation_b"],
            ]
            if with_reference:
                result_row.append(reference)
            result_df.loc[i] = result_row

        return result_df

    def _reference_grading(self, sample_df: pd.DataFrame):
        return self._pairwise_grading(sample_df=sample_df, with_reference=True)

    @abstractmethod
    def _invoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    def _invoke_benchmark_model(self, prompt: str) -> str:
        pass


class OpenAIJudge(BaseJudge, ABC):
    def __init__(
        self,
        metric_name: str,
        judge_type: str,
        model_name: str,
        prompt_template: str = None,
        prompt_config: dict[str, str] = None,
        verbose: bool = True,
        benchmark_model_name: str = None,
    ):
        super().__init__(
            metric_name=metric_name,
            judge_type=judge_type,
            model_name=model_name,
            prompt_template=prompt_template,
            prompt_config=prompt_config,
            verbose=verbose,
        )
        self.benchmark_model_name = benchmark_model_name
        if self.verbose:
            logger.info("Establishing connection to OpenAI")
        api_key = mlrun.get_secret_or_env("OPENAI_API_KEY")
        base_url = mlrun.get_secret_or_env("OPENAI_API_BASE")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _invoke(self, prompt: str, model_name: str = None) -> str:
        model_name = model_name or self.model_name
        # Invoke OpenAI model:
        result = self.client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )

        content = result.choices[0].message.content

        return content

    def _invoke_benchmark_model(self, prompt: str) -> str:
        return self._invoke(prompt, model_name=self.benchmark_model_name)


class HuggingfaceJudge(BaseJudge, ABC):
    def __init__(
        self,
        metric_name: str,
        judge_type: str,
        model_name: str,
        prompt_template: str = None,
        prompt_config: dict[str, str] = None,
        verbose: bool = True,
        model_config: dict[str, Any] = None,
        tokenizer_config: dict[str, Any] = None,
        model_infer_config: dict[str, Any] = None,
        benchmark_model_name: str = None,
        benchmark_model_config: dict[str, Any] = None,
        benchmark_tokenizer_config: dict[str, Any] = None,
        benchmark_model_infer_config: dict[str, Any] = None,
    ):
        super().__init__(
            metric_name=metric_name,
            judge_type=judge_type,
            model_name=model_name,
            prompt_template=prompt_template,
            prompt_config=prompt_config,
            verbose=verbose,
        )
        self.model_config = model_config or {}
        self.tokenizer_config = tokenizer_config or {}
        self.model_infer_config = model_infer_config or {}
        if self.verbose:
            logger.info(f"Loading the judge model {self.model_name} from Huggingface")

        # Loading the model:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name, **self.tokenizer_config
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, **self.model_config
        )

        # Loading the benchmark model if needed:
        if self.judge_type != JudgeTypes.single_grading.value:
            if self.verbose:
                logger.info(
                    f"Loading the benchmark model {self.model_name} from Huggingface"
                )
            self.benchmark_model_name = benchmark_model_name
            self.benchmark_model_config = benchmark_model_config or {}
            self.benchmark_tokenizer_config = benchmark_tokenizer_config or {}
            self.benchmark_model_infer_config = benchmark_model_infer_config or {}

            self.benchmark_tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.benchmark_model_name, **self.benchmark_tokenizer_config
            )
            self.benchmark_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.benchmark_model_name, **self.benchmark_model_config
            )

    def _invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.model_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

    def _invoke_benchmark_model(self, prompt: str):
        input_ids = self.benchmark_tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.benchmark_model.generate(
            input_ids,
            pad_token_id=self.benchmark_tokenizer.pad_token_id,
            eos_token_id=self.benchmark_tokenizer.eos_token_id,
            **self.benchmark_model_infer_config,
        )

        response_ids = outputs[0]
        response = self.benchmark_tokenizer.decode(
            response_ids, skip_special_tokens=True
        )

        return response


FRAMEWORKS = {
    "openai": OpenAIJudge,
    "huggingface": HuggingfaceJudge,
}

STATUS_RESULT_MAPPING = {
    0: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.detected,
    1: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.no_detection,
}


class LLMAsAJudgeApplication(ModelMonitoringApplicationBaseV2):
    def __init__(
        self,
        **kwargs,
    ):
        framework = kwargs.pop("framework")
        self.name = "llm-as-a-judge"
        self.llm_judge = FRAMEWORKS[framework](**kwargs)

    def do_tracking(
        self,
        monitoring_context,
    ) -> Union[
        ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
    ]:
        judge_result = self.llm_judge.judge(monitoring_context.sample_df)

        # log artifact:
        pattern = re.compile("[ :+.]")
        tag = re.sub(pattern, "-", str(monitoring_context.end_infer_time))
        monitoring_context.log_dataset(
            key=self.llm_judge.metric_name,
            df=judge_result,
            tag=tag,
        )

        # calculate value:
        score_column = (
            "score"
            if self.llm_judge.judge_type == JudgeTypes.single_grading.value
            else "score_a"
        )
        mean_score = judge_result[score_column].mean()

        # get status:
        status = STATUS_RESULT_MAPPING[round(mean_score)]

        return ModelMonitoringApplicationResult(
            name=self.llm_judge.metric_name,
            value=mean_score,
            kind=mlrun.common.schemas.model_monitoring.constants.ResultKindApp.model_performance,
            status=status,
            extra_data={},
        )
