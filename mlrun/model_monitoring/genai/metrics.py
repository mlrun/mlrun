# Copyright 2023 Iguazio
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

# This is the base classes of using evluate to compute the metrics score and
# Using LLM as a Judge to compute the metrics score
import re
import torch
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Any, ClassVar
from mlrun.model import ModelObj
from mlrun.model_monitoring.genai.prompt import (
    SINGLE_GRADE_PROMPT,
    PAIR_GRADE_PROMPT,
    REF_GRADE_PROMPT,
)
import transformers


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


class LLMEvaluateMetric(ModelObj):
    _dict_fields = ["name"]
    kind = "llm_metric"
    default_name: ClassVar[str] = "llm_metric"

    def __init__(self, name: str):
        """
        Base class for evaluate metrics
        These metrics are used to evaluate the model performance on a given dataset
        and the algorithm is implemented in the evaluate library
        :param name: name of the metric
        """
        try:
            import evaluate
        except ImportError:
            raise ImportError(
                "evaluate library is not installed. Please install it using pip install evaluate"
            )
        self.name = name or self.default_name
        self.metric = evaluate.load(name)

    def compute_over_data(
        self, predictions: Union[List, Dict], references: Union[List, Dict], **kwargs
    ) -> Dict[str, Any]:
        if kwargs:
            return self.metric.compute(
                predictions=predictions, references=references, **kwargs
            )
        return self.metric.compute(predictions=predictions, references=references)


class LLMJudgeBaseMetric(ModelObj, ABC):
    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "tokenizer_judge_config",
        "model_judge_infer_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_metric"
    default_name: ClassVar[str] = "llm_judge_metric"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, Any],
        tokenizer_judge_config: Dict[str, Any],
        model_judge_infer_config: Dict[str, Any],
        prompt_template: str,
        prompt_config: Dict[str, Any],
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name or self.default_name
        self.model_judge = model_judge
        self.model_judge_config = model_judge_config
        self.tokenizer_judge_config = tokenizer_judge_config
        self.model_judge_infer_config = model_judge_infer_config
        self.prompt_template = prompt_template
        self.prompt_config = prompt_config

    def fill_prompt(self) -> str:
        """
        Fill the prompt template with the prompt config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
        :return: the filled prompt
        """
        return self.prompt_template.format(**self.prompt_config)

    @abstractmethod
    def prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        pass

    @abstractmethod
    def compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        pass

    # @abstractmethod
    # def compute_over_all_data(self, questions, responses) -> Dict[str, Any]:
    #    """
    #    Compute the metrics over one data point
    #    :param kwargs: the data to compute the metrics over
    #    :return: the metrics score and the explanation
    #    """
    #    pass

    # @abstractmethod
    # def agg_score(self, scores: List[int]) -> float:
    #    """
    #    Aggregate the scores
    #    :param scores: the scores to aggregate
    #    :return: the aggregated score
    #    """
    #    pass

    @abstractmethod
    def extract_score_explanation(self, result: str) -> Dict[str, Any]:
        """
        Abstract the store of the result
        :param result: the result to store
        :return: the stored result
        """
        pass


class LLMJudgeSingleGrading(LLMJudgeBaseMetric):
    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "tokenizer_judge_config",
        "model_judge_infer_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_single_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, Any],
        tokenizer_judge_config: Dict[str, Any],
        model_judge_infer_config: Dict[str, Any],
        prompt_template: str,
        prompt_config: Dict[str, Any],
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        super().__init__(
            name,
            model_judge,
            model_judge_config,
            tokenizer_judge_config,
            model_judge_infer_config,
            prompt_template,
            prompt_config,
        )

    def prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_judge, **self.tokenizer_judge_config
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_judge, **self.model_judge_config
        )

    def compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        self.prepare_judge()
        self.prompt_config["question"] = question
        self.prompt_config["answer"] = response
        input_ids = self.tokenizer(self.fill_prompt(), return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.model_judge_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return self.extract_score_explanation(response)

    def extract_score_explanation(self, result: str) -> Dict[str, Any]:
        """
        Abstract the store of the result
        :param result: the result to store
        :return: the stored result
        """
        print(result)
        score_pattern = r"\bscore:\s*(\d+)\b"
        explanation_pattern = r"explanation:\s*(.*?)\s*(?=\bScore:|$)"

        score_match = re.search(score_pattern, result)
        score = int(score_match.group(1)) if score_match else None

        explanation_match = re.search(explanation_pattern, result, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else None

        return {"score": score, "explanation": explanation}


class LLMJudgePairwiseGrading(LLMJudgeBaseMetric):
    _dict_fields = [
        "name",
        "model_judge",
        "tokenizer_judge_config",
        "model_judge_config",
        "model_judge_infer_config",
        "model_bench_mark",
        "model_bench_mark_config",
        "model_bench_mark_infer_config",
        "tokenizer_bench_mark_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_pairwise_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        tokenizer_judge_config: Dict[str, Any],
        model_judge_config: Dict[str, Any],
        model_judge_infer_config: Dict[str, Any],
        model_bench_mark: str,
        model_bench_mark_config: Dict[str, Any],
        model_bench_mark_infer_config: Dict[str, Any],
        tokenizer_bench_mark_config: Dict[str, Any],
        prompt_template: str,
        prompt_config: Dict[str, Any],
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        super().__init__(
            name,
            model_judge,
            model_judge_config,
            tokenizer_judge_config,
            model_judge_infer_config,
            prompt_template,
            prompt_config,
        )
        self.model_bench_mark = model_bench_mark
        self.model_bench_mark_config = model_bench_mark_config
        self.model_bench_mark_infer_config = model_bench_mark_infer_config
        self.tokenizer_bench_mark_config = tokenizer_bench_mark_config

    def prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_judge, **self.tokenizer_judge_config
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_judge, **self.model_judge_config
        )

    def prepare_bench_mark_model(self) -> None:
        """
        Prepare the base model
        """
        self.tokenizer_bench_mark = transformers.AutoTokenizer.from_pretrained(
            self.model_bench_mark, **self.tokenizer_bench_mark_config
        )
        self.model_bench_mark = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_bench_mark, **self.model_bench_mark_config
        )

    def compute_bench_mark_response(self, question) -> str:
        """
        Compute the response of the bench mark model
        :param question: the question to ask the model
        :return: the response
        """
        self.prepare_bench_mark_model()
        input_ids = self.tokenizer_bench_mark(question, return_tensors="pt").input_ids
        outputs = self.model_bench_mark.generate(
            input_ids,
            pad_token_id=self.tokenizer_bench_mark.pad_token_id,
            eos_token_id=self.tokenizer_bench_mark.eos_token_id,
            **self.model_bench_mark_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer_bench_mark.decode(response_ids, skip_special_tokens=True)

        return response

    def compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        self.prepare_judge()
        self.prompt_config["question"] = question
        self.prompt_config["answerA"] = response
        self.prompt_config["answerB"] = self.compute_bench_mark_response(question)
        input_ids = self.tokenizer(self.fill_prompt(), return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.model_judge_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return self.extract_score_explanation(response)


    def extract_score_explanation(self, response) -> Dict[str, Any]:
        """
        Extract the scores and explanations for the professionalism of two AI assistants' responses using regex and return them in a dictionary.
        param response: The combined response containing scores and explanations for both assistants.
        return: A dictionary containing the scores and explanations for both assistants.
        """
        pattern = r"Score of Assistant ([AB]): (\d)\s*Explanation of Assistant \1: (.*?)\n(?=Score of Assistant|$)"

        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            result_dict = {}
            for match in matches:
                assistant, score, explanation = match
                result_dict[f"score_of_assistant_{assistant.lower()}"] = int(score)
                result_dict[
                    f"explanation_of_assistant_{assistant.lower()}"
                ] = explanation.strip()
            return result_dict
        else:
            return "No matches found"


class LLMJudgeReferenceGrading(LLMJudgePairwiseGrading):
    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "model_judge_infer_config",
        "tokenizer_judge_config",
        "model_bench_mark",
        "model_bench_mark_config",
        "model_bench_mark_infer_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_reference_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, Any],
        model_judge_infer_config: Dict[str, Any],
        tokenizer_judge_config: Dict[str, Any],
        model_bench_mark: str,
        model_bench_mark_config: Dict[str, Any],
        tokenizer_bench_mark_config: Dict[str, Any],
        model_bench_mark_infer_config: Dict[str, Any],
        prompt_template: str,
        prompt_config: Dict[str, str],
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        super().__init__(
            name,
            model_judge,
            model_judge_config,
            model_judge_infer_config,
            tokenizer_judge_config,
            model_bench_mark,
            model_bench_mark_config,
            tokenizer_bench_mark_config,
            model_bench_mark_infer_config,
            prompt_template,
            prompt_config,
        )

    def compute_over_one_data(self, question, response, reference) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        self.prepare_judge()
        self.prompt_config["question"] = question
        self.prompt_config["reference"] = reference
        self.prompt_config["answerA"] = response
        self.prompt_config["answerB"] = self.compute_bench_mark_response(question)
        input_ids = self.tokenizer(self.fill_prompt(), return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.model_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return self.extract_score_explanation(response)
