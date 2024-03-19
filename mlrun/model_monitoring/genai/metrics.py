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
from functools import wraps
import mpi4py
from typing import Union, List, Optional, Dict, Any, ClassVar, Tuple
import mlrun
from mlrun.utils import logger
from mlrun.model import ModelObj
from mlrun.model_monitoring.genai.prompt import (
    SINGLE_GRADE_PROMPT,
    PAIR_GRADE_PROMPT,
    REF_GRADE_PROMPT,
)
import transformers
import pandas as pd


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


def _check_mlrun_and_open_mpi() -> Tuple["mlrun.MLClientCtx", "mpi4py.MPI.Intracomm"]:
    is_mpi = False
    try:
        context = mlrun.get_or_create_ctx(name="mlrun")
        is_mpi = context.labels.get("kind", "job") == "mpijob"

        if is_mpi:
            try:
                from mpi4py import MPI

                return context, MPI.COMM_WORLD
            except ModuleNotFoundError as mpi4py_not_found:
                logger.error(
                    "To distribute the function using MLRun's 'mpijob' you need to have `mpi4py` package in your "
                    "interpreter. Please run `pip install mpi4py` and make sure you have open-mpi."
                )
                raise mpi4py_not_found
    except ModuleNotFoundError as module_not_found:
        if is_mpi:
            raise module_not_found
    return None, None


def open_mpi_handler(
    worker_inputs: str,
):
    # Check for MLRun and OpenMPI availability:
    context, comm = _check_mlrun_and_open_mpi()

    def decorator(handler):
        if comm is None or comm.Get_size() == 1:
            return handler

        @wraps(handler)
        def wrapper(**kwargs):
            # Get the open mpi environment properties:
            size = comm.Get_size()
            rank = comm.Get_rank()
            sample_df = kwargs[worker_inputs]

            # Give the correct chunk of the workers inputs:
            even_chunk_size = len(sample_df) // size
            chunk_start = rank * even_chunk_size
            chunk_end = (
                (rank + 1) * even_chunk_size if rank + 1 < size else len(input_argument)
            )
            logger.info(
                f"Rank #{rank}: Processing input chunk sample dataframe"
                f"from index {chunk_start} to {chunk_end}."
            )
            sample_df = sample_df.iloc[chunk_start:chunk_end:, :]
            kwargs[worker_input] = sample_df

            # Run the worker:
            output = handler(**kwargs)

            # Send the output to the root rank (rank #0):
            output = comm.gather(output, root=0)
            if rank == 0:
                # Join the outputs:
                logger.info("Collecting data from workers to root worker.")
                dataframe = pd.concat(objs=[df for df, _ in output], axis=0)
                return dataframe
            return None

        return wrapper

    return decorator


class LLMEvaluateMetric(ModelObj):
    """
    Base class of the metrics that computed by evluate package
    We need the y_true as the reference and y_pred as the prediction to compute the metrics
    """

    _dict_fields = ["name"]
    kind = "llm_metric"
    default_name: ClassVar[str] = "llm_metric"

    def __init__(self, name: str):
        """
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
        """
        Compute the metrics over the given data

        :param predictions: the predictions to compute the metrics over
        :param references: the references to compute the metrics over
        :param kwargs: other arguments to pass to the compute function
        :return: the metrics score and the explanation
        """
        if kwargs:
            return self.metric.compute(
                predictions=predictions, references=references, **kwargs
            )
        logger.info(f"Computing the metrics score of {self.name}")
        return self.metric.compute(predictions=predictions, references=references)


class LLMJudgeBaseMetric(ModelObj, ABC):
    """
    Base class of the metrics that computed by LLM as a judge
    We don't need the y_true as reference. These metrics are used for more open-ended question for the model
    and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
    """

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
        These metrics are used to evaluate the model performance on a given dataset

        :param name: name of the metric
        :param model_judge: the model judge to use
        :param model_judge_config: the model judge config
        :param tokenizer_judge_config: the tokenizer judge config
        :param model_judge_infer_config: the model judge infer config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
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
        logger.info(f"Filling the prompt template with the prompt config")
        return self.prompt_template.format(**self.prompt_config)

    @abstractmethod
    def prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        pass

    @abstractmethod
    def compute_over_one_data(self, question: str, response: str) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param question: the question to compute the metrics over
        :param response: the response to compute the metrics over
        :return: the metrics score and the explanation
        """
        pass

    @abstractmethod
    def compute_over_data(
        self, sample_df: pd.DataFrame, train_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param question: the question to compute the metrics over
        :param response: the response to compute the metrics over
        :return: the metrics score and the explanation
        """
        pass

    @abstractmethod
    def extract_score_explanation(self, result: str) -> Dict[str, Any]:
        """
        Abstract the store of the result
        :param result: the result text
        :return: the stored result
        """
        pass


class LLMJudgeSingleGrading(LLMJudgeBaseMetric):
    """
    Base class for LLM as a judge using single grading.
    you need to define the defnition of the metrics and give the grading of the rubic
    """

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
        init the class

        :param name: name of the metric
        :param model_judge: the model judge to use
        :param model_judge_config: the model judge config
        :param tokenizer_judge_config: the tokenizer judge config
        :param model_judge_infer_config: the model judge infer config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
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
        Prepare the judge model it will init the tokenizer and the model
        """
        logger.info(f"Preparing the judge model {self.model_judge}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_judge, **self.tokenizer_judge_config
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_judge, **self.model_judge_config
        )

    def compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param question: the question to compute the metrics over
        :param response: the response to compute the metrics over
        :return: the metrics score and the explanation
        """
        logger.info(
            f"Computing the metrics over one data point with {question} and {response}"
        )
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
        res_dic = self.extract_score_explanation(response)
        return res_dic

    @open_mpi_handler(worker_inputs="sample_df")
    def compute_over_data(
        self, sample_df: pd.DataFrame, train_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Compute the metrics over all data
        :param sample_df: the sample dataframe
        :param train_df: the train dataframe
        :return: the metrics score and the explanation
        """
        self.prepare_judge()
        res_df = pd.DataFrame(columns=["question", "answer", "score", "explanation"])

        logger.info(f"Computing the metrics over all data")
        for i in range(len(sample_df)):
            res_dic = self.compute_over_one_data(
                sample_df.loc[i, "question"], sample_df.loc[i, "answer"]
            )
            res_df.loc[i] = [
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answer"],
                res_dic["score"],
                res_dic["explanation"],
            ]

        return res_df

    def extract_score_explanation(self, result: str) -> Dict[str, Any]:
        """
        Abstract the store of the result
        :param result: the result to store
        :return: the stored result
        """
        logger.info(f"Extracting the score and explanation from {result}")
        score_pattern = r"\bscore:\s*(\d+)\b"
        explanation_pattern = r"explanation:\s*(.*?)\s*(?=\bScore:|$)"

        score_match = re.search(score_pattern, result)
        score = int(score_match.group(1)) if score_match else None

        explanation_match = re.search(explanation_pattern, result, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else None

        return {"score": score, "explanation": explanation}


class LLMJudgePairwiseGrading(LLMJudgeBaseMetric):
    """
    Base class for LLM as a judge using pairwise grading.
    you need to define the defnition of the metrics and give the grading of the rubic
    you need to give a base model to compare the model to
    """

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
        init the class

        :param name: name of the metric
        :param model_judge: the model judge to use
        :param tokenizer_judge_config: the tokenizer judge config
        :param model_judge_config: the model judge config
        :param model_judge_infer_config: the model judge infer config
        :param model_bench_mark: the model bench mark to use
        :param model_bench_mark_config: the model bench mark config
        :param model_bench_mark_infer_config: the model bench mark infer config
        :param tokenizer_bench_mark_config: the tokenizer bench mark config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
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
        init the tokenizer and the model
        """
        logger.info(f"Preparing the judge model {self.model_judge}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_judge, **self.tokenizer_judge_config
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_judge, **self.model_judge_config
        )

    def prepare_bench_mark_model(self) -> None:
        """
        Prepare the model that used for bench marking
        """
        logger.info(f"Preparing the bench mark model {self.model_bench_mark}")
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
        logger.info(f"Computing the bench mark response for {question}")
        input_ids = self.tokenizer_bench_mark(question, return_tensors="pt").input_ids
        outputs = self.model_bench_mark.generate(
            input_ids,
            pad_token_id=self.tokenizer_bench_mark.pad_token_id,
            eos_token_id=self.tokenizer_bench_mark.eos_token_id,
            **self.model_bench_mark_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer_bench_mark.decode(
            response_ids, skip_special_tokens=True
        )
        logger.info(f"Response of the bench mark model is {response}")

        return response

    def compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        logger.info(f"Computing the metrics over {question} and {response}")
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

        logger.info(f"Response of the judge model is {response}")
        res_dic = self.extract_score_explanation(response)
        res_dic["answerB"] = self.prompt_config["answerB"]
        return res_dic

    @open_mpi_handler(worker_inputs="sample_df")
    def compute_over_data(
        self, sample_df: pd.DataFrame, train_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Compute the metrics over all data
        :param sample_df: the sample dataframe
        :param train_df: the train dataframe
        :return: the metrics score and the explanation
        """
        self.prepare_judge()
        self.prepare_bench_mark_model()
        res_df = pd.DataFrame(
            columns=[
                "question",
                "answerA",
                "answerB",
                "score_of_assistant_a",
                "explanation_of_assistant_a",
                "score_of_assistant_b",
                "explanation_of_assistant_b",
            ]
        )

        for i in range(len(sample_df)):
            res_dic = self.compute_over_one_data(
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answerA"],
            )
            res_df.loc[i] = [
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answerA"],
                res_dic["answerB"],
                res_dic["score_of_assistant_a"],
                res_dic["explanation_of_assistant_a"],
                res_dic["score_of_assistant_b"],
                res_dic["explanation_of_assistant_b"],
            ]

        return res_df

    def extract_score_explanation(self, response) -> Dict[str, Any]:
        """
        Extract the score and the explanation from the response
        :param response: the response to extract the score and the explanation from
        :return: the score and the explanation
        """
        # Find the position of the "[Output]:" marker
        output_marker_index = response.find("[Output]:")
        if output_marker_index == -1:
            return "No '[Output]:' marker found"

        # Extract the part of the response after the "[Output]:" marker
        response_after_output = response[output_marker_index + len("[Output]:") :]

        # Adjusted pattern to match the text format and separate lines
        pattern = r"- score of assistant ([ab]): (\d)\s*- explanation of assistant \1: (.*?)\s*(?=- score of assistant|$)"

        matches = re.findall(pattern, response_after_output, re.DOTALL)

        if matches:
            result_dict = {}
            for match in matches:
                assistant, score, explanation = match
                result_dict[f"score_of_assistant_{assistant}"] = int(score)
                result_dict[
                    f"explanation_of_assistant_{assistant}"
                ] = explanation.strip()
            return result_dict
        else:
            raise ValueError(
                "No matches found after '[Output]:' marker. "
                "Please check the format of the response."
            )


class LLMJudgeReferenceGrading(LLMJudgePairwiseGrading):
    """
    LLM Judge Reference Grading class
    you need to give the name of the metrics, give the grading rubric and the bench mark model to use
    This class requrie you know the y_true of the response
    """

    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "model_judge_infer_config",
        "tokenizer_judge_config",
        "model_bench_mark",
        "model_bench_mark_config",
        "model_bench_mark_infer_config",
        "tokenizer_bench_mark_config",
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
        init the grading with reference class

        :param name: the name of the metrics
        :param model_judge: the model to use for grading
        :param model_judge_config: the config of the model to use for grading
        :param model_judge_infer_config: the config of the model to use for inference
        :param tokenizer_judge_config: the config of the tokenizer to use for grading
        :param model_bench_mark: the model to use for bench marking
        :param model_bench_mark_config: the config of the model to use for bench marking
        :param tokenizer_bench_mark_config: the config of the tokenizer to use for bench marking
        :param model_bench_mark_infer_config: the config of the model to use for inference
        :param prompt_template: the template of the prompt to use
        :param prompt_config: the config of the prompt to use
        """
        super().__init__(
            name,
            model_judge,
            tokenizer_judge_config,
            model_judge_config,
            model_judge_infer_config,
            model_bench_mark,
            model_bench_mark_config,
            model_bench_mark_infer_config,
            tokenizer_bench_mark_config,
            prompt_template,
            prompt_config,
        )

    def compute_over_one_data(self, question, response, reference) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        self.prompt_config["reference"] = reference
        res_dic = super().compute_over_one_data(question, response)
        return res_dic

    @open_mpi_handler(worker_inputs="sample_df")
    def compute_over_data(
        self, sample_df: pd.DataFrame, train_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Compute the metrics over a dataset

        :param sample_df: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        self.prepare_judge()
        self.prepare_bench_mark_model()
        res_df = pd.DataFrame(
            columns=[
                "question",
                "answerA",
                "answerB",
                "reference",
                "score_of_assistant_a",
                "explanation_of_assistant_a",
                "score_of_assistant_b",
                "explanation_of_assistant_b",
            ]
        )

        for i in range(len(sample_df)):
            res_dic = self.compute_over_one_data(
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answerA"],
                sample_df.loc[i, "reference"],
            )
            res_df.loc[i] = [
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answerA"],
                sample_df.loc[i, "reference"],
                res_dic["answerB"],
                res_dic["score_of_assistant_a"],
                res_dic["explanation_of_assistant_a"],
                res_dic["score_of_assistant_b"],
                res_dic["explanation_of_assistant_b"],
            ]

        return res_df
