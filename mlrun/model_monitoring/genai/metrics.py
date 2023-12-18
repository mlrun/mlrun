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
    global _LOGGER
    is_mpi = False
    try:
        import mlrun
        context = mlrun.get_or_create_ctx(name="mlrun")
        _LOGGER = context.logger
        is_mpi = context.labels.get("kind", "job") == "mpijob"

        if is_mpi:
            try:
                from mpi4py import MPI

                return context, MPI.COMM_WORLD
            except ModuleNotFoundError as mpi4py_not_found:
                context.logger.error(
                    "To distribute the function using MLRun's 'mpijob' you need to have `mpi4py` package in your "
                    "interpreter. Please run `pip install mpi4py` and make sure you have open-mpi."
                )
                raise mpi4py_not_found
    except ModuleNotFoundError as module_not_found:
        if is_mpi:
            raise module_not_found
    return None, None


def open_mpi_handler(
        worker_inputs: List[str], root_worker_inputs: Dict[str, Any] = None
):
    global _LOGGER

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

            # Give the correct chunk of the workers inputs:
            for worker_input in worker_inputs:
                input_argument = kwargs[worker_input]
                if input_argument is None:
                    continue
                if isinstance(input_argument, str):
                    input_argument = _get_text_files(
                        data_path=pathlib.Path(input_argument).absolute()
                    )
                if len(input_argument) < size:
                    raise ValueError(
                        f"Cannot split the input '{worker_input}' of length {len(input_argument)} to {size} workers. "
                        f"Please reduce the amount of workers for this input."
                    )
                even_chunk_size = len(input_argument) // size
                chunk_start = rank * even_chunk_size
                chunk_end = (
                    (rank + 1) * even_chunk_size
                    if rank + 1 < size
                    else len(input_argument)
                )
                context.logger.info(
                    f"Rank #{rank}: Processing input chunk of '{worker_input}' "
                    f"from index {chunk_start} to {chunk_end}."
                )
                if isinstance(input_argument, list):
                    input_argument = input_argument[chunk_start:chunk_end]
                elif isinstance(input_argument, pd.DataFrame):
                    input_argument = input_argument.iloc[chunk_start:chunk_end:, :]
                kwargs[worker_input] = input_argument

            # Set the root worker only arguments:
            if rank == 0 and root_worker_inputs:
                kwargs.update(root_worker_inputs)

            # Run the worker:
            output = handler(**kwargs)

            # Send the output to the root rank (rank #0):
            output = comm.gather(output, root=0)
            if rank == 0:
                # Join the outputs:
                context.logger.info("Collecting data from workers to root worker.")
                dataframe = pd.concat(objs=[df for df, _ in output], axis=0)
                errors_dictionary = reduce(operator.ior, [err for _, err in output], {})
                return dataframe, errors_dictionary
            return None

        return wrapper

    return decorator


@open_mpi_handler(worker_inputs=["data_path"], root_worker_inputs={"verbose": True})

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
        response = self.tokenizer_bench_mark.decode(
            response_ids, skip_special_tokens=True
        )

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
            return "No matches found after '[Output]:' marker"


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
        return super().compute_over_one_data(question, response)
