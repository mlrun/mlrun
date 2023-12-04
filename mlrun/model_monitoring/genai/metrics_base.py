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
import uuid
from typing import Union, List, Optional, Dict, Any
from mlrun.model import ModelObj

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


class LLMBaseMetric(ModelObj):
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
        return self.metric.compute(predictions, references, **kwargs)


class LLMJudgeMetric(ModelObj):
    _dict_fields = ["name", "model", "model_config", "prompt_template"]
    kind = "llm_judge_metric"

    def __init__(
        self, name: str, model: str, model_config: Dict[str:str], prompt_template: str
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_config = model_config
        self.prompt_template = prompt_template

    def compute_over_data(self, **kwargs) -> Dict[str, Any]:
        pass


class LLMJudgeSingleGrading(LLMJudgeMetric):
    _dict_fields = ["name", "model_judge", "prompt_template", "grading_examples"]
    kind = "llm_judge_single_grading"

    def __init__(
        self, name: str, model: str, model_config: Dict[str:str], prompt_template: str
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_config = model_config
        self.prompt_template = prompt_template
        self.grading_examples = grading_examples

    def compute_over_data(self, **kwargs) -> Dict[str, Any]:
        pass


class LLMJudgePairwiseGrading(LLMJudgeMetric):
    _dict_fields = [
        "name",
        "model_judge",
        "bench_mark_model",
        "bench_mark_model_config",
        "prompt_template",
        "grading_examples",
    ]
    kind = "llm_judge_pairwise_grading"

    def __init__(
        self, name: str, model: str, model_config: Dict[str:str], prompt_template: str
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_config = model_config
        self.prompt_template = prompt_template
        self.grading_examples = grading_examples


class LLMJudgeReferenceGrading(LLMJudgeMetric):
    _dict_fields = [
        "name",
        "model_judge",
        "bench_mark_model",
        "bench_mark_model_config",
        "prompt_template",
        "reference",
    ]
    kind = "llm_judge_reference_grading"

    def __init__(
        self, name: str, model: str, model_config: Dict[str:str], prompt_template: str
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_config = model_config
        self.prompt_template = prompt_template
        self.reference = reference


# TODO add pairwise metrics, sinlge answer grading, reference-guided grading
# MLflow chose the simplest single answer grading.
