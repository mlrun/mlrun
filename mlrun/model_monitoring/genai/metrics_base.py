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
from typing import Union, List, Optional, Dict, Any, ClassVar

from mlrun.model import ModelObj
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


class LLMJudgeSingleGrading(ModelObj):
    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_single_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, str],
        prompt_template: str,
        prompt_config: Dict[str, str],
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_judge = model_judge
        self.model_judge_config = model_judge_config
        self.prompt_template = prompt_template
        self.prompt_config = prompt_config

    def fill_prompt(self) -> str:
        """
        Fill the prompt template with the prompt config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
        :return: the filled prompt
        """
        prompt = self.prompt_template
        for key, value in self.prompt_config.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt

    def prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_judge)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.model_judge)

        return tokenizer, model

    def compute_over_one_data(self, model, tokenizer) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :return: the metrics score and the explanation
        """
        input_ids = tokenizer(self.fill_prompt(), return_tensors="pt").input_ids
        outputs = model.generate(input_ids, **self.model_judge_config)

        response_ids = outputs[0]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        return response


class LLMJudgePairwiseGrading(ModelObj):
    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "bench_mark_model",
        "bench_mark_model_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_pairwise_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, str],
        prompt_template: str,
        bench_mark_model: str,
        bench_mark_model_config: Dict[str, str],
        prompt_config: Dict[str, str],
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_judge = model_judge
        self.model_judge_config = model_config
        self.bench_mark_model = bench_mark_model
        self.bench_mark_model_config = bench_mark_model_config
        self.prompt_template = prompt_template
        self.prompt_config = prompt_config


class LLMJudgeReferenceGrading(ModelObj):
    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "bench_mark_model",
        "bench_mark_model_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_reference_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, str],
        prompt_template: str,
        bench_mark_model: str,
        bench_mark_model_config: Dict[str, str],
        prompt_config: Dict[str, str],
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_judge = model_judge
        self.model_judge_config = model_config
        self.bench_mark_model = bench_mark_model
        self.bench_mark_model_config = bench_mark_model_config
        self.prompt_template = prompt_template
        self.prompt_config = prompt_config


# TODO figure out a way to viz the different metrics in a Radar plot
