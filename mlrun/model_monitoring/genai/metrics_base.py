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


class BaseMetric(object):
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
        self.name = name
        self.metric = evaluate.load(name)

    def __call__(
        self, predictions: Union[List, Dict], references: Union[List, Dict], **kwargs
    ) -> Dict[str, Any]:
        return self.metric.compute(predictions, references, **kwargs)


class LlmMetric(object):
    def __init__(
        self, name: str, model_config: Dict[str, Any], data_config: Dict[str, Any]
    ):
        """
        Base class for LLM as a judge metrics.
        These metrics are used for more open-ended question for the model
        and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
        """
        self.name = name
        self.model_config = model_config
        self.data_config = data_config

    def __call__(self, **kwargs) -> Dict[str, Any]:
        pass


# TODO add pairwise metrics, sinlge answer grading, reference-guided grading
# MLflow chose the simplest single answer grading.
