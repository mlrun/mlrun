# Copyright 2018 Iguazio
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
from abc import ABC, abstractmethod
from typing import Tuple, Union

from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split

import mlrun
from mlrun.frameworks._ml_common.utils import AlgorithmFunctionality, MLTypes


class MLFunctions(ABC):
    @staticmethod
    @abstractmethod
    def train(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_name: str = None
    ):
        pass

    @staticmethod
    @abstractmethod
    def evaluate(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_path: str
    ):
        pass

    @staticmethod
    @abstractmethod
    def get_model(algorithm_functionality: AlgorithmFunctionality) -> MLTypes.ModelType:
        pass

    @staticmethod
    def get_dataset(
        algorithm_functionality: AlgorithmFunctionality, for_training: bool
    ) -> Union[
        Tuple[MLTypes.DatasetType, MLTypes.DatasetType],
        Tuple[
            MLTypes.DatasetType,
            MLTypes.DatasetType,
            MLTypes.DatasetType,
            MLTypes.DatasetType,
        ],
    ]:
        if algorithm_functionality.is_regression():
            n_targets = 1 if algorithm_functionality.is_single_output() else 5
            x, y = make_regression(n_targets=n_targets)
            stratify = None
        else:
            n_classes = 2 if algorithm_functionality.is_binary_classification() else 5
            if algorithm_functionality.is_single_output():
                x, y = make_classification(n_classes=n_classes, n_informative=n_classes)
                stratify = y
            else:
                x, y = make_multilabel_classification(n_classes=n_classes)
                stratify = None

        if not for_training:
            return x, y
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=stratify
        )
        return x_train, x_test, y_train, y_test
