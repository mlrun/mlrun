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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.svm import SVC, LinearSVC

import mlrun
from mlrun.frameworks._ml_common import AlgorithmFunctionality
from mlrun.frameworks.sklearn import SKLearnTypes, apply_mlrun

from ..ml_functions import MLFunctions


class SKLearnFunctions(MLFunctions):
    @staticmethod
    def train(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_name: str = None
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        model = SKLearnFunctions.get_model(
            algorithm_functionality=algorithm_functionality
        )
        x_train, x_test, y_train, y_test = SKLearnFunctions.get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=True
        )

        apply_mlrun(model=model, model_name=model_name, x_test=x_test, y_test=y_test)
        model.fit(x_train, y_train)

    @staticmethod
    def evaluate(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_path: str
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        x, y = SKLearnFunctions.get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=False
        )
        model_handler = apply_mlrun(model_path=model_path, y_test=y)
        model = model_handler.model
        model.predict(x)

    @staticmethod
    def get_model(
        algorithm_functionality: AlgorithmFunctionality,
    ) -> SKLearnTypes.ModelType:
        if algorithm_functionality == AlgorithmFunctionality.BINARY_CLASSIFICATION:
            return RandomForestClassifier()
        if algorithm_functionality == AlgorithmFunctionality.MULTICLASS_CLASSIFICATION:
            return OneVsRestClassifier(LinearSVC())
        if (
            algorithm_functionality
            == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
        ):
            return MultiOutputClassifier(LogisticRegression())
        if (
            algorithm_functionality
            == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION
        ):
            return MultiOutputClassifier(OneVsRestClassifier(SVC()))
        if algorithm_functionality == AlgorithmFunctionality.REGRESSION:
            return SGDRegressor()
        if algorithm_functionality == AlgorithmFunctionality.MULTI_OUTPUT_REGRESSION:
            return MultiOutputRegressor(Ridge())
        return None
