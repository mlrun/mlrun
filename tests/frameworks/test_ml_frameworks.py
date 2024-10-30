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
import json

import pytest

import mlrun
from mlrun.frameworks._ml_common import AlgorithmFunctionality, MLPlanStages
from mlrun.frameworks.sklearn import MetricsLibrary, SKLearnArtifactsLibrary
from mlrun.frameworks.xgboost import XGBoostArtifactsLibrary

from .sklearn import SKLearnFunctions
from .xgboost import XGBoostFunctions


class FrameworkKeys:
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"


FRAMEWORKS = {
    FrameworkKeys.XGBOOST: (
        XGBoostFunctions,
        XGBoostArtifactsLibrary,
        MetricsLibrary,
    ),
    FrameworkKeys.SKLEARN: (
        SKLearnFunctions,
        SKLearnArtifactsLibrary,
        MetricsLibrary,
    ),
}  # type: Dict[str, Tuple[MLFunctions, ArtifactsLibrary, MetricsLibrary]]
FRAMEWORKS_KEYS = [
    FrameworkKeys.XGBOOST,
    FrameworkKeys.SKLEARN,
]  # type: List[str]
ALGORITHM_FUNCTIONALITIES = [
    algorithm_functionality.value
    for algorithm_functionality in AlgorithmFunctionality
    if "Unknown" not in algorithm_functionality.value
]  # type: List[str]
FRAMEWORKS_ALGORITHM_FUNCTIONALITIES = [
    (framework, algorithm_functionality)
    for framework in FRAMEWORKS_KEYS
    for algorithm_functionality in ALGORITHM_FUNCTIONALITIES
    if (
        framework is not FrameworkKeys.XGBOOST
        or algorithm_functionality
        != AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION.value
    )
]  # type: List[Tuple[str, str]]


def framework_algorithm_functionality_pair_ids(
    framework_algorithm_functionality_pair: tuple[str, str],
) -> str:
    framework, algorithm_functionality = framework_algorithm_functionality_pair
    return f"{framework}-{algorithm_functionality}"


@pytest.mark.parametrize(
    "framework_algorithm_functionality_pair",
    FRAMEWORKS_ALGORITHM_FUNCTIONALITIES,
    ids=framework_algorithm_functionality_pair_ids,
)
def test_training(rundb_mock, framework_algorithm_functionality_pair: tuple[str, str]):
    framework, algorithm_functionality = framework_algorithm_functionality_pair
    # Unpack the framework classes:
    (functions, artifacts_library, metrics_library) = FRAMEWORKS[framework]  # type: MLFunctions, ArtifactsLibrary, MetricsLibrary

    # Run training:
    train_run = mlrun.new_function().run(
        artifact_path="./temp",
        handler=functions.train,
        params={"algorithm_functionality": algorithm_functionality},
    )

    # Print the outputs for manual validation:
    print(json.dumps(train_run.outputs, indent=4))

    # Get assertion parameters:
    algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
    dummy_model = functions.get_model(algorithm_functionality=algorithm_functionality)
    _, dummy_y = functions.get_dataset(
        algorithm_functionality=algorithm_functionality, for_training=False
    )
    expected_artifacts = artifacts_library.get_plans(model=dummy_model, y=dummy_y)
    expected_results = metrics_library.get_metrics(model=dummy_model, y=dummy_y)

    # Validate artifacts (model artifact shouldn't be counted, hence the '-1'):
    assert len(train_run.status.artifacts) - 1 == len(expected_artifacts)

    # Validate results:
    assert len(train_run.status.results) == len(expected_results)


@pytest.mark.parametrize(
    "framework_algorithm_functionality_pair",
    FRAMEWORKS_ALGORITHM_FUNCTIONALITIES,
    ids=framework_algorithm_functionality_pair_ids,
)
def test_evaluation(
    rundb_mock,
    framework_algorithm_functionality_pair: tuple[str, str],
):
    framework, algorithm_functionality = framework_algorithm_functionality_pair
    # Unpack the framework classes:
    (functions, artifacts_library, metrics_library) = FRAMEWORKS[framework]  # type: MLFunctions, ArtifactsLibrary, MetricsLibrary

    # Run training:
    train_run = mlrun.new_function().run(
        artifact_path="./temp2",
        handler=functions.train,
        params={"algorithm_functionality": algorithm_functionality},
    )

    # Run evaluation (on the model that was just trained):
    evaluate_run = mlrun.new_function().run(
        artifact_path="./temp2",
        handler=functions.evaluate,
        params={
            "algorithm_functionality": algorithm_functionality,
            "model_path": train_run.outputs["model"],
        },
    )

    # Print the outputs for manual validation:
    print(json.dumps(evaluate_run.outputs, indent=4))

    # Get assertion parameters:
    algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
    dummy_model = functions.get_model(algorithm_functionality=algorithm_functionality)
    _, dummy_y = functions.get_dataset(
        algorithm_functionality=algorithm_functionality, for_training=False
    )
    expected_artifacts = [
        plan
        for plan in artifacts_library.get_plans(model=dummy_model, y=dummy_y)
        if not (
            # Count only pre and post prediction artifacts (evaluation artifacts).
            plan.is_ready(stage=MLPlanStages.POST_FIT, is_probabilities=False)
            or plan.is_ready(stage=MLPlanStages.PRE_FIT, is_probabilities=False)
        )
    ]
    expected_results = metrics_library.get_metrics(model=dummy_model, y=dummy_y)

    # Validate artifacts:
    assert len(evaluate_run.status.artifacts) == len(expected_artifacts)

    # Validate results:
    assert len(evaluate_run.status.results) == len(expected_results)
