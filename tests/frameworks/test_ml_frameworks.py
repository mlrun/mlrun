import json
from typing import Dict, List, Tuple

import pytest

import mlrun
from mlrun.frameworks._common import ArtifactsLibrary
from mlrun.frameworks._ml_common import AlgorithmFunctionality, MLPlanStages
from mlrun.frameworks.sklearn import MetricsLibrary, SKLearnArtifactsLibrary
from mlrun.frameworks.xgboost import XGBoostArtifactsLibrary

from .ml_functions import MLFunctions
from .sklearn import SKLearnFunctions
from .xgboost import XGBoostFunctions


class FrameworkKeys:
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"


FRAMEWORKS = {  # type: Dict[str, Tuple[MLFunctions, ArtifactsLibrary, MetricsLibrary]]
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
}
FRAMEWORKS_KEYS = [  # type: List[str]
    FrameworkKeys.XGBOOST,
    FrameworkKeys.SKLEARN,
]
ALGORITHM_FUNCTIONALITIES = [  # type: List[str]
    algorithm_functionality.value
    for algorithm_functionality in AlgorithmFunctionality
    if "Unknown" not in algorithm_functionality.value
]


@pytest.mark.parametrize("framework", FRAMEWORKS_KEYS)
@pytest.mark.parametrize("algorithm_functionality", ALGORITHM_FUNCTIONALITIES)
def test_training(framework: str, algorithm_functionality: str):
    # Unpack the framework classes:
    (functions, artifacts_library, metrics_library) = FRAMEWORKS[
        framework
    ]  # type: MLFunctions, ArtifactsLibrary, MetricsLibrary

    # Skips:
    if (
        functions is XGBoostFunctions
        and algorithm_functionality
        == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION.value
    ):
        pytest.skip(
            "multiclass multi output classification are not supported in 'xgboost'."
        )

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


@pytest.mark.parametrize("framework", FRAMEWORKS_KEYS)
@pytest.mark.parametrize("algorithm_functionality", ALGORITHM_FUNCTIONALITIES)
def test_evaluation(framework: str, algorithm_functionality: str):
    # Unpack the framework classes:
    (functions, artifacts_library, metrics_library) = FRAMEWORKS[
        framework
    ]  # type: MLFunctions, ArtifactsLibrary, MetricsLibrary

    # Skips:
    if (
        functions is XGBoostFunctions
        and algorithm_functionality
        == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION.value
    ):
        pytest.skip(
            "multiclass multi output classification are not supported in 'xgboost'."
        )

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
        if not (  # Count only pre and post prediction artifacts (evaluation artifacts).
            plan.is_ready(stage=MLPlanStages.POST_FIT, is_probabilities=False)
            or plan.is_ready(stage=MLPlanStages.PRE_FIT, is_probabilities=False)
        )
    ]
    expected_results = metrics_library.get_metrics(model=dummy_model, y=dummy_y)

    # Validate artifacts:
    assert len(evaluate_run.status.artifacts) == len(expected_artifacts)

    # Validate results:
    assert len(evaluate_run.status.results) == len(expected_results)
