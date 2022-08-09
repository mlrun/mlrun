import json
from typing import List

import pytest

import mlrun
from mlrun.frameworks._ml_common import AlgorithmFunctionality, MLPlanStages
from mlrun.frameworks.lgbm import LGBMArtifactsLibrary
from mlrun.frameworks.sklearn import MetricsLibrary

from ..ml_common import get_dataset
from .functions import LightGBMHandlers, get_model

ALGORITHM_FUNCTIONALITIES = [  # type: List[str]
    AlgorithmFunctionality.REGRESSION.value,
    AlgorithmFunctionality.BINARY_CLASSIFICATION.value,
    AlgorithmFunctionality.MULTICLASS_CLASSIFICATION.value,
    # Currently, LightGBM do not support multi-output functionalities.
]


@pytest.mark.parametrize("algorithm_functionality", ALGORITHM_FUNCTIONALITIES)
def test_training_api_training(algorithm_functionality: str):
    # Run training:
    train_run = mlrun.new_function().run(
        artifact_path="./temp",
        handler=LightGBMHandlers.training_api_train,
        params={"algorithm_functionality": algorithm_functionality},
    )

    # Print the outputs for manual validation:
    print(json.dumps(train_run.outputs, indent=4))

    # Validate artifacts (model artifact shouldn't be counted, hence the '-1'):
    assert len(train_run.status.artifacts) - 1 > 0

    # Validate results (context parameters shouldn't be counted, hence the '-1'):
    assert len(train_run.status.results) - 1 > 0


@pytest.mark.parametrize("algorithm_functionality", ALGORITHM_FUNCTIONALITIES)
def test_sklearn_api_training(algorithm_functionality: str):
    # Run training:
    train_run = mlrun.new_function().run(
        artifact_path="./temp",
        handler=LightGBMHandlers.sklearn_api_train,
        params={"algorithm_functionality": algorithm_functionality},
    )

    # Print the outputs for manual validation:
    print(json.dumps(train_run.outputs, indent=4))

    # Get assertion parameters:
    algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
    dummy_model = get_model(algorithm_functionality=algorithm_functionality)
    _, dummy_y = get_dataset(
        algorithm_functionality=algorithm_functionality, for_training=False
    )
    expected_artifacts = LGBMArtifactsLibrary.get_plans(model=dummy_model, y=dummy_y)
    expected_results = MetricsLibrary.get_metrics(model=dummy_model, y=dummy_y)

    # Validate artifacts (model artifact shouldn't be counted, hence the '-1'):
    assert len(train_run.status.artifacts) - 1 == len(expected_artifacts)

    # Validate results:
    assert len(train_run.status.results) == len(expected_results)


@pytest.mark.parametrize("algorithm_functionality", ALGORITHM_FUNCTIONALITIES)
def test_sklearn_api_evaluation(algorithm_functionality: str):
    # Run training:
    train_run = mlrun.new_function().run(
        artifact_path="./temp2",
        handler=LightGBMHandlers.sklearn_api_train,
        params={"algorithm_functionality": algorithm_functionality},
    )

    # Run evaluation (on the model that was just trained):
    evaluate_run = mlrun.new_function().run(
        artifact_path="./temp2",
        handler=LightGBMHandlers.sklearn_api_evaluate,
        params={
            "algorithm_functionality": algorithm_functionality,
            "model_path": train_run.outputs["model"],
        },
    )

    # Print the outputs for manual validation:
    print(json.dumps(evaluate_run.outputs, indent=4))

    # Get assertion parameters:
    algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
    dummy_model = get_model(algorithm_functionality=algorithm_functionality)
    _, dummy_y = get_dataset(
        algorithm_functionality=algorithm_functionality, for_training=False
    )
    expected_artifacts = (
        [  # Count only pre and post prediction artifacts (evaluation artifacts).
            plan
            for plan in LGBMArtifactsLibrary.get_plans(model=dummy_model, y=dummy_y)
            if not (
                plan.is_ready(stage=MLPlanStages.POST_FIT, is_probabilities=False)
                or plan.is_ready(stage=MLPlanStages.PRE_FIT, is_probabilities=False)
            )
        ]
    )
    expected_results = MetricsLibrary.get_metrics(model=dummy_model, y=dummy_y)

    # Validate artifacts:
    assert len(evaluate_run.status.artifacts) == len(expected_artifacts)

    # Validate results:
    assert len(evaluate_run.status.results) == len(expected_results)
