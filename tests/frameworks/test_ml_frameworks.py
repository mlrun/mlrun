import json

import pytest

import mlrun
from mlrun.frameworks._ml_common.utils import AlgorithmFunctionality

from .functions import MLFunctions
from .lgbm import LGBMFunctions
from .sklearn import SKLearnFunctions
from .xgboost import XGBoostFunctions

ML_FUNCTIONS = [XGBoostFunctions, LGBMFunctions, SKLearnFunctions]
ALGORITHM_FUNCTIONALITIES = [
    algorithm_functionality.value
    for algorithm_functionality in AlgorithmFunctionality
    if "Unknown" not in algorithm_functionality.value
]


@pytest.mark.parametrize("functions", ML_FUNCTIONS)
@pytest.mark.parametrize("algorithm_functionality", ALGORITHM_FUNCTIONALITIES)
def test_training(functions: MLFunctions, algorithm_functionality: str):
    if (
        (functions is LGBMFunctions or functions is XGBoostFunctions)
        and algorithm_functionality
        == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION.value
    ):
        pytest.skip(
            "May be bug in lightgbm and xgboost for multiclass multi output classification."
        )

    train_run = mlrun.new_function().run(
        artifact_path="./temp",
        handler=functions.train,
        params={"algorithm_functionality": algorithm_functionality},
    )

    print(json.dumps(train_run.outputs, indent=4))

    assert len(train_run.status.artifacts) >= 2
    assert len(train_run.status.results) >= 1


@pytest.mark.parametrize("functions", ML_FUNCTIONS)
@pytest.mark.parametrize("algorithm_functionality", ALGORITHM_FUNCTIONALITIES)
def test_evaluation(functions: MLFunctions, algorithm_functionality: str):
    if (
        (functions is LGBMFunctions or functions is XGBoostFunctions)
        and algorithm_functionality
        == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION.value
    ):
        pytest.skip(
            "May be bug in lightgbm and xgboost for multiclass multi output classification."
        )

    model_name = "train_to_eval"

    train_run = mlrun.new_function().run(
        artifact_path="./temp2",
        handler=functions.train,
        params={
            "algorithm_functionality": algorithm_functionality,
            "model_name": model_name,
        },
    )

    evaluate_run = mlrun.new_function().run(
        artifact_path="./temp2",
        handler=functions.evaluate,
        params={
            "algorithm_functionality": algorithm_functionality,
            "model_path": train_run.outputs[model_name],
        },
    )

    print(json.dumps(evaluate_run.outputs, indent=4))

    assert len(evaluate_run.status.artifacts) >= 1
    assert len(evaluate_run.status.results) >= 1
