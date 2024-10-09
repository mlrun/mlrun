import os
import tempfile
from typing import Tuple

import sklearn
import xgboost
from sklearn.base import BaseEstimator

# import mlrun.frameworks.sklearn as sklearn_imp
from mlrun.frameworks.auto_mlrun import AutoMLRun
from mlrun.package.packagers.sklearn_packager import SklearnModelPack

from sklearn.linear_model import LinearRegression
from mlrun.package.utils import Pickler
from tests.package.packager_tester import (
    COMMON_OBJECT_INSTRUCTIONS,
    PackagerTester,
    PackTest,
    PackToUnpackTest,
    UnpackTest,
)

_BASE_MODEL_SAMPLE = BaseEstimator()
_LINEAR_MODEL_SAMPLE = LinearRegression()
_XGBOOST_MODEL_SAMPLE = xgboost.XGBClassifier()

_BASE_PIPELINE_SAMPLE = sklearn.pipeline.Pipeline(
    steps=[("model", _BASE_MODEL_SAMPLE)]
)
_LINEAR_PIPELINE_SAMPLE = sklearn.pipeline.Pipeline(
    steps=[("model", _LINEAR_MODEL_SAMPLE)]
)
_XGBOOST_PIPELINE_SAMPLE = sklearn.pipeline.Pipeline(
    steps=[("model", _XGBOOST_MODEL_SAMPLE)]
)


def pack_model(do_apply_mlrun=False) -> BaseEstimator:
    model = _BASE_MODEL_SAMPLE

    # apply_mlrun changes 'fit' method which doesn't exist in BaseEstimator
    if do_apply_mlrun:
        model = _LINEAR_MODEL_SAMPLE
        AutoMLRun.apply_mlrun(model)
    return model


def pack_pipeline(do_apply_mlrun=False, do_apply_mlrun_steps=False) -> sklearn.pipeline.Pipeline:
    pipeline = _BASE_PIPELINE_SAMPLE

    if do_apply_mlrun or do_apply_mlrun_steps:
        pipeline = _LINEAR_PIPELINE_SAMPLE  # apply_mlrun changes 'fit' method which doesn't exist in BaseEstimator

    # Does apply_mlrun to all the steps inside the Pipeline
    if do_apply_mlrun_steps:
        for step in pipeline.steps:
            # Uses AutoMLRun to get the matching 'apply_mlrun' function
            AutoMLRun.apply_mlrun(step[1])

    # Apply mlrun to pipeline
    if do_apply_mlrun:
        AutoMLRun.apply_mlrun(pipeline)
    return pipeline


def unpack_model(obj: sklearn.base.BaseEstimator):
    assert isinstance(obj, sklearn.base.BaseEstimator)


def unpack_pipeline(obj: sklearn.pipeline.Pipeline):
    assert isinstance(obj, sklearn.pipeline.Pipeline)


def validate_model_packing(model: dict) -> bool:
    return (
            isinstance(model, dict)
            and model["kind"] == "model"
            and model["metadata"]["key"] == "my_model"
            and model["spec"]["unpackaging_instructions"]["packager_name"]
            == "SklearnModelPack"
    )


def validate_pipeline_packing(model: dict) -> bool:
    return (
            isinstance(model, dict)
            and model["kind"] == "model"
            and model["metadata"]["key"] == "my_pipeline"
            and model["spec"]["unpackaging_instructions"]["packager_name"]
            == "SklearnModelPack"
    )


def prepare_model_file(
        file_format: str, obj: sklearn.base.BaseEstimator
) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    Pickler.pickle(obj=obj, pickle_module_name="cloudpickle", output_path=file_path)
    return file_path, temp_directory


class SklearnPackagerTester(PackagerTester):
    """
    A tester for the `DictPackager`.
    """

    PACKAGER_IN_TEST = SklearnModelPack

    TESTS = [
        # Model
        PackTest(
            pack_handler="pack_model",
            log_hint={"key": "my_model", "artifact_type": "model"},
            validation_function=validate_model_packing,
            pack_parameters={"do_apply_mlrun": False},
        ),
        PackTest(
            pack_handler="pack_model",
            log_hint={"key": "my_model", "artifact_type": "model"},
            validation_function=validate_model_packing,
            pack_parameters={"do_apply_mlrun": True},
        ),
        UnpackTest(
            prepare_input_function=prepare_model_file,
            unpack_handler="unpack_model",
            prepare_parameters={"file_format": "json", "obj": _BASE_MODEL_SAMPLE},
        ),
        PackToUnpackTest(
            pack_handler="pack_model",
            pack_parameters={"do_apply_mlrun": False},
            log_hint={"key": "my_model", "artifact_type": "model"},
            expected_instructions={
                "object_module_name": "sklearn",
                "pickle_module_name": "cloudpickle",
                "pickle_module_version": "2.2.1",
                "python_version": "3.9.16",
            },
            unpack_handler="unpack_model",
        ),
        PackToUnpackTest(
            pack_handler="pack_model",
            pack_parameters={"do_apply_mlrun": True},
            log_hint={"key": "my_model", "artifact_type": "model"},
            expected_instructions={
                "object_module_name": "sklearn",
                "pickle_module_name": "cloudpickle",
                "pickle_module_version": "2.2.1",
                "python_version": "3.9.16",
            },
            unpack_handler="unpack_model",
        ),
        # TODO: not sure what is this test and how different is it from the previous 2
        PackToUnpackTest(
            pack_handler="pack_model",
            log_hint="my_model: model",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": "sklearn",
            },
            unpack_handler="unpack_model",
        ),
        # Pipeline
        *[
            PackTest(
                pack_handler="pack_pipeline",
                pack_parameters={
                    "do_apply_mlrun": do_apply_mlrun,
                    "do_apply_mlrun_steps": do_apply_mlrun_steps,
                },
                log_hint={"key": "my_pipeline", "artifact_type": "model"},
                validation_function=validate_pipeline_packing,
            )
            for do_apply_mlrun, do_apply_mlrun_steps in [
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ]
        ],
        UnpackTest(
            prepare_input_function=prepare_model_file,
            unpack_handler="unpack_pipeline",
            prepare_parameters={"file_format": "json", "obj": _BASE_PIPELINE_SAMPLE},
        ),
        UnpackTest(
            prepare_input_function=prepare_model_file,
            unpack_handler="unpack_pipeline",
            prepare_parameters={"file_format": "json", "obj": _XGBOOST_PIPELINE_SAMPLE},
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_pipeline",
                pack_parameters={
                    "do_apply_mlrun": do_apply_mlrun,
                    "do_apply_mlrun_steps": do_apply_mlrun_steps,
                },
                log_hint={"key": "my_model", "artifact_type": "model"},
                expected_instructions={
                    "object_module_name": "sklearn",
                    "pickle_module_name": "cloudpickle",
                    "pickle_module_version": "2.2.1",
                    "python_version": "3.9.16",
                },
                unpack_handler="unpack_model",
            )
            for do_apply_mlrun, do_apply_mlrun_steps in [
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ]
        ],
        # TODO: not sure what is this test and how different is it from the previous one
        PackToUnpackTest(
            pack_handler="pack_pipeline",
            log_hint="my_model: model",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": "sklearn",
            },
            unpack_handler="unpack_model",
        ),
    ]
