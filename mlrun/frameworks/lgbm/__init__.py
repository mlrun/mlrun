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

from typing import Any, Optional, Union

import lightgbm as lgb

import mlrun

from .._ml_common import MLArtifactsLibrary, MLPlan
from ..sklearn import Metric, MetricsLibrary
from .mlrun_interfaces import (
    LGBMBoosterMLRunInterface,
    LGBMMLRunInterface,
    LGBMModelMLRunInterface,
)
from .model_handler import LGBMModelHandler
from .model_server import LGBMModelServer
from .utils import LGBMTypes, LGBMUtils

# Placeholders as the SciKit-Learn API is commonly used among all ML frameworks:
LGBMArtifactsLibrary = MLArtifactsLibrary


def _apply_mlrun_on_module(
    model_name: str = "model",
    tag: str = "",
    modules_map: Optional[Union[dict[str, Union[None, str, list[str]]], str]] = None,
    custom_objects_map: Optional[Union[dict[str, Union[str, list[str]]], str]] = None,
    custom_objects_directory: Optional[str] = None,
    context: mlrun.MLClientCtx = None,
    model_format: str = LGBMModelHandler.ModelFormats.PKL,
    sample_set: Union[LGBMTypes.DatasetType, mlrun.DataItem, str] = None,
    y_columns: Optional[Union[list[str], list[int]]] = None,
    feature_vector: Optional[str] = None,
    feature_weights: Optional[list[float]] = None,
    labels: Optional[dict[str, Union[str, int, float]]] = None,
    parameters: Optional[dict[str, Union[str, int, float]]] = None,
    extra_data: Optional[dict[str, LGBMTypes.ExtraDataType]] = None,
    auto_log: bool = True,
    mlrun_logging_callback_kwargs: Optional[dict[str, Any]] = None,
):
    # Apply MLRun's interface on the LightGBM module:
    LGBMMLRunInterface.add_interface(obj=lgb)

    # If automatic logging is required, set the future logging callbacks that will be applied to the training functions:
    if auto_log:
        lgb.configure_logging(
            context=context,
            model_handler_kwargs={
                "model_name": model_name,
                "modules_map": modules_map,
                "custom_objects_map": custom_objects_map,
                "custom_objects_directory": custom_objects_directory,
                "model_format": model_format,
            },
            log_model_kwargs={
                "tag": tag,
                "sample_set": sample_set,
                "target_columns": y_columns,
                "feature_vector": feature_vector,
                "feature_weights": feature_weights,
                "labels": labels,
                "parameters": parameters,
                "extra_data": extra_data,
            },
            mlrun_logging_callback_kwargs=mlrun_logging_callback_kwargs,
        )


def _apply_mlrun_on_model(
    model: LGBMTypes.ModelType = None,
    model_name: str = "model",
    tag: str = "",
    model_path: Optional[str] = None,
    modules_map: Optional[Union[dict[str, Union[None, str, list[str]]], str]] = None,
    custom_objects_map: Optional[Union[dict[str, Union[str, list[str]]], str]] = None,
    custom_objects_directory: Optional[str] = None,
    context: mlrun.MLClientCtx = None,
    model_format: str = LGBMModelHandler.ModelFormats.PKL,
    artifacts: Optional[Union[list[MLPlan], list[str], dict[str, dict]]] = None,
    metrics: Optional[
        Union[
            list[Metric],
            list[LGBMTypes.MetricEntryType],
            dict[str, LGBMTypes.MetricEntryType],
        ]
    ] = None,
    x_test: LGBMTypes.DatasetType = None,
    y_test: LGBMTypes.DatasetType = None,
    sample_set: Union[LGBMTypes.DatasetType, mlrun.DataItem, str] = None,
    y_columns: Optional[Union[list[str], list[int]]] = None,
    feature_vector: Optional[str] = None,
    feature_weights: Optional[list[float]] = None,
    labels: Optional[dict[str, Union[str, int, float]]] = None,
    parameters: Optional[dict[str, Union[str, int, float]]] = None,
    extra_data: Optional[dict[str, LGBMTypes.ExtraDataType]] = None,
    auto_log: bool = True,
    **kwargs,
):
    # Create a model handler:
    model_handler_kwargs = (
        kwargs.pop("model_handler_kwargs") if "model_handler_kwargs" in kwargs else {}
    )
    handler = LGBMModelHandler(
        model_name=model_name,
        model_path=model_path,
        model=model,
        context=context,
        model_format=model_format,
        modules_map=modules_map,
        custom_objects_map=custom_objects_map,
        custom_objects_directory=custom_objects_directory,
        **model_handler_kwargs,
    )

    # Set the handler's logging attributes:
    handler.set_tag(tag=tag)
    if sample_set is not None:
        handler.set_sample_set(sample_set=sample_set)
    if y_columns is not None:
        handler.set_target_columns(target_columns=y_columns)
    if feature_vector is not None:
        handler.set_feature_vector(feature_vector=feature_vector)
    if feature_weights is not None:
        handler.set_feature_weights(feature_weights=feature_weights)
    if labels is not None:
        handler.set_labels(to_add=labels)
    if parameters is not None:
        handler.set_parameters(to_add=parameters)
    if extra_data is not None:
        handler.set_extra_data(to_add=extra_data)

    # Load the model if it was not provided:
    if handler.model is None:
        handler.load()
    model = handler.model

    # Add MLRun's interface to the model according to the model type (LGBMModel or Booster):
    if isinstance(model, lgb.LGBMModel):
        # Apply MLRun's interface on the `LGBMModel`:
        LGBMModelMLRunInterface.add_interface(obj=model)
        # Configure the logger:
        model.configure_logging(
            context=context,
            plans=LGBMArtifactsLibrary.get_plans(
                artifacts=artifacts,
                context=context,
                include_default=auto_log,
                model=model,
                y=y_test,
            ),
            metrics=MetricsLibrary.get_metrics(
                metrics=metrics,
                context=context,
                include_default=auto_log,
                model=model,
                y=y_test,
            ),
            x_test=x_test,
            y_test=y_test,
            model_handler=handler,
        )
    else:  # lgb.Booster
        LGBMBoosterMLRunInterface.add_interface(obj=model)
        model.model_handler = handler

    return handler


def apply_mlrun(
    model: LGBMTypes.ModelType = None,
    model_name: str = "model",
    tag: str = "",
    model_path: Optional[str] = None,
    modules_map: Optional[Union[dict[str, Union[None, str, list[str]]], str]] = None,
    custom_objects_map: Optional[Union[dict[str, Union[str, list[str]]], str]] = None,
    custom_objects_directory: Optional[str] = None,
    context: mlrun.MLClientCtx = None,
    model_format: str = LGBMModelHandler.ModelFormats.PKL,
    artifacts: Optional[Union[list[MLPlan], list[str], dict[str, dict]]] = None,
    metrics: Optional[
        Union[
            list[Metric],
            list[LGBMTypes.MetricEntryType],
            dict[str, LGBMTypes.MetricEntryType],
        ]
    ] = None,
    x_test: LGBMTypes.DatasetType = None,
    y_test: LGBMTypes.DatasetType = None,
    sample_set: Union[LGBMTypes.DatasetType, mlrun.DataItem, str] = None,
    y_columns: Optional[Union[list[str], list[int]]] = None,
    feature_vector: Optional[str] = None,
    feature_weights: Optional[list[float]] = None,
    labels: Optional[dict[str, Union[str, int, float]]] = None,
    parameters: Optional[dict[str, Union[str, int, float]]] = None,
    extra_data: Optional[dict[str, LGBMTypes.ExtraDataType]] = None,
    auto_log: bool = True,
    mlrun_logging_callback_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Union[LGBMModelHandler, None]:
    """
    Apply MLRun's interface on top of LightGBM by wrapping the module itself or the given model, providing both with
    MLRun's quality of life features.

    :param model:                    The model to wrap. Can be loaded from the model path given as well.
    :param model_name:               The model name to use for storing the model artifact. Default: "model".
    :param tag:                      The model's tag to log with.
    :param model_path:               The model's store object path. Mandatory for evaluation (to know which model to
                                     update). If model is not provided, it will be loaded from this path.
    :param modules_map:              A dictionary of all the modules required for loading the model. Each key is a
                                     path to a module and its value is the object name to import from it. All the
                                     modules will be imported globally. If multiple objects needed to be imported
                                     from the same module a list can be given. The map can be passed as a path to a
                                     json file as well. For example:

                                     .. code-block:: python

                                         {
                                             "module1": None,  # import module1
                                             "module2": ["func1", "func2"],  # from module2 import func1, func2
                                             "module3.sub_module": "func3",  # from module3.sub_module import func3
                                         }

                                     If the model path given is of a store object, the modules map will be read from
                                     the logged modules map artifact of the model.
    :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key is
                                     a path to a python file and its value is the custom object name to import from it.
                                     If multiple objects needed to be imported from the same py file a list can be
                                     given. The map can be passed as a path to a json file as well. For example:

                                     .. code-block:: python

                                         {
                                             "/.../custom_model.py": "MyModel",
                                             "/.../custom_objects.py": ["object1", "object2"],
                                         }

                                     All the paths will be accessed from the given 'custom_objects_directory', meaning
                                     each py file will be read from 'custom_objects_directory/<MAP VALUE>'. If the model
                                     path given is of a store object, the custom objects map will be read from the
                                     logged custom object map artifact of the model.
                                     Notice: The custom objects will be imported in the order they came in this
                                     dictionary (or json). If a custom object is depended on another, make sure to
                                     put it below the one it relies on.
    :param custom_objects_directory: Path to the directory with all the python files required for the custom objects.
                                     Can be passed as a zip file as well (will be extracted during the run before
                                     loading the model). If the model path given is of a store object, the custom
                                     objects files will be read from the logged custom object artifact of the model.
    :param context:                  MLRun context to work with. If no context is given it will be retrieved via
                                     'mlrun.get_or_create_ctx(None)'
    :param artifacts:                A list of artifacts plans to produce during the run.
    :param metrics:                  A list of metrics to calculate during the run.
    :param x_test:                   The validation data for producing and calculating artifacts and metrics post
                                     training. Without this, validation will not be performed.
    :param y_test:                   The test data ground truth for producing and calculating artifacts and metrics post
                                     training or post predict / predict_proba.
    :param sample_set:               A sample set of inputs for the model for logging its stats along the model in
                                     favour of model monitoring.
    :param y_columns:                List of names of all the columns in the ground truth labels in case its a
                                     pd.DataFrame or a list of integers in case the dataset is a np.ndarray. If not
                                     given but 'y_train' / 'y_test' is given then the labels / indices in it will be
                                     used by default.
    :param feature_vector:           Feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
    :param feature_weights:          List of feature weights, one per input column.
    :param labels:                   Labels to log with the model.
    :param parameters:               Parameters to log with the model.
    :param extra_data:               Extra data to log with the model.
    :param auto_log:                 Whether to apply MLRun's auto logging on the model. Auto logging will add the
                                     default artifacts and metrics to the lists of artifacts and metrics. Default:
                                     True.
    :param mlrun_logging_callback_kwargs: Key word arguments for the MLRun callback. For further information see the
                                     documentation of the class 'MLRunLoggingCallback'. Note that 'context' is already
                                     given here.


    :return: If a model was provided via `model` or `model_path` the model handler initialized with the provided model
             will be returned. Otherwise, None.
    """
    # Get the default context:
    if context is None:
        context = mlrun.get_or_create_ctx(LGBMMLRunInterface.DEFAULT_CONTEXT_NAME)

    # If a model or a model path were provided, apply on the provided model, otherwise on the LightGBM module:
    if model is None and model_path is None:
        _apply_mlrun_on_module(
            model_name=model_name,
            tag=tag,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
            model_format=model_format,
            sample_set=sample_set,
            y_columns=y_columns,
            feature_vector=feature_vector,
            feature_weights=feature_weights,
            labels=labels,
            parameters=parameters,
            extra_data=extra_data,
            auto_log=auto_log,
            mlrun_logging_callback_kwargs=mlrun_logging_callback_kwargs,
        )
        return
    return _apply_mlrun_on_model(
        model=model,
        model_name=model_name,
        tag=tag,
        model_path=model_path,
        modules_map=modules_map,
        custom_objects_map=custom_objects_map,
        custom_objects_directory=custom_objects_directory,
        context=context,
        model_format=model_format,
        artifacts=artifacts,
        metrics=metrics,
        x_test=x_test,
        y_test=y_test,
        sample_set=sample_set,
        y_columns=y_columns,
        feature_vector=feature_vector,
        feature_weights=feature_weights,
        labels=labels,
        parameters=parameters,
        extra_data=extra_data,
        auto_log=auto_log,
        **kwargs,
    )
