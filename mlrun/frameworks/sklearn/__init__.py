from typing import Any, Dict, List, Union

import mlrun

from .._common import get_plans
from .._ml_common import (
    DatasetType,
    MetricEntry,
    MetricsLibrary,
    MLArtifactsLibrary,
    MLPlan,
    PickleModelServer,
    get_metrics,
)
from .mlrun_interface import SKLearnMLRunInterface
from .model_handler import SKLearnModelHandler
from .utils import SKLearnModelType

# Placeholders as the SciKit-Learn API is commonly used among all of the ML frameworks:
SKLearnArtifactsLibrary = MLArtifactsLibrary
SKLearnMetricsLibrary = MetricsLibrary
SklearnModelServer = PickleModelServer


def apply_mlrun(
    model: SKLearnModelType,
    model_name: str = None,
    tag: str = "",
    model_path: str = None,
    modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
    custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
    custom_objects_directory: str = None,
    context: mlrun.MLClientCtx = None,
    artifacts: Union[List[MLPlan], Dict[str, dict]] = None,
    metrics: Union[List[MetricEntry], Dict[str, MetricEntry]] = None,
    x_validation: DatasetType = None,
    y_validation: DatasetType = None,
    y_test: DatasetType = None,
    y_columns: Union[List[str], List[int]] = None,
    auto_log: bool = True,
    model_logging_kwargs: Dict[str, Any] = None,
) -> SKLearnModelHandler:
    """
    Wrap the given model with MLRun's interface providing it with mlrun's additional features.

    :param model:                    The model to wrap.
    :param model_name:               The model name to use for storing the model artifact. If not given, the
                                     tf.keras.Model.name will be used.
    :param tag:                      The model's tag to log with.
    :param model_path:               The model's store object path. Mandatory for evaluation (to know which model to
                                     update).
    :param modules_map:              A dictionary of all the modules required for loading the model. Each key is a
                                     path to a module and its value is the object name to import from it. All the
                                     modules will be imported globally. If multiple objects needed to be imported
                                     from the same module a list can be given. The map can be passed as a path to a
                                     json file as well. For example:
                                     {
                                         "module1": None,  # => import module1
                                         "module2": ["func1", "func2"],  # => from module2 import func1, func2
                                         "module3.sub_module": "func3",  # => from module3.sub_module import func3
                                     }
                                     If the model path given is of a store object, the modules map will be read from
                                     the logged modules map artifact of the model.
    :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key is
                                     a path to a python file and its value is the custom object name to import from it.
                                     If multiple objects needed to be imported from the same py file a list can be
                                     given. The map can be passed as a path to a json file as well. For example:
                                     {
                                         "/.../custom_model.py": "MyModel",
                                         "/.../custom_objects.py": ["object1", "object2"]
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
    :param x_validation:             The validation data for producing and calculating artifacts and metrics post
                                     training. Without this, validation will not be performed.
    :param y_validation:             The validation data ground truths (labels).
    :param y_test:                   The test data for producing and calculating artifacts and metrics post
                                     predict / predict_proba.
    :param y_columns:                List of names of all the columns in the ground truth labels in case its a
                                     pd.DataFrame or a list of integers in case the dataset is a np.ndarray. If
                                     'y_train' / 'y_validation' / 'y_test' is given then the labels / indices in it will
                                     be used by default.
    :param auto_log:                 Whether or not to apply MLRun's auto logging on the model. Auto logging will add
                                     the default artifacts and metrics to the lists of artifacts and metrics. Defaulted
                                     to True.
    :param model_logging_kwargs:     Key word arguments for the MLRun callback. For further information see the
                                     documentation of the MLMLRunInterface 'auto_log' method. Notice some of the
                                     attributes are provided here as well so there is no need to give them.

    :return: The model handler initialized with the provided model.
    """
    # Get the default context:
    if context is None:
        context = mlrun.get_or_create_ctx(SKLearnMLRunInterface.DEFAULT_CONTEXT_NAME)

    # Create a model handler:
    handler = SKLearnModelHandler(
        model_name=model_name,
        model_path=model_path,
        model=model,
        context=context,
        modules_map=modules_map,
        custom_objects_map=custom_objects_map,
        custom_objects_directory=custom_objects_directory,
    )

    # Add MLRun's interface to the model:
    SKLearnMLRunInterface.add_interface(obj=model)

    # Add auto-logging if needed:
    if auto_log:
        # Get the artifacts plans and metrics lists:
        y = y_test if y_test is not None else y_validation
        plans = get_plans(
            artifacts_library=SKLearnArtifactLibrary,
            artifacts=artifacts,
            context=context,
            model=model,
            y=y,
        )
        metrics = get_metrics(
            metrics_library=MetricsLibrary,
            metrics=metrics,
            context=context,
            model=model,
            y=y,
        )
        # Set the kwargs dictionaries defaults:
        if model_logging_kwargs is None:
            model_logging_kwargs = {}
        # Add the logging callbacks with the provided parameters:
        model.auto_log(
            context=context,
            model_handler=handler,
            plans=plans,
            metrics=metrics,
            x_validation=x_validation,
            y_validation=y_validation,
            y_test=y_test,
            y_columns=y_columns,
            tag=tag,
            **model_logging_kwargs,
        )

    return handler
