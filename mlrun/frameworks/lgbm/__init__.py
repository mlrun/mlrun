import warnings
from typing import Dict, List, Union

import lightgbm as lgb

import mlrun

from .._common import ExtraDataType, get_plans
from .._ml_common import (
    DatasetType,
    Metric,
    MetricEntry,
    MetricsLibrary,
    MLArtifactsLibrary,
    MLPlan,
    PickleModelServer,
    get_metrics,
)
from .mlrun_interface import LGBMModelMLRunInterface
from .model_handler import LGBMModelHandler

# Placeholders as the SciKit-Learn API is commonly used among all of the ML frameworks:
LGBMArtifactsLibrary = MLArtifactsLibrary
LGBMMetricsLibrary = MetricsLibrary
LGBMModelServer = PickleModelServer


def apply_mlrun(
    model: lgb.LGBMModel = None,
    model_name: str = "model",
    tag: str = "",
    model_path: str = None,
    modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
    custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
    custom_objects_directory: str = None,
    context: mlrun.MLClientCtx = None,
    artifacts: Union[List[MLPlan], List[str], Dict[str, dict]] = None,
    metrics: Union[List[Metric], List[MetricEntry], Dict[str, MetricEntry]] = None,
    x_test: DatasetType = None,
    y_test: DatasetType = None,
    sample_set: Union[DatasetType, mlrun.DataItem, str] = None,
    y_columns: Union[List[str], List[int]] = None,
    feature_vector: str = None,
    feature_weights: List[float] = None,
    labels: Dict[str, Union[str, int, float]] = None,
    parameters: Dict[str, Union[str, int, float]] = None,
    extra_data: Dict[str, ExtraDataType] = None,
    auto_log: bool = True,
    **kwargs
) -> LGBMModelHandler:
    """
    Wrap the given model with MLRun's interface providing it with mlrun's additional features.

    :param model:                    The model to wrap. Can be loaded from the model path given as well.
    :param model_name:               The model name to use for storing the model artifact. Defaulted to "model".
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
    :param auto_log:                 Whether or not to apply MLRun's auto logging on the model. Auto logging will add
                                     the default artifacts and metrics to the lists of artifacts and metrics. Defaulted
                                     to True.

    :return: The model handler initialized with the provided model.
    """
    if "X_test" in kwargs:
        warnings.warn(
            "The attribute 'X_test' was changed to 'x_test' and will be removed next version.",
            # TODO: Remove in mlrun 1.0.0
            PendingDeprecationWarning,
        )
        x_test = kwargs["X_test"]
    if "X_train" in kwargs or "y_train" in kwargs:
        warnings.warn(
            "The attributes 'X_train' and 'y_train' are no longer required and will be removed next version.",
            # TODO: Remove in mlrun 1.0.0
            PendingDeprecationWarning,
        )

    # Get the default context:
    if context is None:
        context = mlrun.get_or_create_ctx(LGBMModelMLRunInterface.DEFAULT_CONTEXT_NAME)

    # Create a model handler:
    model_handler_kwargs = (
        kwargs.pop("model_handler_kwargs") if "model_handler_kwargs" in kwargs else {}
    )
    handler = LGBMModelHandler(
        model_name=model_name,
        model_path=model_path,
        model=model,
        context=context,
        modules_map=modules_map,
        custom_objects_map=custom_objects_map,
        custom_objects_directory=custom_objects_directory,
        **model_handler_kwargs,
    )

    # Load the model if it was not provided:
    if model is None:
        handler.load()
        model = handler.model

    # Set the handler's logging attributes:
    handler.set_tag(tag=tag)
    if sample_set is not None:
        handler.set_sample_set(sample_set=sample_set)
    if y_columns is not None:
        handler.set_y_columns(y_columns=y_columns)
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

    # Add MLRun's interface to the model:
    LGBMModelMLRunInterface.add_interface(obj=model)

    # Set the handler to the model:
    model.set_model_handler(model_handler=handler)

    # Configure the logger:
    model.configure_logger(
        context=context,
        plans=get_plans(
            artifacts_library=LGBMArtifactsLibrary,
            artifacts=artifacts,
            context=context,
            include_default=auto_log,
            model=model,
            y=y_test,
        ),
        metrics=get_metrics(
            metrics_library=LGBMMetricsLibrary,
            metrics=metrics,
            context=context,
            include_default=auto_log,
            model=model,
            y=y_test,
        ),
        x_test=x_test,
        y_test=y_test,
    )

    return handler
