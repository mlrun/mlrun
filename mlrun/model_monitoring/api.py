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

import hashlib
import typing
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import mlrun.artifacts
import mlrun.common.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.datastore.base
import mlrun.feature_store
import mlrun.model_monitoring.applications as mm_app
import mlrun.serving
from mlrun.common.schemas import ModelEndpoint
from mlrun.common.schemas.model_monitoring import (
    FunctionURI,
)
from mlrun.data_types.infer import InferOptions, get_df_stats
from mlrun.utils import datetime_now, logger

from .helpers import update_model_endpoint_last_request

# A union of all supported dataset types:
DatasetType = typing.Union[
    mlrun.datastore.base.DataItem,
    list,
    dict,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    typing.Any,
]


def get_or_create_model_endpoint(
    project: str,
    model_endpoint_name: str,
    model_path: str = "",
    endpoint_id: str = "",
    function_name: str = "",
    function_tag: str = "latest",
    context: typing.Optional["mlrun.MLClientCtx"] = None,
    sample_set_statistics: typing.Optional[dict[str, typing.Any]] = None,
    monitoring_mode: mm_constants.ModelMonitoringMode = mm_constants.ModelMonitoringMode.enabled,
    db_session=None,
    feature_analysis: bool = False,
) -> ModelEndpoint:
    """
    Get a single model endpoint object. If not exist, generate a new model endpoint with the provided parameters. Note
    that in case of generating a new model endpoint, by default the monitoring features are disabled. To enable these
    features, set `monitoring_mode=enabled`.

    :param project:                  Project name.
    :param model_endpoint_name:      If a new model endpoint is created, the model endpoint name will be presented
                                     under this endpoint (applicable only to new endpoint_id).
    :param model_path:               The model store path (applicable only to new endpoint_id).
    :param endpoint_id:              Model endpoint unique ID. If not exist in DB, will generate a new record based
                                     on the provided `endpoint_id`.
    :param function_name:            If a new model endpoint is created, use this function name.
    :param function_tag:             If a new model endpoint is created, use this function tag.
    :param context:                  MLRun context. If `function_name` not provided, use the context to generate the
                                     full function hash.
    :param sample_set_statistics:    Dictionary of sample set statistics that will be used as a reference data for
                                     the new model endpoint (applicable only to new endpoint_id).
    :param monitoring_mode:          If enabled, apply model monitoring features on the provided endpoint id
                                     (applicable only to new endpoint_id).
    :param db_session:               A runtime session that manages the current dialog with the database.
    :param feature_analysis:         If True, the model endpoint will be retrieved with the feature analysis mode.

    :return: A ModelEndpoint object
    """

    if not db_session:
        # Generate a runtime database
        db_session = mlrun.get_run_db()
    model_endpoint = None
    if not function_name and context:
        function_name = FunctionURI.from_string(
            context.to_dict()["spec"]["function"]
        ).function
    try:
        model_endpoint = db_session.get_model_endpoint(
            project=project,
            name=model_endpoint_name,
            endpoint_id=endpoint_id,
            function_name=function_name,
            function_tag=function_tag or "latest",
            feature_analysis=feature_analysis,
        )
        # If other fields provided, validate that they are correspond to the existing model endpoint data
        _model_endpoint_validations(
            model_endpoint=model_endpoint,
            model_path=model_path,
            sample_set_statistics=sample_set_statistics,
        )

    except (mlrun.errors.MLRunNotFoundError, mlrun.errors.MLRunInvalidArgumentError):
        # Create a new model endpoint with the provided details
        pass
    if not model_endpoint:
        model_endpoint = _generate_model_endpoint(
            project=project,
            db_session=db_session,
            model_path=model_path,
            model_endpoint_name=model_endpoint_name,
            function_name=function_name,
            function_tag=function_tag,
            monitoring_mode=monitoring_mode,
        )
    return model_endpoint


def record_results(
    project: str,
    model_path: str,
    model_endpoint_name: str,
    endpoint_id: str = "",
    function_name: str = "",
    context: typing.Optional["mlrun.MLClientCtx"] = None,
    infer_results_df: typing.Optional[pd.DataFrame] = None,
    sample_set_statistics: typing.Optional[dict[str, typing.Any]] = None,
    monitoring_mode: mm_constants.ModelMonitoringMode = mm_constants.ModelMonitoringMode.enabled,
    # Deprecated arguments:
    drift_threshold: typing.Optional[float] = None,
    possible_drift_threshold: typing.Optional[float] = None,
    trigger_monitoring_job: bool = False,
    artifacts_tag: str = "",
    default_batch_image: str = "mlrun/mlrun",
) -> ModelEndpoint:
    """
    Write a provided inference dataset to model endpoint parquet target. If not exist, generate a new model endpoint
    record and use the provided sample set statistics as feature stats that will be used later for the drift analysis.
    To activate model monitoring, run `project.enable_model_monitoring()`. The model monitoring applications will be
    triggered with the recorded data according to a periodic schedule.

    :param project:                  Project name.
    :param model_path:               The model Store path.
    :param model_endpoint_name:      If a new model endpoint is generated, the model endpoint name will be presented
                                     under this endpoint.
    :param endpoint_id:              Model endpoint unique ID. If not exist in DB, will generate a new record based
                                     on the provided `endpoint_id`.
    :param function_name:            If a new model endpoint is created, use this function name for generating the
                                     function URI.
    :param context:                  MLRun context. Note that the context is required generating the model endpoint.
    :param infer_results_df:         DataFrame that will be stored under the model endpoint parquet target. Will be
                                     used for doing the drift analysis. Please make sure that the dataframe includes
                                     both feature names and label columns. If you are recording results for existing
                                     model endpoint, the endpoint should be a batch endpoint.
    :param sample_set_statistics:    Dictionary of sample set statistics that will be used as a reference data for
                                     the current model endpoint.
    :param monitoring_mode:          If enabled, apply model monitoring features on the provided endpoint id. Enabled
                                     by default.
    :param drift_threshold:          (deprecated) The threshold of which to mark drifts.
    :param possible_drift_threshold: (deprecated) The threshold of which to mark possible drifts.
    :param trigger_monitoring_job:   (deprecated) If true, run the batch drift job. If not exists, the monitoring
                                     batch function will be registered through MLRun API with the provided image.
    :param artifacts_tag:            (deprecated) Tag to use for all the artifacts resulted from the function.
                                     Will be relevant only if the monitoring batch job has been triggered.
    :param default_batch_image:      (deprecated) The image that will be used when registering the model monitoring
                                     batch job.

    :return: A ModelEndpoint object
    """

    if drift_threshold is not None or possible_drift_threshold is not None:
        warnings.warn(
            "Custom drift threshold arguments are deprecated since version "
            "1.7.0 and have no effect. They will be removed in version 1.10.0.\n"
            "To enable the default histogram data drift application, run:\n"
            "`project.enable_model_monitoring()`.",
            FutureWarning,
        )
    if trigger_monitoring_job is not False:
        warnings.warn(
            "`trigger_monitoring_job` argument is deprecated since version "
            "1.7.0 and has no effect. It will be removed in version 1.10.0.\n"
            "To enable the default histogram data drift application, run:\n"
            "`project.enable_model_monitoring()`.",
            FutureWarning,
        )
    if artifacts_tag != "":
        warnings.warn(
            "`artifacts_tag` argument is deprecated since version "
            "1.7.0 and has no effect. It will be removed in version 1.10.0.",
            FutureWarning,
        )
    if default_batch_image != "mlrun/mlrun":
        warnings.warn(
            "`default_batch_image` argument is deprecated since version "
            "1.7.0 and has no effect. It will be removed in version 1.10.0.",
            FutureWarning,
        )

    db = mlrun.get_run_db()

    model_endpoint = get_or_create_model_endpoint(
        project=project,
        endpoint_id=endpoint_id,
        model_path=model_path,
        model_endpoint_name=model_endpoint_name,
        function_name=function_name,
        context=context,
        sample_set_statistics=sample_set_statistics,
        monitoring_mode=monitoring_mode,
        db_session=db,
    )
    logger.debug("Model endpoint", endpoint=model_endpoint)

    if infer_results_df is not None:
        if (
            model_endpoint.metadata.endpoint_type
            != mlrun.common.schemas.model_monitoring.EndpointType.BATCH_EP
        ):
            logger.warning(
                "Inference results can be recorded only for batch endpoints. "
                "Therefore the current results won't be monitored."
            )
        else:
            timestamp = datetime_now()
            # Write the monitoring parquet to the relevant model endpoint context
            write_monitoring_df(
                feature_set_uri=model_endpoint.spec.monitoring_feature_set_uri,
                infer_datetime=timestamp,
                endpoint_id=model_endpoint.metadata.uid,
                infer_results_df=infer_results_df,
            )

            # Update the last request time
            update_model_endpoint_last_request(
                project=project,
                model_endpoint=model_endpoint,
                current_request=timestamp,
                db=db,
            )

    return model_endpoint


def _model_endpoint_validations(
    model_endpoint: ModelEndpoint,
    model_path: str = "",
    sample_set_statistics: typing.Optional[dict[str, typing.Any]] = None,
) -> None:
    """
    Validate that provided model endpoint configurations match the stored fields of the provided `ModelEndpoint`
    object. Usually, this method is called by `get_or_create_model_endpoint()` in cases that the model endpoint
    already exist. If one of the validations fails, this method might raise an error, indicating on possible conflict.

    :param model_endpoint:           A `ModelEndpoint` object that contains the expected values.
    :param model_path:               Model store path. In case of endpoint_id reuse, should be similar to the model_uri
                                     that is stored under model_endpoint.spec.model_uri. Model endpoint record refers
                                     to a single model store path.
    :param sample_set_statistics:    Dictionary of sample set statistics. Once the model endpoint is registered, it
                                     is forbidden to provide a different reference data to that model endpoint.
                                     In case of discrepancy between the provided `sample_set_statistics` and the
                                     `model_endpoints.spec.feature_stats`, a warning will be presented to the user.
    """

    # Model Path
    if model_path:
        # Generate the parsed model uri that is based on hash, key, iter, and tree
        model_obj = mlrun.datastore.get_store_resource(model_path)

        model_artifact_uri = mlrun.utils.helpers.generate_artifact_uri(
            project=model_endpoint.metadata.project,
            key=model_obj.key,
            iter=model_obj.iter,
            tree=model_obj.tree,
        )

        # Enrich the uri schema with the store prefix
        model_artifact_uri = mlrun.datastore.get_store_uri(
            kind=mlrun.utils.helpers.StorePrefix.Model, uri=model_artifact_uri
        )

        if model_endpoint.spec.model_uri != model_artifact_uri:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"provided model store path {model_path} does not match "
                f"the path that is stored under the existing model "
                f"endpoint record: {model_endpoint.spec.model_uri}"
            )

    # Feature stats
    if (
        sample_set_statistics
        and sample_set_statistics != model_endpoint.spec.feature_stats
    ):
        logger.warning(
            "Provided sample set statistics is different from the registered statistics. "
            "If new sample set statistics is to be used, new model endpoint should be created"
        )


def write_monitoring_df(
    endpoint_id: str,
    infer_results_df: pd.DataFrame,
    infer_datetime: datetime,
    monitoring_feature_set: typing.Optional["mlrun.feature_store.FeatureSet"] = None,
    feature_set_uri: str = "",
) -> None:
    """Write infer results dataframe to the monitoring parquet target of the current model endpoint. The dataframe will
    be written using feature set ingest process. Please make sure that you provide either a valid monitoring feature
    set (with parquet target) or a valid monitoring feature set uri.

    :param endpoint_id:             Model endpoint unique ID.
    :param infer_results_df:        DataFrame that will be stored under the model endpoint parquet target.
    :param monitoring_feature_set:  A `mlrun.feature_store.FeatureSet` object corresponding to the provided endpoint_id.
    :param feature_set_uri:         if monitoring_feature_set not provided, use the feature_set_uri value to get the
                                    relevant `mlrun.feature_store.FeatureSet`.
    """

    if not monitoring_feature_set:
        if not feature_set_uri:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Please provide either a valid monitoring feature set object or a monitoring feature set uri"
            )

        monitoring_feature_set = mlrun.feature_store.get_feature_set(
            uri=feature_set_uri
        )

    # Modify the DataFrame to the required structure that will be used later by the monitoring batch job
    if mm_constants.EventFieldType.TIMESTAMP not in infer_results_df.columns:
        # Initialize timestamp column with the current time
        infer_results_df[mm_constants.EventFieldType.TIMESTAMP] = infer_datetime

    # `endpoint_id` is the monitoring feature set entity and therefore it should be defined as the df index before
    # the ingest process
    infer_results_df[mm_constants.EventFieldType.ENDPOINT_ID] = endpoint_id
    infer_results_df.set_index(mm_constants.EventFieldType.ENDPOINT_ID, inplace=True)

    monitoring_feature_set.ingest(source=infer_results_df, overwrite=False)


def _generate_model_endpoint(
    project: str,
    db_session,
    model_path: str,
    model_endpoint_name: str,
    function_name: str,
    function_tag: str,
    monitoring_mode: mm_constants.ModelMonitoringMode = mm_constants.ModelMonitoringMode.enabled,
) -> ModelEndpoint:
    """
    Write a new model endpoint record.

    :param project:                  Project name.

    :param db_session:               A session that manages the current dialog with the database.
    :param model_path:               The model Store path.
    :param model_endpoint_name:      Model endpoint name will be presented under the new model endpoint.
    :param function_name:            If a new model endpoint is created, use this function name.
    :param function_tag:             If a new model endpoint is created, use this function tag.
    :param monitoring_mode:          Monitoring mode of the new model endpoint.

    :return `mlrun.common.schemas.ModelEndpoint` object.
    """
    current_time = datetime_now()
    model_endpoint = mlrun.common.schemas.ModelEndpoint(
        metadata=mlrun.common.schemas.ModelEndpointMetadata(
            project=project,
            name=model_endpoint_name,
            endpoint_type=mlrun.common.schemas.model_monitoring.EndpointType.BATCH_EP,
        ),
        spec=mlrun.common.schemas.ModelEndpointSpec(
            function_name=function_name or "function",
            function_tag=function_tag or "latest",
            model_path=model_path,
            model_class="drift-analysis",
        ),
        status=mlrun.common.schemas.ModelEndpointStatus(
            monitoring_mode=monitoring_mode,
            first_request=current_time,
            last_request=current_time,
        ),
    )

    return db_session.create_model_endpoint(model_endpoint=model_endpoint)


def get_sample_set_statistics(
    sample_set: DatasetType = None,
    model_artifact_feature_stats: typing.Optional[dict] = None,
    sample_set_columns: typing.Optional[list] = None,
    sample_set_drop_columns: typing.Optional[list] = None,
    sample_set_label_columns: typing.Optional[list] = None,
) -> dict:
    """
    Get the sample set statistics either from the given sample set or the statistics logged with the model while
    favoring the given sample set.

    :param sample_set:                   A sample dataset to give to compare the inputs in the drift analysis.
    :param model_artifact_feature_stats: The `feature_stats` attribute in the spec of the model artifact, where the
                                         original sample set statistics of the model was used.
    :param sample_set_columns: The column names of sample_set.
    :param sample_set_drop_columns: ``str`` / ``int`` or a list of ``str`` / ``int`` that
                                    represent the column names / indices to drop.
    :param sample_set_label_columns: The target label(s) of the column(s) in the dataset. for Regression or
                                     Classification tasks.
    :returns: The sample set statistics.

    raises MLRunInvalidArgumentError: If no sample set or statistics were given.
    """
    # Check if a sample set was provided:
    if sample_set is None:
        # Check if the model was logged with a sample set:
        if model_artifact_feature_stats is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Cannot perform drift analysis as there is no sample set to compare to. The model artifact was not "
                "logged with a sample set and `sample_set` was not provided to the function."
            )
        # Return the statistics logged with the model:
        return model_artifact_feature_stats

    # Turn other object types to DataFrame:
    # A DataFrame is necessary for executing the "drop features" operation.
    dataset_types = list(DatasetType.__args__)
    if typing.Any in dataset_types:
        dataset_types.remove(typing.Any)
    if isinstance(
        sample_set,
        tuple(dataset_types),
    ):
        sample_set, _ = read_dataset_as_dataframe(
            dataset=sample_set,
            feature_columns=sample_set_columns,
            drop_columns=sample_set_drop_columns,
            label_columns=sample_set_label_columns,
        )
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Parameter sample_set has an unsupported type: {type(sample_set)}"
        )

    # Return the sample set statistics:
    return get_df_stats(df=sample_set, options=InferOptions.Histogram)


def read_dataset_as_dataframe(
    dataset: DatasetType,
    feature_columns: typing.Optional[typing.Union[str, list[str]]] = None,
    label_columns: typing.Optional[typing.Union[str, list[str]]] = None,
    drop_columns: typing.Optional[typing.Union[str, list[str], int, list[int]]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse the given dataset into a DataFrame and drop the columns accordingly. In addition, the label columns will be
    parsed and validated as well.

    :param dataset:         A dataset that will be converted into a DataFrame.
                            Can be either a list of lists, numpy.ndarray, dict, pd.Series, DataItem
                            or a FeatureVector.
    :param feature_columns: List of feature columns that will be used to build the dataframe when dataset is from
                            type list or numpy array.
    :param label_columns:   The target label(s) of the column(s) in the dataset. for Regression or
                            Classification tasks.
    :param drop_columns:    ``str`` / ``int`` or a list of ``str`` / ``int`` that represent the column names / indices
                            to drop.

    :returns: A tuple of:
              [0] = The parsed dataset as a DataFrame
              [1] = Label columns.

    raises MLRunInvalidArgumentError: If the `drop_columns` are not matching the dataset or unsupported dataset type.
    """
    # Turn the `drop labels` into a list if given:
    if drop_columns is not None:
        if not isinstance(drop_columns, list):
            drop_columns = [drop_columns]

    # Check if the dataset is in fact a Feature Vector:
    if isinstance(dataset, mlrun.feature_store.FeatureVector):
        # Try to get the label columns if not provided:
        if label_columns is None:
            label_columns = dataset.status.label_column
        # Get the features and parse to DataFrame:
        dataset = dataset.get_offline_features(drop_columns=drop_columns).to_dataframe()

    elif isinstance(dataset, (list, np.ndarray)):
        if not feature_columns:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Feature columns list must be provided when dataset input as from type list or numpy array"
            )
        # Parse the list / numpy array into a DataFrame:
        dataset = pd.DataFrame(dataset, columns=feature_columns)
        # Validate the `drop_columns` is given as integers:
        if drop_columns and not all(isinstance(col, int) for col in drop_columns):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "`drop_columns` must be an integer / list of integers if provided as a list."
            )
    elif isinstance(dataset, mlrun.DataItem):
        if (
            not dataset.url
            and dataset.artifact_url
            and mlrun.datastore.parse_store_uri(dataset.artifact_url)[0]
            == mlrun.utils.StorePrefix.FeatureVector
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"No data has been found. Make sure you have applied `get_offline_features` "
                f"on your feature vector {dataset.artifact_url} with a valid target before passing "
                f"it as an input."
            )
        # Turn the DataItem to DataFrame:
        dataset = dataset.as_df()
    else:
        # Parse the object (should be a pd.DataFrame / pd.Series, dictionary) into a DataFrame:
        try:
            dataset = pd.DataFrame(dataset)
        except ValueError as e:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Could not parse the given dataset of type {type(dataset)} into a pandas DataFrame. "
                f"Received the following error: {e}"
            )
    # Drop columns if needed:
    if drop_columns:
        dataset.drop(drop_columns, axis=1, inplace=True)

    # Turn the `label_columns` into a list by default:
    if label_columns is None:
        label_columns = []
    elif isinstance(label_columns, (str, int)):
        label_columns = [label_columns]

    return dataset, label_columns


def log_result(
    context: "mlrun.MLClientCtx",
    result_set_name: str,
    result_set: pd.DataFrame,
    artifacts_tag: str,
    batch_id: str,
) -> None:
    # Log the result set:
    context.logger.info("Logging result set (x | prediction)...")
    context.log_dataset(
        key=result_set_name,
        df=result_set,
        db_key=result_set_name,
        tag=artifacts_tag,
    )
    # Log the batch ID:
    if batch_id is None:
        batch_id = hashlib.sha224(str(datetime_now()).encode()).hexdigest()
    context.log_result(
        key="batch_id",
        value=batch_id,
    )


def _create_model_monitoring_function_base(
    *,
    project: str,
    func: typing.Union[str, None] = None,
    application_class: typing.Union[
        str, "mm_app.ModelMonitoringApplicationBase", None
    ] = None,
    name: typing.Optional[str] = None,
    image: typing.Optional[str] = None,
    tag: typing.Optional[str] = None,
    requirements: typing.Union[str, list[str], None] = None,
    requirements_file: str = "",
    **application_kwargs,
) -> mlrun.runtimes.ServingRuntime:
    """
    Note: this is an internal API only.
    This function does not set the labels or mounts v3io.
    """
    if name in mm_constants._RESERVED_FUNCTION_NAMES:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "An application cannot have the following names: "
            f"{mm_constants._RESERVED_FUNCTION_NAMES}"
        )
    if func is None:
        func = ""
    func_obj = typing.cast(
        mlrun.runtimes.ServingRuntime,
        mlrun.code_to_function(
            filename=func,
            name=name,
            project=project,
            tag=tag,
            kind=mlrun.run.RuntimeKinds.serving,
            image=image,
            requirements=requirements,
            requirements_file=requirements_file,
        ),
    )
    graph = func_obj.set_topology(mlrun.serving.states.StepKinds.flow)
    prepare_step = graph.to(
        class_name="mlrun.model_monitoring.applications._application_steps._PrepareMonitoringEvent",
        name="PrepareMonitoringEvent",
        application_name=name,
    )
    if isinstance(application_class, str):
        app_step = prepare_step.to(class_name=application_class, **application_kwargs)
    else:
        app_step = prepare_step.to(class_name=application_class)

    app_step.__class__ = mlrun.serving.MonitoringApplicationStep

    app_step.error_handler(
        class_name="mlrun.model_monitoring.applications._application_steps._ApplicationErrorHandler",
        name="ApplicationErrorHandler",
        full_event=True,
        project=project,
    )

    app_step.to(
        class_name="mlrun.model_monitoring.applications._application_steps._PushToMonitoringWriter",
        name="PushToMonitoringWriter",
        project=project,
    )

    def block_to_mock_server(*args, **kwargs) -> typing.NoReturn:
        raise NotImplementedError(
            "Model monitoring serving functions do not support `.to_mock_server`. "
            "You may call your model monitoring application object logic via the `.evaluate` method."
        )

    func_obj.to_mock_server = block_to_mock_server  # Until ML-7643 is implemented

    return func_obj
