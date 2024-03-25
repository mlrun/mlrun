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
import json
import typing
from datetime import datetime

import numpy as np
import pandas as pd

import mlrun.artifacts
import mlrun.common.helpers
import mlrun.feature_store
from mlrun.common.schemas.model_monitoring import EventFieldType, ModelMonitoringMode
from mlrun.data_types.infer import InferOptions, get_df_stats
from mlrun.utils import datetime_now, logger

from .batch import VirtualDrift
from .features_drift_table import FeaturesDriftTablePlot
from .helpers import update_model_endpoint_last_request
from .model_endpoint import ModelEndpoint

# A union of all supported dataset types:
DatasetType = typing.Union[
    mlrun.DataItem, list, dict, pd.DataFrame, pd.Series, np.ndarray, typing.Any
]


def get_or_create_model_endpoint(
    project: str,
    model_path: str = "",
    model_endpoint_name: str = "",
    endpoint_id: str = "",
    function_name: str = "",
    context: mlrun.MLClientCtx = None,
    sample_set_statistics: dict[str, typing.Any] = None,
    drift_threshold: float = None,
    possible_drift_threshold: float = None,
    monitoring_mode: ModelMonitoringMode = ModelMonitoringMode.disabled,
    db_session=None,
) -> ModelEndpoint:
    """
    Get a single model endpoint object. If not exist, generate a new model endpoint with the provided parameters. Note
    that in case of generating a new model endpoint, by default the monitoring features are disabled. To enable these
    features, set `monitoring_mode=enabled`.

    :param project:                  Project name.
    :param model_path:               The model store path (applicable only to new endpoint_id).
    :param model_endpoint_name:      If a new model endpoint is created, the model endpoint name will be presented
                                     under this endpoint (applicable only to new endpoint_id).
    :param endpoint_id:              Model endpoint unique ID. If not exist in DB, will generate a new record based
                                     on the provided `endpoint_id`.
    :param function_name:            If a new model endpoint is created, use this function name for generating the
                                     function URI (applicable only to new endpoint_id).
    :param context:                  MLRun context. If `function_name` not provided, use the context to generate the
                                     full function hash.
    :param sample_set_statistics:    Dictionary of sample set statistics that will be used as a reference data for
                                     the new model endpoint (applicable only to new endpoint_id).
    :param drift_threshold:          The threshold of which to mark drifts (applicable only to new endpoint_id).
    :param possible_drift_threshold: The threshold of which to mark possible drifts (applicable only to new
                                     endpoint_id).
    :param monitoring_mode:          If enabled, apply model monitoring features on the provided endpoint id
                                     (applicable only to new endpoint_id).
    :param db_session:               A runtime session that manages the current dialog with the database.


    :return: A ModelEndpoint object
    """

    if not endpoint_id:
        # Generate a new model endpoint id based on the project name and model name
        endpoint_id = hashlib.sha1(
            f"{project}_{model_endpoint_name}".encode()
        ).hexdigest()

    if not db_session:
        # Generate a runtime database
        db_session = mlrun.get_run_db()
    try:
        model_endpoint = db_session.get_model_endpoint(
            project=project, endpoint_id=endpoint_id
        )
        # If other fields provided, validate that they are correspond to the existing model endpoint data
        _model_endpoint_validations(
            model_endpoint=model_endpoint,
            model_path=model_path,
            sample_set_statistics=sample_set_statistics,
            drift_threshold=drift_threshold,
            possible_drift_threshold=possible_drift_threshold,
        )

    except mlrun.errors.MLRunNotFoundError:
        # Create a new model endpoint with the provided details
        model_endpoint = _generate_model_endpoint(
            project=project,
            db_session=db_session,
            endpoint_id=endpoint_id,
            model_path=model_path,
            model_endpoint_name=model_endpoint_name,
            function_name=function_name,
            context=context,
            sample_set_statistics=sample_set_statistics,
            drift_threshold=drift_threshold,
            possible_drift_threshold=possible_drift_threshold,
            monitoring_mode=monitoring_mode,
        )
    return model_endpoint


def record_results(
    project: str,
    model_path: str,
    model_endpoint_name: str,
    endpoint_id: str = "",
    function_name: str = "",
    context: typing.Optional[mlrun.MLClientCtx] = None,
    infer_results_df: typing.Optional[pd.DataFrame] = None,
    sample_set_statistics: typing.Optional[dict[str, typing.Any]] = None,
    monitoring_mode: ModelMonitoringMode = ModelMonitoringMode.enabled,
    drift_threshold: typing.Optional[float] = None,
    possible_drift_threshold: typing.Optional[float] = None,
    trigger_monitoring_job: bool = False,
    artifacts_tag: str = "",
    default_batch_image="mlrun/mlrun",
) -> ModelEndpoint:
    """
    Write a provided inference dataset to model endpoint parquet target. If not exist, generate a new model endpoint
    record and use the provided sample set statistics as feature stats that will be used later for the drift analysis.
    To manually trigger the monitoring batch job, set `trigger_monitoring_job=True` and then the batch
    job will immediately perform drift analysis between the sample set statistics stored in the model and the new
    input data (along with the outputs). The drift rule is the value per-feature mean of the TVD and Hellinger scores
    according to the provided thresholds.

    :param project:                  Project name.
    :param model_path:               The model Store path.
    :param model_endpoint_name:      If a new model endpoint is generated, the model endpoint name will be presented
                                     under this endpoint.
    :param endpoint_id:              Model endpoint unique ID. If not exist in DB, will generate a new record based
                                     on the provided `endpoint_id`.
    :param function_name:            If a new model endpoint is created, use this function name for generating the
                                     function URI.
    :param context:                  MLRun context. Note that the context is required for logging the artifacts
                                     following the batch drift job.
    :param infer_results_df:         DataFrame that will be stored under the model endpoint parquet target. Will be
                                     used for doing the drift analysis. Please make sure that the dataframe includes
                                     both feature names and label columns.
    :param sample_set_statistics:    Dictionary of sample set statistics that will be used as a reference data for
                                     the current model endpoint.
    :param monitoring_mode:          If enabled, apply model monitoring features on the provided endpoint id. Enabled
                                     by default.
    :param drift_threshold:          The threshold of which to mark drifts.
    :param possible_drift_threshold: The threshold of which to mark possible drifts.
    :param trigger_monitoring_job:   If true, run the batch drift job. If not exists, the monitoring batch function
                                     will be registered through MLRun API with the provided image.
    :param artifacts_tag:            Tag to use for all the artifacts resulted from the function. Will be relevant
                                     only if the monitoring batch job has been triggered.

    :param default_batch_image:      The image that will be used when registering the model monitoring batch job.

    :return: A ModelEndpoint object
    """
    db = mlrun.get_run_db()

    model_endpoint = get_or_create_model_endpoint(
        project=project,
        endpoint_id=endpoint_id,
        model_path=model_path,
        model_endpoint_name=model_endpoint_name,
        function_name=function_name,
        context=context,
        sample_set_statistics=sample_set_statistics,
        drift_threshold=drift_threshold,
        possible_drift_threshold=possible_drift_threshold,
        monitoring_mode=monitoring_mode,
        db_session=db,
    )
    logger.debug("Model endpoint", endpoint=model_endpoint.to_dict())

    timestamp = datetime_now()
    if infer_results_df is not None:
        # Write the monitoring parquet to the relevant model endpoint context
        write_monitoring_df(
            feature_set_uri=model_endpoint.status.monitoring_feature_set_uri,
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

    if trigger_monitoring_job:
        # Run the monitoring batch drift job
        trigger_drift_batch_job(
            project=project,
            default_batch_image=default_batch_image,
            model_endpoints_ids=[model_endpoint.metadata.uid],
            db_session=db,
        )

        # Getting drift thresholds if not provided
        drift_threshold, possible_drift_threshold = get_drift_thresholds_if_not_none(
            model_endpoint=model_endpoint,
            drift_threshold=drift_threshold,
            possible_drift_threshold=possible_drift_threshold,
        )

        perform_drift_analysis(
            project=project,
            context=context,
            sample_set_statistics=model_endpoint.status.feature_stats,
            drift_threshold=drift_threshold,
            possible_drift_threshold=possible_drift_threshold,
            artifacts_tag=artifacts_tag,
            endpoint_id=model_endpoint.metadata.uid,
            db_session=db,
        )

    return model_endpoint


def _model_endpoint_validations(
    model_endpoint: ModelEndpoint,
    model_path: str = "",
    sample_set_statistics: dict[str, typing.Any] = None,
    drift_threshold: float = None,
    possible_drift_threshold: float = None,
):
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
    :param drift_threshold:          The threshold of which to mark drifts. Should be similar to the drift threshold
                                     that has already assigned to the current model endpoint.
    :param possible_drift_threshold: The threshold of which to mark possible drifts. Should be similar to the possible
                                     drift threshold  that has already assigned to the current model endpoint.

    """
    # Model path
    if model_path and model_endpoint.spec.model_uri != model_path:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"provided model store path {model_path} does not match "
            f"the path that is stored under the existing model "
            f"endpoint record: {model_endpoint.spec.model_uri}"
        )

    # Feature stats
    if (
        sample_set_statistics
        and sample_set_statistics != model_endpoint.status.feature_stats
    ):
        logger.warning(
            "Provided sample set statistics is different from the registered statistics. "
            "If new sample set statistics is to be used, new model endpoint should be created"
        )
    # drift and possible drift thresholds
    if drift_threshold:
        current_drift_threshold = model_endpoint.spec.monitor_configuration.get(
            EventFieldType.DRIFT_DETECTED_THRESHOLD,
            mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.drift_detected,
        )
        if current_drift_threshold != drift_threshold:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Cannot change existing drift threshold. Expected {current_drift_threshold}, got {drift_threshold} "
                f"Please update drift threshold or generate a new model endpoint record"
            )

    if possible_drift_threshold:
        current_possible_drift_threshold = model_endpoint.spec.monitor_configuration.get(
            EventFieldType.POSSIBLE_DRIFT_THRESHOLD,
            mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.possible_drift,
        )
        if current_possible_drift_threshold != possible_drift_threshold:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Cannot change existing possible drift threshold. Expected {current_possible_drift_threshold}, "
                f"got {possible_drift_threshold}. Please update drift threshold or generate a new model endpoint record"
            )


def get_drift_thresholds_if_not_none(
    model_endpoint: ModelEndpoint,
    drift_threshold: float = None,
    possible_drift_threshold: float = None,
) -> tuple[float, float]:
    """
    Get drift and possible drift thresholds. If one of the thresholds is missing, will try to retrieve
    it from the `ModelEndpoint` object. If not defined under the `ModelEndpoint` as well, will retrieve it from
    the default mlrun configuration.

    :param model_endpoint:           `ModelEndpoint` object.
    :param drift_threshold:           The threshold of which to mark drifts.
    :param possible_drift_threshold:  The threshold of which to mark possible drifts.

    :return: A Tuple of:
            [0] drift threshold as a float
            [1] possible drift threshold as a float
    """
    if not drift_threshold:
        # Getting drift threshold value from either model endpoint or monitoring default configurations
        drift_threshold = model_endpoint.spec.monitor_configuration.get(
            EventFieldType.DRIFT_DETECTED_THRESHOLD,
            mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.drift_detected,
        )
    if not possible_drift_threshold:
        # Getting possible drift threshold value from either model endpoint or monitoring default configurations
        possible_drift_threshold = model_endpoint.spec.monitor_configuration.get(
            EventFieldType.POSSIBLE_DRIFT_THRESHOLD,
            mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.possible_drift,
        )

    return drift_threshold, possible_drift_threshold


def write_monitoring_df(
    endpoint_id: str,
    infer_results_df: pd.DataFrame,
    infer_datetime: datetime,
    monitoring_feature_set: typing.Optional[mlrun.feature_store.FeatureSet] = None,
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
    if EventFieldType.TIMESTAMP not in infer_results_df.columns:
        # Initialize timestamp column with the current time
        infer_results_df[EventFieldType.TIMESTAMP] = infer_datetime

    # `endpoint_id` is the monitoring feature set entity and therefore it should be defined as the df index before
    # the ingest process
    infer_results_df[EventFieldType.ENDPOINT_ID] = endpoint_id
    infer_results_df.set_index(EventFieldType.ENDPOINT_ID, inplace=True)

    monitoring_feature_set.ingest(source=infer_results_df, overwrite=False)


def _generate_model_endpoint(
    project: str,
    db_session,
    endpoint_id: str,
    model_path: str,
    model_endpoint_name: str,
    function_name: str,
    context: mlrun.MLClientCtx,
    sample_set_statistics: dict[str, typing.Any],
    drift_threshold: float,
    possible_drift_threshold: float,
    monitoring_mode: ModelMonitoringMode = ModelMonitoringMode.disabled,
) -> ModelEndpoint:
    """
    Write a new model endpoint record.

    :param project:                  Project name.

    :param db_session:               A session that manages the current dialog with the database.
    :param endpoint_id:              Model endpoint unique ID.
    :param model_path:               The model Store path.
    :param model_endpoint_name:      Model endpoint name will be presented under the new model endpoint.
    :param function_name:            If a new model endpoint is created, use this function name for generating the
                                     function URI.
    :param context:                  MLRun context. If function_name not provided, use the context to generate the
                                     full function hash.
    :param sample_set_statistics:    Dictionary of sample set statistics that will be used as a reference data for
                                     the current model endpoint. Will be stored under
                                     `model_endpoint.status.feature_stats`.
    :param drift_threshold:          The threshold of which to mark drifts.
    :param possible_drift_threshold: The threshold of which to mark possible drifts.

    :return `mlrun.model_monitoring.model_endpoint.ModelEndpoint` object.
    """
    model_endpoint = ModelEndpoint()
    model_endpoint.metadata.project = project
    model_endpoint.metadata.uid = endpoint_id
    if function_name:
        model_endpoint.spec.function_uri = project + "/" + function_name
    elif not context:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Please provide either a function name or a valid MLRun context"
        )
    else:
        model_endpoint.spec.function_uri = context.to_dict()["spec"]["function"]
    model_endpoint.spec.model_uri = model_path
    model_endpoint.spec.model = model_endpoint_name
    model_endpoint.spec.model_class = "drift-analysis"
    if drift_threshold:
        model_endpoint.spec.monitor_configuration[
            EventFieldType.DRIFT_DETECTED_THRESHOLD
        ] = drift_threshold
    if possible_drift_threshold:
        model_endpoint.spec.monitor_configuration[
            EventFieldType.POSSIBLE_DRIFT_THRESHOLD
        ] = possible_drift_threshold

    model_endpoint.spec.monitoring_mode = monitoring_mode
    model_endpoint.status.first_request = model_endpoint.status.last_request = (
        datetime_now().isoformat()
    )
    if sample_set_statistics:
        model_endpoint.status.feature_stats = sample_set_statistics

    db_session.create_model_endpoint(
        project=project, endpoint_id=endpoint_id, model_endpoint=model_endpoint
    )

    return db_session.get_model_endpoint(project=project, endpoint_id=endpoint_id)


def trigger_drift_batch_job(
    project: str,
    default_batch_image="mlrun/mlrun",
    model_endpoints_ids: list[str] = None,
    batch_intervals_dict: dict[str, float] = None,
    db_session=None,
):
    """
    Run model monitoring drift analysis job. If not exists, the monitoring batch function will be registered through
    MLRun API with the provided image.

    :param project:              Project name.
    :param default_batch_image:  The image that will be used when registering the model monitoring batch job.
    :param model_endpoints_ids:  List of model endpoints to include in the current run.
    :param batch_intervals_dict: Batch interval range (days, hours, minutes). By default, the batch interval is
                                 configured to run through the last hour.
    :param db_session:           A runtime session that manages the current dialog with the database.

    """
    if not model_endpoints_ids:
        raise mlrun.errors.MLRunNotFoundError(
            "No model endpoints provided",
        )
    if not db_session:
        db_session = mlrun.get_run_db()

    # Register the monitoring batch job (do nothing if already exist) and get the job function as a dictionary
    batch_function_dict: dict[str, typing.Any] = db_session.deploy_monitoring_batch_job(
        project=project,
        default_batch_image=default_batch_image,
    )

    # Prepare current run params
    job_params = _generate_job_params(
        model_endpoints_ids=model_endpoints_ids,
        batch_intervals_dict=batch_intervals_dict,
    )

    # Generate runtime and trigger the job function
    batch_function = mlrun.new_function(runtime=batch_function_dict)
    batch_function.run(name="model-monitoring-batch", params=job_params, watch=True)


def _generate_job_params(
    model_endpoints_ids: list[str],
    batch_intervals_dict: dict[str, float] = None,
):
    """
    Generate the required params for the model monitoring batch job function.

    :param model_endpoints_ids:  List of model endpoints to include in the current run.
    :param batch_intervals_dict: Batch interval range (days, hours, minutes). By default, the batch interval is
                                 configured to run through the last hour.

    """
    if not batch_intervals_dict:
        # Generate default batch intervals dict
        batch_intervals_dict = {"minutes": 0, "hours": 1, "days": 0}

    return {
        "model_endpoints": model_endpoints_ids,
        "batch_intervals_dict": batch_intervals_dict,
    }


def get_sample_set_statistics(
    sample_set: DatasetType = None,
    model_artifact_feature_stats: dict = None,
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
    feature_columns: typing.Union[str, list[str]] = None,
    label_columns: typing.Union[str, list[str]] = None,
    drop_columns: typing.Union[str, list[str], int, list[int]] = None,
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


def perform_drift_analysis(
    project: str,
    endpoint_id: str,
    context: mlrun.MLClientCtx,
    sample_set_statistics: dict,
    drift_threshold: float,
    possible_drift_threshold: float,
    artifacts_tag: str = "",
    db_session=None,
) -> None:
    """
    Calculate drift per feature and produce the drift table artifact for logging post prediction. Note that most of
    the calculations were already made through the monitoring batch job.

    :param project:                  Project name.
    :param endpoint_id:              Model endpoint unique ID.
    :param context:                  MLRun context. Will log the artifacts.
    :param sample_set_statistics:    The statistics of the sample set logged along a model.
    :param drift_threshold:          The threshold of which to mark drifts.
    :param possible_drift_threshold: The threshold of which to mark possible drifts.
    :param artifacts_tag:            Tag to use for all the artifacts resulted from the function.
    :param db_session:               A runtime session that manages the current dialog with the database.

    """
    if not db_session:
        db_session = mlrun.get_run_db()

    model_endpoint = db_session.get_model_endpoint(
        project=project, endpoint_id=endpoint_id
    )

    # Get the drift metrics results along with the feature statistics from the latest batch
    metrics = model_endpoint.status.drift_measures
    inputs_statistics = model_endpoint.status.current_stats

    inputs_statistics.pop(EventFieldType.TIMESTAMP, None)

    # Calculate drift for each feature
    virtual_drift = VirtualDrift()
    drift_results = virtual_drift.check_for_drift_per_feature(
        metrics_results_dictionary=metrics,
        possible_drift_threshold=possible_drift_threshold,
        drift_detected_threshold=drift_threshold,
    )

    # Drift table artifact
    plotly_artifact = FeaturesDriftTablePlot().produce(
        sample_set_statistics=sample_set_statistics,
        inputs_statistics=inputs_statistics,
        metrics=metrics,
        drift_results=drift_results,
    )

    # Prepare drift result per feature dictionary
    metrics_per_feature = {
        feature: _get_drift_result(
            tvd=metric_dictionary["tvd"],
            hellinger=metric_dictionary["hellinger"],
            threshold=drift_threshold,
        )[1]
        for feature, metric_dictionary in metrics.items()
        if isinstance(metric_dictionary, dict)
    }

    # Calculate the final analysis result as well
    drift_status, drift_metric = _get_drift_result(
        tvd=metrics["tvd_mean"],
        hellinger=metrics["hellinger_mean"],
        threshold=drift_threshold,
    )
    # Log the different artifacts
    _log_drift_artifacts(
        context=context,
        plotly_artifact=plotly_artifact,
        metrics_per_feature=metrics_per_feature,
        drift_status=drift_status,
        drift_metric=drift_metric,
        artifacts_tag=artifacts_tag,
    )


def _log_drift_artifacts(
    context: mlrun.MLClientCtx,
    plotly_artifact: mlrun.artifacts.Artifact,
    metrics_per_feature: dict[str, float],
    drift_status: bool,
    drift_metric: float,
    artifacts_tag: str,
):
    """
    Log the following artifacts/results:
    1 - Drift table plot which includes a detailed drift analysis per feature
    2 - Drift result per feature in a JSON format
    3 - Results of the total drift analysis

    :param context:             MLRun context. Will log the artifacts.
    :param plotly_artifact:     The plotly artifact.
    :param metrics_per_feature: Dictionary in which the key is a feature name and the value is the drift numerical
                                result.
    :param drift_status:        Boolean value that represents the final drift analysis result.
    :param drift_metric:        The final drift numerical result.
    :param artifacts_tag:       Tag to use for all the artifacts resulted from the function.
    """
    context.log_artifact(plotly_artifact, tag=artifacts_tag)
    context.log_artifact(
        mlrun.artifacts.Artifact(
            body=json.dumps(metrics_per_feature),
            format="json",
            key="features_drift_results",
        ),
        tag=artifacts_tag,
    )
    context.log_results(
        results={"drift_status": drift_status, "drift_metric": drift_metric}
    )


def _get_drift_result(
    tvd: float,
    hellinger: float,
    threshold: float,
) -> tuple[bool, float]:
    """
    Calculate the drift result by the following equation: (tvd + hellinger) / 2

    :param tvd:       The feature's TVD value.
    :param hellinger: The feature's Hellinger value.
    :param threshold: The threshold from which the value is considered a drift.

    :returns: A tuple of:
              [0] = Boolean value as the drift status.
              [1] = The result.
    """
    result = (tvd + hellinger) / 2
    if result >= threshold:
        return True, result
    return False, result


def log_result(
    context: mlrun.MLClientCtx,
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
