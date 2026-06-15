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

import numpy as np
import pandas as pd

import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.datastore.base
import mlrun.feature_store
import mlrun.model_monitoring.applications as mm_app
import mlrun.serving
from mlrun.data_types.infer import InferOptions, get_df_stats
from mlrun.utils import check_if_hub_uri, datetime_now, merge_requirements

from ..common.schemas.hub import HubModuleType

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


def get_sample_set_statistics(
    sample_set: DatasetType = None,
    model_artifact_feature_stats: dict | None = None,
    sample_set_columns: list | None = None,
    sample_set_drop_columns: list | None = None,
    sample_set_label_columns: list | None = None,
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
    feature_columns: typing.Union[str, list[str]] | None = None,
    label_columns: typing.Union[str, list[str]] | None = None,
    drop_columns: typing.Union[str, list[str], int, list[int]] | None = None,
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

    elif isinstance(dataset, list | np.ndarray):
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
    elif isinstance(label_columns, str | int):
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
    name: str | None = None,
    image: str | None = None,
    tag: str | None = None,
    requirements: typing.Union[list[str], None] = None,
    requirements_file: str = "",
    local_path: str | None = None,
    otlp_enabled: bool = False,
    **application_kwargs,
) -> mlrun.runtimes.ServingRuntime:
    """
    Note: this is an internal API only.
    This function does not set the labels or mounts v3io.
    """
    if name in mm_constants._RESERVED_FUNCTION_NAMES:
        raise mlrun.errors.MLRunValueError(
            "An application cannot have the following names: "
            f"{mm_constants._RESERVED_FUNCTION_NAMES}"
        )
    _, has_valid_suffix, suffix = mlrun.utils.helpers.ensure_batch_job_suffix(name)
    if name and not has_valid_suffix:
        raise mlrun.errors.MLRunValueError(
            f"Model monitoring application names cannot end with `{suffix}`"
        )
    if func is None:
        func = ""
    if check_if_hub_uri(func):
        hub_module = mlrun.get_hub_module(url=func, local_path=local_path)
        if hub_module.kind != HubModuleType.monitoring_app:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The provided module is not a monitoring application"
            )
        requirements = mlrun.model.ImageBuilder.resolve_requirements(
            requirements, requirements_file
        )
        requirements = merge_requirements(
            reqs_priority=requirements, reqs_secondary=hub_module.requirements
        )
        func = hub_module.get_module_file_path()
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
    app_step.to(
        class_name="mlrun.model_monitoring.applications._application_steps._PushToMonitoringWriter",
        name="PushToMonitoringWriter",
        project=project,
    )

    if otlp_enabled:
        otel_prep = app_step.to(
            class_name="mlrun.model_monitoring.applications._application_steps._PrepareOTelEvent",
            name="PrepareOTelEvent",
        )
        otel_prep.to(
            class_name="mlrun.serving.OTelMetricsExporter",
            name="OTelMetricsExporter",
            headers_source="file",
        )
        func_obj.spec.mount_otlp_secret = otlp_enabled
    graph.error_handler(
        class_name="mlrun.model_monitoring.applications._application_steps._ApplicationErrorHandler",
        name="ApplicationErrorHandler",
        full_event=True,
        project=project,
        application_name=name,
        user_step_name=app_step.name,
    )

    def block_to_mock_server(*args, **kwargs) -> typing.NoReturn:
        raise NotImplementedError(
            "Model monitoring serving functions do not support `.to_mock_server`. "
            "You may call your model monitoring application object logic via the `.evaluate` method."
        )

    func_obj.to_mock_server = block_to_mock_server  # Until ML-7643 is implemented

    return func_obj
