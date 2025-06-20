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

import copy
import importlib.util
import pathlib
import sys
from datetime import datetime
from typing import Any, Optional, Union

import pandas as pd

import mlrun
import mlrun.errors

from ..data_types import InferOptions, get_infer_interface
from ..datastore.sources import BaseSourceDriver, StreamSource
from ..datastore.store_resources import parse_store_uri
from ..datastore.targets import (
    BaseStoreTarget,
    get_default_prefix_for_source,
    get_target_driver,
    kind_to_driver,
    validate_target_list,
    validate_target_paths_for_engine,
    write_spark_dataframe_with_options,
)
from ..model import DataSource, DataTargetBase
from ..runtimes import BaseRuntime, RuntimeKinds
from ..runtimes.function_reference import FunctionReference
from ..serving.server import Response
from ..utils import get_caller_globals, logger, normalize_name
from .common import (
    RunConfig,
    get_feature_set_by_uri,
    get_feature_vector_by_uri,
    verify_feature_set_exists,
    verify_feature_set_permissions,
)
from .feature_set import FeatureSet
from .ingestion import (
    context_to_ingestion_params,
    init_featureset_graph,
    run_ingestion_job,
    run_spark_graph,
)

_v3iofs = None
spark_transform_handler = "transform"
_TRANS_TABLE = str.maketrans({" ": "_", "(": "", ")": ""})


def norm_column_name(name: str) -> str:
    """
    Remove parentheses () and replace whitespaces with an underscore _.
    Used to normalize a column/feature name.
    """
    return name.translate(_TRANS_TABLE)


def _rename_source_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_mapping = {}
    column_set = set(df.columns)
    for column in df.columns:
        if isinstance(column, str):
            rename_to = norm_column_name(column)
            if rename_to != column:
                if rename_to in column_set:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f'column "{column}" cannot be renamed to "{rename_to}" because such a column already exists'
                    )
                rename_mapping[column] = rename_to
                column_set.add(rename_to)
    if rename_mapping:
        logger.warn(
            f"the following dataframe columns have been renamed due to unsupported characters: {rename_mapping}"
        )
        df = df.rename(rename_mapping, axis=1)
    return df


def _get_namespace(run_config: RunConfig) -> dict[str, Any]:
    # if running locally, we need to import the file dynamically to get its namespace
    if run_config and run_config.local and run_config.function:
        filename = run_config.function.spec.filename
        if filename:
            module_name = pathlib.Path(filename).name.rsplit(".", maxsplit=1)[0]
            spec = importlib.util.spec_from_file_location(module_name, filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return vars(__import__(module_name))
    else:
        return get_caller_globals()


def ingest(
    mlrun_context: Union["mlrun.MLrunProject", "mlrun.MLClientCtx"],
    featureset: Union[FeatureSet, str] = None,
    source=None,
    targets: Optional[list[DataTargetBase]] = None,
    namespace=None,
    return_df: bool = True,
    infer_options: InferOptions = InferOptions.default(),
    run_config: RunConfig = None,
    spark_context=None,
    overwrite=None,
) -> Optional[pd.DataFrame]:
    """Read local DataFrame, file, URL, or source into the feature store
    Ingest reads from the source, run the graph transformations, infers  metadata and stats
    and writes the results to the default of specified targets

    when targets are not specified data is stored in the configured default targets
    (will usually be NoSQL for real-time and Parquet for offline).

    the `run_config` parameter allow specifying the function and job configuration,
    see: :py:class:`~mlrun.feature_store.RunConfig`

    example::

        stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
        stocks = pd.read_csv("stocks.csv")
        df = ingest(stocks_set, stocks, infer_options=fstore.InferOptions.default())

        # for running as remote job
        config = RunConfig(image="mlrun/mlrun")
        df = ingest(stocks_set, stocks, run_config=config)

        # specify source and targets
        source = CSVSource("mycsv", path="measurements.csv")
        targets = [CSVTarget("mycsv", path="./mycsv.csv")]
        ingest(measurements, source, targets)

    :param mlrun_context: mlrun context
    :param featureset:    feature set object or featureset.uri. (uri must be of a feature set that is in the DB,
                          call `.save()` if it's not)
    :param source:        source dataframe or other sources (e.g. parquet source see:
                          :py:class:`~mlrun.datastore.ParquetSource` and other classes in mlrun.datastore with suffix
                          Source)
    :param targets:       optional list of data target objects
    :param namespace:     namespace or module containing graph classes
    :param return_df:     indicate if to return a dataframe with the graph results
    :param infer_options: schema (for discovery of entities, features in featureset), index, stats,
                          histogram and preview infer options (:py:class:`~mlrun.feature_store.InferOptions`)
    :param run_config:    function and/or run configuration for remote jobs,
                          see :py:class:`~mlrun.feature_store.RunConfig`
    :param spark_context: local spark session for spark ingestion, example for creating the spark context:
                          `spark = SparkSession.builder.appName("Spark function").getOrCreate()`
                          For remote spark ingestion, this should contain the remote spark service name
    :param overwrite:     delete the targets' data prior to ingestion
                          (default: True for non scheduled ingest - deletes the targets that are about to be ingested.
                          False for scheduled ingest - does not delete the target)
    :return:              if return_df is True, a dataframe will be returned based on the graph
    """
    if not mlrun_context:
        raise mlrun.errors.MLRunValueError(
            "mlrun_context must be defined when calling ingest()"
        )

    return _ingest(
        featureset,
        source,
        targets,
        namespace,
        return_df,
        infer_options,
        run_config,
        mlrun_context,
        spark_context,
        overwrite,
    )


def _ingest(
    featureset: Union[FeatureSet, str] = None,
    source=None,
    targets: Optional[list[DataTargetBase]] = None,
    namespace=None,
    return_df: bool = True,
    infer_options: InferOptions = InferOptions.default(),
    run_config: RunConfig = None,
    mlrun_context=None,
    spark_context=None,
    overwrite=None,
) -> Optional[pd.DataFrame]:
    if isinstance(source, pd.DataFrame):
        source = _rename_source_dataframe_columns(source)

    if featureset:
        if isinstance(featureset, str):
            # need to strip store prefix from the uri
            _, stripped_name = parse_store_uri(featureset)
            try:
                featureset = get_feature_set_by_uri(stripped_name)
            except mlrun.db.RunDBError as exc:
                # TODO: this handling is needed because the generic httpdb error handling doesn't raise the correct
                #  error class and doesn't propagate the correct message, until it solved we're manually handling this
                #  case to give better user experience, remove this when the error handling is fixed.
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"{exc}. Make sure the feature set is saved in DB (call feature_set.save())"
                )

        # feature-set spec always has a source property that is not None. It may be default-constructed, in which
        # case the path will be 'None'. That's why we need a special check
        if source is None and featureset.has_valid_source():
            source = featureset.spec.source

    if not mlrun_context and (not featureset or source is None):
        raise mlrun.errors.MLRunInvalidArgumentError(
            "feature set and source must be specified"
        )
    if (
        not mlrun_context
        and not targets
        and not (featureset.spec.targets or featureset.spec.with_default_targets)
        and (run_config is not None and not run_config.local)
    ):
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Feature set {featureset.metadata.name} is remote ingested with no targets defined, aborting"
        )

    if featureset is not None:
        featureset.validate_steps(namespace=namespace)
    # This flow may happen both on client side (user provides run config) and server side (through the ingest API)
    if run_config and not run_config.local:
        if isinstance(source, pd.DataFrame):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "DataFrame source is illegal in conjunction with run_config"
            )
        # remote job execution
        verify_feature_set_permissions(
            featureset, mlrun.common.schemas.AuthorizationAction.update
        )
        run_config = run_config.copy() if run_config else RunConfig()
        source, run_config.parameters = set_task_params(
            featureset, source, targets, run_config.parameters, infer_options, overwrite
        )
        name = f"{featureset.metadata.name}_ingest"
        schedule = source.schedule
        if schedule == "mock":
            schedule = None
        return run_ingestion_job(name, featureset, run_config, schedule, spark_context)

    if mlrun_context:
        # extract ingestion parameters from mlrun context
        if isinstance(source, pd.DataFrame):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "DataFrame source is illegal when running ingest remotely"
            )
        if featureset or source is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot specify mlrun_context with feature set or source"
            )
        (
            featureset,
            source,
            targets,
            infer_options,
            overwrite,
        ) = context_to_ingestion_params(mlrun_context)

        featureset.validate_steps(namespace=namespace)
        verify_feature_set_permissions(
            featureset, mlrun.common.schemas.AuthorizationAction.update
        )
        if not source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "data source was not specified"
            )

        filter_time_string = ""
        if source.schedule:
            featureset.reload(update_spec=False)

    if isinstance(source, DataSource) and source.schedule:
        min_time = datetime.max
        for target in featureset.status.targets:
            if target.last_written:
                cur_last_written = target.last_written
                if isinstance(cur_last_written, str):
                    cur_last_written = datetime.fromisoformat(target.last_written)
                if cur_last_written < min_time:
                    min_time = cur_last_written
        if min_time != datetime.max:
            source.start_time = min_time
            time_zone = min_time.tzinfo
            source.end_time = datetime.now(tz=time_zone)
            filter_time_string = (
                f"Source.start_time for the job is{str(source.start_time)}. "
                f"Source.end_time is {str(source.end_time)}"
            )

        if mlrun_context:
            mlrun_context.logger.info(
                f"starting ingestion task to {featureset.uri}.{filter_time_string}"
            )

        return_df = False

    if featureset.spec.passthrough:
        featureset.spec.source = source
        featureset.spec.validate_no_processing_for_passthrough()

    if not namespace:
        namespace = _get_namespace(run_config)

    targets_to_ingest = targets or featureset.spec.targets
    targets_to_ingest = copy.deepcopy(targets_to_ingest)

    validate_target_paths_for_engine(targets_to_ingest, featureset.spec.engine, source)

    if overwrite is None:
        if isinstance(source, BaseSourceDriver) and source.schedule:
            overwrite = False
        else:
            overwrite = True

    if overwrite:
        validate_target_list(targets=targets_to_ingest)
        purge_target_names = [
            t if isinstance(t, str) else t.name for t in targets_to_ingest
        ]
        featureset.purge_targets(target_names=purge_target_names, silent=True)

        featureset.update_targets_for_ingest(
            targets=targets_to_ingest,
            overwrite=overwrite,
        )
    else:
        featureset.update_targets_for_ingest(
            targets=targets_to_ingest,
            overwrite=overwrite,
        )

        for target in targets_to_ingest:
            if not kind_to_driver[target.kind].support_append:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"{target.kind} target does not support overwrite=False ingestion"
                )
            if hasattr(target, "is_single_file") and target.is_single_file():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "overwrite=False isn't supported in single files. Please use folder path."
                )

    if spark_context and featureset.spec.engine != "spark":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "featureset.spec.engine must be set to 'spark' to ingest with spark"
        )
    if featureset.spec.engine == "spark":
        import pyspark.sql

        if (
            isinstance(source, (pd.DataFrame, pyspark.sql.DataFrame))
            and run_config is not None
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "DataFrame source is illegal when ingesting with remote spark or spark operator"
            )
        # use local spark session to ingest
        return _ingest_with_spark(
            spark_context,
            featureset,
            source,
            targets_to_ingest,
            infer_options=infer_options,
            mlrun_context=mlrun_context,
            namespace=namespace,
            overwrite=overwrite,
            return_df=return_df,
        )

    if isinstance(source, str):
        source = mlrun.store_manager.object(url=source).as_df()

    schema_options = InferOptions.get_common_options(
        infer_options, InferOptions.schema()
    )
    if schema_options:
        _preview(
            featureset,
            source,
            options=schema_options,
            namespace=namespace,
        )
    infer_stats = InferOptions.get_common_options(
        infer_options, InferOptions.all_stats()
    )
    # Check if dataframe is already calculated (for feature set graph):
    calculate_df = return_df or infer_stats != InferOptions.Null
    featureset.save()

    df = init_featureset_graph(
        source,
        featureset,
        namespace,
        targets=targets_to_ingest,
        return_df=calculate_df,
    )
    if not InferOptions.get_common_options(
        infer_stats, InferOptions.Index
    ) and InferOptions.get_common_options(infer_options, InferOptions.Index):
        infer_stats += InferOptions.Index

    _infer_from_static_df(df, featureset, options=infer_stats)

    if isinstance(source, DataSource):
        for target in featureset.status.targets:
            if (
                target.last_written == datetime.min
                and source.schedule
                and source.start_time
            ):
                # datetime.min is a special case that indicated that nothing was written in storey. we need the fix so
                # in the next scheduled run, we will have the same start time
                target.last_written = source.start_time

    _post_ingestion(mlrun_context, featureset, spark_context)
    if return_df:
        return df


def _preview(
    featureset: FeatureSet,
    source,
    entity_columns: Optional[list] = None,
    namespace=None,
    options: InferOptions = None,
    verbose: bool = False,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        source = _rename_source_dataframe_columns(source)

    # preview reads the source as a pandas df, which is not fully compatible with spark
    if featureset.spec.engine == "spark":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "preview with spark engine is not supported"
        )

    options = options if options is not None else InferOptions.default()

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    verify_feature_set_permissions(
        featureset, mlrun.common.schemas.AuthorizationAction.update
    )

    featureset.spec.validate_no_processing_for_passthrough()
    featureset.validate_steps(namespace=namespace)

    namespace = namespace or get_caller_globals()
    if featureset.spec.require_processing():
        _, default_final_step, _ = featureset.graph.check_and_process_graph(
            allow_empty=True
        )
        if not default_final_step:
            raise mlrun.errors.MLRunPreconditionFailedError(
                "Split flow graph must have a default final step defined"
            )
        # find/update entities schema
        if len(featureset.spec.entities) == 0:
            _infer_from_static_df(
                source,
                featureset,
                entity_columns,
                InferOptions.get_common_options(options, InferOptions.Entities),
            )
        # reduce the size of the ingestion if we do not infer stats
        rows_limit = (
            None
            if InferOptions.get_common_options(options, InferOptions.Stats)
            else 1000
        )
        source = init_featureset_graph(
            source,
            featureset,
            namespace,
            return_df=True,
            verbose=verbose,
            rows_limit=rows_limit,
        )

    df = _infer_from_static_df(
        source, featureset, entity_columns, options, sample_size=sample_size
    )
    featureset.save()
    return df


def _run_ingestion_job(
    featureset: Union[FeatureSet, str],
    source: DataSource = None,
    targets: Optional[list[DataTargetBase]] = None,
    name: Optional[str] = None,
    infer_options: InferOptions = InferOptions.default(),
    run_config: RunConfig = None,
):
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    run_config = run_config.copy() if run_config else RunConfig()
    source, run_config.parameters = set_task_params(
        featureset, source, targets, run_config.parameters, infer_options
    )

    return run_ingestion_job(name, featureset, run_config, source.schedule)


def _deploy_ingestion_service_v2(
    featureset: Union[FeatureSet, str],
    source: DataSource = None,
    targets: Optional[list[DataTargetBase]] = None,
    name: Optional[str] = None,
    run_config: RunConfig = None,
    verbose=False,
) -> tuple[str, BaseRuntime]:
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    verify_feature_set_permissions(
        featureset, mlrun.common.schemas.AuthorizationAction.update
    )

    verify_feature_set_exists(featureset)

    run_config = run_config.copy() if run_config else RunConfig()
    if isinstance(source, StreamSource) and not source.path:
        source.path = get_default_prefix_for_source(source.kind).format(
            project=featureset.metadata.project,
            kind=source.kind,
            name=featureset.metadata.name,
        )

    targets_to_ingest = targets or featureset.spec.targets
    targets_to_ingest = copy.deepcopy(targets_to_ingest)
    featureset.update_targets_for_ingest(targets_to_ingest)

    source, run_config.parameters = set_task_params(
        featureset, source, targets_to_ingest, run_config.parameters
    )

    name = normalize_name(name or f"{featureset.metadata.name}-ingest")
    if not run_config.function:
        function_ref = featureset.spec.function.copy()
        if function_ref.is_empty():
            function_ref = FunctionReference(name=name, kind=RuntimeKinds.serving)
        function_ref.kind = function_ref.kind or RuntimeKinds.serving
        if not function_ref.url:
            function_ref.code = function_ref.code or ""
        run_config.function = function_ref

    function = run_config.to_function(
        RuntimeKinds.serving, mlrun.mlconf.feature_store.default_job_image
    )
    function.metadata.project = featureset.metadata.project
    function.metadata.name = function.metadata.name or name

    function.spec.graph = featureset.spec.graph
    function.spec.graph.engine = (
        "async" if featureset.spec.engine == "storey" else "sync"
    )
    function.spec.parameters = run_config.parameters
    function.spec.graph_initializer = (
        "mlrun.feature_store.ingestion.featureset_initializer"
    )
    function.verbose = function.verbose or verbose
    function = source.add_nuclio_trigger(function)

    if run_config.local:
        return function.to_mock_server(namespace=get_caller_globals())
    return function.deploy(), function


def _ingest_with_spark(
    spark=None,
    featureset: Union[FeatureSet, str] = None,
    source: BaseSourceDriver = None,
    targets: Optional[list[BaseStoreTarget]] = None,
    infer_options: InferOptions = InferOptions.default(),
    mlrun_context=None,
    namespace=None,
    overwrite=None,
    return_df=None,
):
    created_spark_context = False
    try:
        import pyspark.sql

        from mlrun.datastore.spark_utils import check_special_columns_exists

        if spark is None or spark is True:
            # create spark context

            if mlrun_context:
                session_name = f"{mlrun_context.name}-{mlrun_context.uid}"
            else:
                session_name = (
                    f"{featureset.metadata.project}-{featureset.metadata.name}"
                )

            spark = (
                pyspark.sql.SparkSession.builder.appName(session_name)
                .config("spark.driver.memory", "2g")
                .config("spark.sql.session.timeZone", "UTC")
                .getOrCreate()
            )
            created_spark_context = True

        timestamp_key = featureset.spec.timestamp_key
        if isinstance(source, pd.DataFrame):
            df = spark.createDataFrame(source)
        elif isinstance(source, pyspark.sql.DataFrame):
            df = source
        else:
            df = source.to_spark_df(spark, time_field=timestamp_key)
        if featureset.spec.graph and featureset.spec.graph.steps:
            df = run_spark_graph(df, featureset, namespace, spark)

        if isinstance(df, Response) and df.status_code != 0:
            raise mlrun.errors.err_for_status_code(
                df.status_code, df.body.split(": ")[1]
            )

        df.persist()

        _infer_from_static_df(df, featureset, options=infer_options)

        key_columns = list(featureset.spec.entities.keys())
        targets = targets or featureset.spec.targets

        targets_to_ingest = copy.deepcopy(targets)
        featureset.update_targets_for_ingest(targets_to_ingest, overwrite=overwrite)

        for target in targets_to_ingest or []:
            if type(target) is DataTargetBase:
                target = get_target_driver(target, featureset)
            target.set_resource(featureset)
            if featureset.spec.passthrough and target.is_offline:
                check_special_columns_exists(
                    spark_df=df,
                    entities=featureset.spec.entities,
                    timestamp_key=timestamp_key,
                    label_column=featureset.spec.label_column,
                )
                continue
            spark_options = target.get_spark_options(
                key_columns, timestamp_key, overwrite
            )

            df_to_write = df
            df_to_write = target.prepare_spark_df(
                df_to_write, key_columns, timestamp_key, spark_options
            )
            write_format = spark_options.pop("format", None)
            # We can get to this point if the column exists in different letter cases,
            # so PySpark will be able to read it, but we still have to raise an exception for it.

            # This check is here and not in to_spark_df because in spark_merger we can have a target
            # that has different letter cases than the source, like in SnowflakeTarget.
            check_special_columns_exists(
                spark_df=df_to_write,
                entities=featureset.spec.entities,
                timestamp_key=timestamp_key,
                label_column=featureset.spec.label_column,
            )
            if overwrite:
                write_spark_dataframe_with_options(
                    spark_options, df_to_write, "overwrite", write_format=write_format
                )
            else:
                # appending an empty dataframe may cause an empty file to be created (e.g. when writing to parquet)
                # we would like to avoid that
                df_to_write.persist()
                if df_to_write.count() > 0:
                    write_spark_dataframe_with_options(
                        spark_options, df_to_write, "append", write_format=write_format
                    )
            target.update_resource_status("ready")

        if isinstance(source, BaseSourceDriver) and source.schedule:
            max_time = df.agg({timestamp_key: "max"}).collect()[0][0]
            if not max_time:
                # if max_time is None(no data), next scheduled run should be with same start_time
                max_time = source.start_time
            for target in featureset.status.targets:
                featureset.status.update_last_written_for_target(
                    target.get_path().get_absolute_path(
                        project_name=featureset.metadata.project
                    ),
                    max_time,
                )

        _post_ingestion(mlrun_context, featureset, spark)
    finally:
        if created_spark_context:
            spark.stop()
            # We shouldn't return a dataframe that depends on a stopped context
            df = None
    if return_df:
        return df


def _post_ingestion(context, featureset, spark=None):
    featureset.save()
    if context:
        context.logger.info("ingestion task completed, targets:")
        context.logger.info(f"{featureset.status.targets.to_dict()}")
        context.log_result("featureset", featureset.uri)


def _infer_from_static_df(
    df,
    featureset,
    entity_columns=None,
    options: InferOptions = InferOptions.default(),
    sample_size=None,
):
    """infer feature-set schema & stats from static dataframe (without pipeline)"""
    if hasattr(df, "to_dataframe"):
        if hasattr(df, "time_field"):
            time_field = df.time_field or featureset.spec.timestamp_key
        else:
            time_field = featureset.spec.timestamp_key
        if df.is_iterator():
            # todo: describe over multiple chunks
            df = next(df.to_dataframe(time_field=time_field))
        else:
            df = df.to_dataframe(time_field=time_field)
    inferer = get_infer_interface(df)
    if InferOptions.get_common_options(options, InferOptions.schema()):
        featureset.spec.timestamp_key = inferer.infer_schema(
            df,
            featureset.spec.features,
            featureset.spec.entities,
            featureset.spec.timestamp_key,
            entity_columns,
            options=options,
        )
    if InferOptions.get_common_options(options, InferOptions.Stats):
        featureset.status.stats = inferer.get_stats(
            df, options, sample_size=sample_size
        )
    if InferOptions.get_common_options(options, InferOptions.Preview):
        featureset.status.preview = inferer.get_preview(df)
    return df


def set_task_params(
    featureset: FeatureSet,
    source: DataSource = None,
    targets: Optional[list[DataTargetBase]] = None,
    parameters: Optional[dict] = None,
    infer_options: InferOptions = InferOptions.Null,
    overwrite=None,
):
    """convert ingestion parameters to dict, return source + params dict"""
    source = source or featureset.spec.source
    parameters = parameters or {}
    parameters["infer_options"] = infer_options
    parameters["overwrite"] = overwrite
    parameters["featureset"] = featureset.uri
    if source:
        parameters["source"] = source.to_dict()
    if targets:
        parameters["targets"] = [target.to_dict() for target in targets]
    elif not featureset.spec.targets:
        featureset.set_targets()
    featureset.save()
    return source, parameters


def get_feature_set(uri, project=None):
    """get feature set object from the db

    :param uri:  a feature set uri({project}/{name}[:version])
    :param project:  project name if not specified in uri or not using the current/default
    """
    return get_feature_set_by_uri(uri, project)


def get_feature_vector(uri, project=None):
    """get feature vector object from the db

    :param uri:  a feature vector uri({project}/{name}[:version])
    :param project:  project name if not specified in uri or not using the current/default
    """
    return get_feature_vector_by_uri(uri, project, update=False)


def delete_feature_set(name, project="", tag=None, uid=None, force=False):
    """Delete a :py:class:`~mlrun.feature_store.FeatureSet` object from the DB.

    :param name: Name of the object to delete
    :param project: Name of the object's project
    :param tag: Specific object's version tag
    :param uid: Specific object's uid
    :param force: Delete feature set without purging its targets

    If ``tag`` or ``uid`` are specified, then just the version referenced by them will be deleted. Using both
        is not allowed.
        If none are specified, then all instances of the object whose name is ``name`` will be deleted.
    """
    db = mlrun.get_run_db()
    if not force:
        feature_set = db.get_feature_set(name=name, project=project, tag=tag, uid=uid)
        if feature_set.status.targets:
            raise mlrun.errors.MLRunPreconditionFailedError(
                "delete_feature_set requires targets purging. Use either FeatureSet's purge_targets or the force flag."
            )
    return db.delete_feature_set(name=name, project=project, tag=tag, uid=uid)


def delete_feature_vector(name, project="", tag=None, uid=None):
    """Delete a :py:class:`~mlrun.feature_store.FeatureVector` object from the DB.

    :param name: Name of the object to delete
    :param project: Name of the object's project
    :param tag: Specific object's version tag
    :param uid: Specific object's uid

    If ``tag`` or ``uid`` are specified, then just the version referenced by them will be deleted. Using both
        is not allowed.
        If none are specified, then all instances of the object whose name is ``name`` will be deleted.
    """
    db = mlrun.get_run_db()
    return db.delete_feature_vector(name=name, project=project, tag=tag, uid=uid)
