# Copyright 2018 Iguazio
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
import warnings
from datetime import datetime
from typing import List, Optional, Union
from urllib.parse import urlparse

import pandas as pd

import mlrun
import mlrun.errors

from ..data_types import InferOptions, get_infer_interface
from ..datastore.sources import BaseSourceDriver, StreamSource
from ..datastore.store_resources import parse_store_uri
from ..datastore.targets import (
    BaseStoreTarget,
    get_default_prefix_for_source,
    get_default_targets,
    get_target_driver,
    kind_to_driver,
    validate_target_list,
    validate_target_paths_for_engine,
)
from ..db import RunDBError
from ..model import DataSource, DataTargetBase
from ..runtimes import RuntimeKinds
from ..runtimes.function_reference import FunctionReference
from ..serving.server import Response
from ..utils import get_caller_globals, logger, normalize_name, str_to_timestamp
from .common import (
    RunConfig,
    get_feature_set_by_uri,
    get_feature_vector_by_uri,
    verify_feature_set_exists,
    verify_feature_set_permissions,
    verify_feature_vector_permissions,
)
from .feature_set import FeatureSet
from .feature_vector import (
    FeatureVector,
    FixedWindowType,
    OfflineVectorResponse,
    OnlineVectorService,
)
from .ingestion import (
    context_to_ingestion_params,
    init_featureset_graph,
    run_ingestion_job,
    run_spark_graph,
)
from .retrieval import get_merger, init_feature_vector_graph, run_merge_job

_v3iofs = None
spark_transform_handler = "transform"


def _features_to_vector_and_check_permissions(features, update_stats):
    if isinstance(features, str):
        vector = get_feature_vector_by_uri(features, update=update_stats)
    elif isinstance(features, FeatureVector):
        vector = features
        if not vector.metadata.name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "feature vector name must be specified"
            )
        verify_feature_vector_permissions(
            vector, mlrun.api.schemas.AuthorizationAction.update
        )

        vector.save()
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"illegal features value/type ({type(features)})"
        )
    return vector


def get_offline_features(
    feature_vector: Union[str, FeatureVector],
    entity_rows=None,
    entity_timestamp_column: str = None,
    target: DataTargetBase = None,
    run_config: RunConfig = None,
    drop_columns: List[str] = None,
    start_time: Union[str, datetime] = None,
    end_time: Union[str, datetime] = None,
    with_indexes: bool = False,
    update_stats: bool = False,
    engine: str = None,
    engine_args: dict = None,
    query: str = None,
    join_type: str = "inner",
    spark_service: str = None,
) -> OfflineVectorResponse:
    """retrieve offline feature vector results

    specify a feature vector object/uri and retrieve the desired features, their metadata
    and statistics. returns :py:class:`~mlrun.feature_store.OfflineVectorResponse`,
    results can be returned as a dataframe or written to a target

    The start_time and end_time attributes allow filtering the data to a given time range, they accept
    string values or pandas `Timestamp` objects, string values can also be relative, for example:
    "now", "now - 1d2h", "now+5m", where a valid pandas Timedelta string follows the verb "now",
    for time alignment you can use the verb "floor" e.g. "now -1d floor 1H" will align the time to the last hour
    (the floor string is passed to pandas.Timestamp.floor(), can use D, H, T, S for day, hour, min, sec alignment).
    Another option to filter the data is by the `query` argument - can be seen in the example.
    example::

        features = [
            "stock-quotes.bid",
            "stock-quotes.asks_sum_5h",
            "stock-quotes.ask as mycol",
            "stocks.*",
        ]
        vector = FeatureVector(features=features)
        resp = get_offline_features(
            vector, entity_rows=trades, entity_timestamp_column="time", query="ticker in ['GOOG'] and bid>100"
        )
        print(resp.to_dataframe())
        print(vector.get_stats_table())
        resp.to_parquet("./out.parquet")

    :param feature_vector: feature vector uri or FeatureVector object. passing feature vector obj requires update
                            permissions
    :param entity_rows:    dataframe with entity rows to join with
    :param target:         where to write the results to
    :param drop_columns:   list of columns to drop from the final result
    :param entity_timestamp_column: timestamp column name in the entity rows dataframe
    :param run_config:     function and/or run configuration
                           see :py:class:`~mlrun.feature_store.RunConfig`
    :param start_time:      datetime, low limit of time needed to be filtered. Optional.
        entity_timestamp_column must be passed when using time filtering.
    :param end_time:        datetime, high limit of time needed to be filtered. Optional.
        entity_timestamp_column must be passed when using time filtering.
    :param with_indexes:    return vector with index columns and timestamp_key from the feature sets (default False)
    :param update_stats:    update features statistics from the requested feature sets on the vector. Default is False.
    :param engine:          processing engine kind ("local", "dask", or "spark")
    :param engine_args:     kwargs for the processing engine
    :param query:           The query string used to filter rows
    :param spark_service:   Name of the spark service to be used (when using a remote-spark runtime)
    :param join_type:               {'left', 'right', 'outer', 'inner'}, default 'inner'
                                    Supported retrieval engines: "dask", "local"
                                    This parameter is in use when entity_timestamp_column and
                                    feature_vector.spec.timestamp_field are None, if one of them
                                    isn't none we're preforming as_of join.
                                    Possible values :
                                    * left: use only keys from left frame (SQL: left outer join)
                                    * right: use only keys from right frame (SQL: right outer join)
                                    * outer: use union of keys from both frames (SQL: full outer join)
                                    * inner: use intersection of keys from both frames (SQL: inner join).
    """
    if isinstance(feature_vector, FeatureVector):
        update_stats = True

    feature_vector = _features_to_vector_and_check_permissions(
        feature_vector, update_stats
    )

    entity_timestamp_column = (
        entity_timestamp_column or feature_vector.spec.timestamp_field
    )

    merger_engine = get_merger(engine)

    if run_config and not run_config.local:
        return run_merge_job(
            feature_vector,
            target,
            merger_engine,
            engine,
            engine_args,
            spark_service,
            entity_rows,
            timestamp_column=entity_timestamp_column,
            run_config=run_config,
            drop_columns=drop_columns,
            with_indexes=with_indexes,
            query=query,
            join_type=join_type,
        )

    start_time = str_to_timestamp(start_time)
    end_time = str_to_timestamp(end_time)
    if (start_time or end_time) and not entity_timestamp_column:
        raise TypeError(
            "entity_timestamp_column or feature_vector.spec.timestamp_field is required when passing start/end time"
        )
    if start_time and not end_time:
        # if end_time is not specified set it to now()
        end_time = pd.Timestamp.now()
    merger = merger_engine(feature_vector, **(engine_args or {}))
    return merger.start(
        entity_rows,
        entity_timestamp_column,
        target=target,
        drop_columns=drop_columns,
        start_time=start_time,
        end_time=end_time,
        with_indexes=with_indexes,
        update_stats=update_stats,
        query=query,
        join_type=join_type,
    )


def get_online_feature_service(
    feature_vector: Union[str, FeatureVector],
    run_config: RunConfig = None,
    fixed_window_type: FixedWindowType = FixedWindowType.LastClosedWindow,
    impute_policy: dict = None,
    update_stats: bool = False,
) -> OnlineVectorService:
    """initialize and return online feature vector service api,
    returns :py:class:`~mlrun.feature_store.OnlineVectorService`

    :**usage**:
        There are two ways to use the function:

        1. As context manager

            Example::

                with get_online_feature_service(vector_uri) as svc:
                    resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
                    print(resp)
                    resp = svc.get([{"ticker": "AAPL"}], as_list=True)
                    print(resp)

            Example with imputing::

                with get_online_feature_service(vector_uri, impute_policy={"*": "$mean", "amount": 0)) as svc:
                    resp = svc.get([{"id": "C123487"}])

        2. as simple function, note that in that option you need to close the session.

            Example::

                svc = get_online_feature_service(vector_uri)
                try:
                    resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
                    print(resp)
                    resp = svc.get([{"ticker": "AAPL"}], as_list=True)
                    print(resp)

                finally:
                    svc.close()

            Example with imputing::

                svc = get_online_feature_service(vector_uri, impute_policy={"*": "$mean", "amount": 0))
                try:
                    resp = svc.get([{"id": "C123487"}])
                except Exception as e:
                    handling exception...
                finally:
                    svc.close()

    :param feature_vector:    feature vector uri or FeatureVector object. passing feature vector obj requires update
                            permissions
    :param run_config:        function and/or run configuration for remote jobs/services
    :param impute_policy:     a dict with `impute_policy` per feature, the dict key is the feature name and the dict
                            value indicate which value will be used in case the feature is NaN/empty, the replaced
                            value can be fixed number for constants or $mean, $max, $min, $std, $count for statistical
                            values. "*" is used to specify the default for all features, example: `{"*": "$mean"}`
    :param fixed_window_type: determines how to query the fixed window values which were previously inserted by ingest
    :param update_stats:      update features statistics from the requested feature sets on the vector. Default: False.
    """
    if isinstance(feature_vector, FeatureVector):
        update_stats = True
    feature_vector = _features_to_vector_and_check_permissions(
        feature_vector, update_stats
    )

    # Impute policies rely on statistics in many cases, so verifying that the fvec has stats in it
    if impute_policy and not feature_vector.status.stats:
        update_stats = True

    graph, index_columns = init_feature_vector_graph(
        feature_vector, fixed_window_type, update_stats=update_stats
    )
    service = OnlineVectorService(
        feature_vector, graph, index_columns, impute_policy=impute_policy
    )
    service.initialize()

    # todo: support remote service (using remote nuclio/mlrun function if run_config)

    return service


def _rename_source_dataframe_columns(df):
    rename_mapping = {}
    column_set = set(df.columns)
    for column in df.columns:
        if isinstance(column, str):
            rename_to = column.replace(" ", "_").replace("(", "").replace(")", "")
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


def ingest(
    featureset: Union[FeatureSet, str] = None,
    source=None,
    targets: List[DataTargetBase] = None,
    namespace=None,
    return_df: bool = True,
    infer_options: InferOptions = InferOptions.default(),
    run_config: RunConfig = None,
    mlrun_context=None,
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
        config = RunConfig(image='mlrun/mlrun')
        df = ingest(stocks_set, stocks, run_config=config)

        # specify source and targets
        source = CSVSource("mycsv", path="measurements.csv")
        targets = [CSVTarget("mycsv", path="./mycsv.csv")]
        ingest(measurements, source, targets)

    :param featureset:    feature set object or featureset.uri. (uri must be of a feature set that is in the DB,
                          call `.save()` if it's not)
    :param source:        source dataframe or other sources (e.g. parquet source see:
                          :py:class:`~mlrun.datastore.ParquetSource` and other classes in mlrun.datastore with suffix
                          Source)
    :param targets:       optional list of data target objects
    :param namespace:     namespace or module containing graph classes
    :param return_df:     indicate if to return a dataframe with the graph results
    :param infer_options: schema and stats infer options
    :param run_config:    function and/or run configuration for remote jobs,
                          see :py:class:`~mlrun.feature_store.RunConfig`
    :param mlrun_context: mlrun context (when running as a job), for internal use !
    :param spark_context: local spark session for spark ingestion, example for creating the spark context:
                          `spark = SparkSession.builder.appName("Spark function").getOrCreate()`
                          For remote spark ingestion, this should contain the remote spark service name
    :param overwrite:     delete the targets' data prior to ingestion
                          (default: True for non scheduled ingest - deletes the targets that are about to be ingested.
                          False for scheduled ingest - does not delete the target)
    :return:              if return_df is True, a dataframe will be returned based on the graph
    """
    if isinstance(source, pd.DataFrame):
        source = _rename_source_dataframe_columns(source)

    if featureset:
        if isinstance(featureset, str):
            # need to strip store prefix from the uri
            _, stripped_name = parse_store_uri(featureset)
            try:
                featureset = get_feature_set_by_uri(stripped_name)
            except RunDBError as exc:
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

    # This flow may happen both on client side (user provides run config) and server side (through the ingest API)
    if run_config:
        if isinstance(source, pd.DataFrame):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "DataFrame source is illegal in conjunction with run_config"
            )
        # remote job execution
        verify_feature_set_permissions(
            featureset, mlrun.api.schemas.AuthorizationAction.update
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

        verify_feature_set_permissions(
            featureset, mlrun.api.schemas.AuthorizationAction.update
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

    namespace = namespace or get_caller_globals()

    targets_to_ingest = targets or featureset.spec.targets or get_default_targets()
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
        preview(
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


def preview(
    featureset: FeatureSet,
    source,
    entity_columns: list = None,
    timestamp_key: str = None,
    namespace=None,
    options: InferOptions = None,
    verbose: bool = False,
    sample_size: int = None,
) -> pd.DataFrame:
    """run the ingestion pipeline with local DataFrame/file data and infer features schema and stats

    example::

        quotes_set = FeatureSet("stock-quotes", entities=[Entity("ticker")])
        quotes_set.add_aggregation("ask", ["sum", "max"], ["1h", "5h"], "10m")
        quotes_set.add_aggregation("bid", ["min", "max"], ["1h"], "10m")
        df = preview(
            quotes_set,
            quotes_df,
            entity_columns=["ticker"],
            timestamp_key="time",
        )

    :param featureset:     feature set object or uri
    :param source:         source dataframe or csv/parquet file path
    :param entity_columns: list of entity (index) column names
    :param timestamp_key:  DEPRECATED. Use FeatureSet parameter.
    :param namespace:      namespace or module containing graph classes
    :param options:        schema and stats infer options (:py:class:`~mlrun.feature_store.InferOptions`)
    :param verbose:        verbose log
    :param sample_size:    num of rows to sample from the dataset (for large datasets)
    """
    # preview reads the source as a pandas df, which is not fully compatible with spark
    if featureset.spec.engine == "spark":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "preview with spark engine is not supported"
        )

    options = options if options is not None else InferOptions.default()
    if timestamp_key is not None:
        warnings.warn(
            "preview's 'timestamp_key' parameter is deprecated in 1.3.0 and will be removed in 1.5.0. "
            "Pass this parameter to 'FeatureSet' instead.",
            # TODO: Remove this API in 1.5.0
            FutureWarning,
        )
        featureset.spec.timestamp_key = timestamp_key
        for step in featureset.graph.steps.values():
            if step.class_name == "storey.AggregateByKey":
                step.class_args["time_field"] = timestamp_key

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    verify_feature_set_permissions(
        featureset, mlrun.api.schemas.AuthorizationAction.update
    )

    featureset.spec.validate_no_processing_for_passthrough()

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
            0 if InferOptions.get_common_options(options, InferOptions.Stats) else 1000
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
    targets: List[DataTargetBase] = None,
    name: str = None,
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


def deploy_ingestion_service(
    featureset: Union[FeatureSet, str],
    source: DataSource = None,
    targets: List[DataTargetBase] = None,
    name: str = None,
    run_config: RunConfig = None,
    verbose=False,
):
    """Start real-time ingestion service using nuclio function

    Deploy a real-time function implementing feature ingestion pipeline
    the source maps to Nuclio event triggers (http, kafka, v3io stream, etc.)

    the `run_config` parameter allow specifying the function and job configuration,
    see: :py:class:`~mlrun.feature_store.RunConfig`

    example::

        source = HTTPSource()
        func = mlrun.code_to_function("ingest", kind="serving").apply(mount_v3io())
        config = RunConfig(function=func)
        fstore.deploy_ingestion_service(my_set, source, run_config=config)

    :param featureset:    feature set object or uri
    :param source:        data source object describing the online or offline source
    :param targets:       list of data target objects
    :param name:          name for the job/function
    :param run_config:    service runtime configuration (function object/uri, resources, etc..)
    :param verbose:       verbose log
    """
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    verify_feature_set_permissions(
        featureset, mlrun.api.schemas.AuthorizationAction.update
    )

    verify_feature_set_exists(featureset)

    run_config = run_config.copy() if run_config else RunConfig()
    if isinstance(source, StreamSource) and not source.path:
        source.path = get_default_prefix_for_source(source.kind).format(
            project=featureset.metadata.project,
            kind=source.kind,
            name=featureset.metadata.name,
        )

    targets_to_ingest = targets or featureset.spec.targets or get_default_targets()
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
    function.spec.parameters = run_config.parameters
    function.spec.graph_initializer = (
        "mlrun.feature_store.ingestion.featureset_initializer"
    )
    function.verbose = function.verbose or verbose
    function = source.add_nuclio_trigger(function)

    if run_config.local:
        return function.to_mock_server(namespace=get_caller_globals())
    return function.deploy()


def _ingest_with_spark(
    spark=None,
    featureset: Union[FeatureSet, str] = None,
    source: BaseSourceDriver = None,
    targets: List[BaseStoreTarget] = None,
    infer_options: InferOptions = InferOptions.default(),
    mlrun_context=None,
    namespace=None,
    overwrite=None,
    return_df=None,
):
    created_spark_context = False
    try:
        import pyspark.sql

        if spark is None or spark is True:
            # create spark context

            if mlrun_context:
                session_name = f"{mlrun_context.name}-{mlrun_context.uid}"
            else:
                session_name = (
                    f"{featureset.metadata.project}-{featureset.metadata.name}"
                )

            spark = pyspark.sql.SparkSession.builder.appName(session_name).getOrCreate()
            created_spark_context = True

        timestamp_key = featureset.spec.timestamp_key

        if isinstance(source, pd.DataFrame):
            df = spark.createDataFrame(source)
        elif isinstance(source, pyspark.sql.DataFrame):
            df = source
        else:
            df = source.to_spark_df(spark, time_field=timestamp_key)
            df = source.filter_df_start_end_time(df, timestamp_key)
        if featureset.spec.graph and featureset.spec.graph.steps:
            df = run_spark_graph(df, featureset, namespace, spark)

        if isinstance(df, Response) and df.status_code != 0:
            mlrun.errors.raise_for_status_code(df.status_code, df.body.split(": ")[1])
        _infer_from_static_df(df, featureset, options=infer_options)

        key_columns = list(featureset.spec.entities.keys())
        targets = targets or featureset.spec.targets

        targets_to_ingest = copy.deepcopy(targets)
        featureset.update_targets_for_ingest(targets_to_ingest, overwrite=overwrite)

        for target in targets_to_ingest or []:
            if type(target) is DataTargetBase:
                target = get_target_driver(target, featureset)
            if featureset.spec.passthrough and target.is_offline:
                continue
            if target.path and urlparse(target.path).scheme == "":
                if mlrun_context:
                    mlrun_context.logger.error(
                        "Paths for spark ingest must contain schema, i.e v3io, s3, az"
                    )
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Paths for spark ingest must contain schema, i.e v3io, s3, az"
                )
            spark_options = target.get_spark_options(
                key_columns, timestamp_key, overwrite
            )
            logger.info(
                f"writing to target {target.name}, spark options {spark_options}"
            )

            df_to_write = df

            # If partitioning by time, add the necessary columns
            if timestamp_key and "partitionBy" in spark_options:
                from pyspark.sql.functions import (
                    dayofmonth,
                    hour,
                    minute,
                    month,
                    second,
                    year,
                )

                time_unit_to_op = {
                    "year": year,
                    "month": month,
                    "day": dayofmonth,
                    "hour": hour,
                    "minute": minute,
                    "second": second,
                }
                timestamp_col = df_to_write[timestamp_key]
                for partition in spark_options["partitionBy"]:
                    if (
                        partition not in df_to_write.columns
                        and partition in time_unit_to_op
                    ):
                        op = time_unit_to_op[partition]
                        df_to_write = df_to_write.withColumn(
                            partition, op(timestamp_col)
                        )
            df_to_write = target.prepare_spark_df(df_to_write, key_columns)
            if overwrite:
                df_to_write.write.mode("overwrite").save(**spark_options)
            else:
                # appending an empty dataframe may cause an empty file to be created (e.g. when writing to parquet)
                # we would like to avoid that
                df_to_write.persist()
                if df_to_write.count() > 0:
                    df_to_write.write.mode("append").save(**spark_options)
            target.set_resource(featureset)
            target.update_resource_status("ready")

        if isinstance(source, BaseSourceDriver) and source.schedule:
            max_time = df.agg({timestamp_key: "max"}).collect()[0][0]
            if not max_time:
                # if max_time is None(no data), next scheduled run should be with same start_time
                max_time = source.start_time
            for target in featureset.status.targets:
                featureset.status.update_last_written_for_target(
                    target.get_path().get_absolute_path(), max_time
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
        if df.is_iterator():
            # todo: describe over multiple chunks
            df = next(df.to_dataframe())
        else:
            df = df.to_dataframe()
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
    targets: List[DataTargetBase] = None,
    parameters: dict = None,
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
