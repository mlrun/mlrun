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
from deprecated import deprecated

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
from .retrieval import RemoteVectorResponse, get_merger, run_merge_job

_v3iofs = None
spark_transform_handler = "transform"
_TRANS_TABLE = str.maketrans({" ": "_", "(": "", ")": ""})


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
            vector, mlrun.common.schemas.AuthorizationAction.update
        )

        vector.save()
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"illegal features value/type ({type(features)})"
        )
    return vector


@deprecated(
    version="1.6.0",
    reason="get_offline_features() will be removed in 1.8.0, please instead use "
    "get_feature_vector('store://feature_vector_name').get_offline_features()",
    category=FutureWarning,
)
def get_offline_features(
    feature_vector: Union[str, FeatureVector],
    entity_rows=None,
    entity_timestamp_column: str = None,
    target: DataTargetBase = None,
    run_config: RunConfig = None,
    drop_columns: list[str] = None,
    start_time: Union[str, datetime] = None,
    end_time: Union[str, datetime] = None,
    with_indexes: bool = False,
    update_stats: bool = False,
    engine: str = None,
    engine_args: dict = None,
    query: str = None,
    order_by: Union[str, list[str]] = None,
    spark_service: str = None,
    timestamp_for_filtering: Union[str, dict[str, str]] = None,
    additional_filters: list = None,
):
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
            vector,
            entity_rows=trades,
            entity_timestamp_column="time",
            query="ticker in ['GOOG'] and bid>100",
        )
        print(resp.to_dataframe())
        print(vector.get_stats_table())
        resp.to_parquet("./out.parquet")

    :param feature_vector:          feature vector uri or FeatureVector object. passing feature vector obj requires
                                    update permissions
    :param entity_rows:             dataframe with entity rows to join with
    :param target:                  where to write the results to
    :param drop_columns:            list of columns to drop from the final result
    :param entity_timestamp_column: timestamp column name in the entity rows dataframe. can be specified
                                    only if param entity_rows was specified.
    :param run_config:              function and/or run configuration
                                    see :py:class:`~mlrun.feature_store.RunConfig`
    :param start_time:              datetime, low limit of time needed to be filtered. Optional.
    :param end_time:                datetime, high limit of time needed to be filtered. Optional.
    :param with_indexes:            Return vector with/without the entities and the timestamp_key of the feature sets
                                    and with/without entity_timestamp_column and timestamp_for_filtering columns.
                                    This property can be specified also in the feature vector spec
                                    (feature_vector.spec.with_indexes)
                                    (default False)
    :param update_stats:            update features statistics from the requested feature sets on the vector.
                                    (default False).
    :param engine:                  processing engine kind ("local", "dask", or "spark")
    :param engine_args:             kwargs for the processing engine
    :param query:                   The query string used to filter rows on the output
    :param spark_service:           Name of the spark service to be used (when using a remote-spark runtime)
    :param order_by:                Name or list of names to order by. The name or the names in the list can be the
                                    feature name or the alias of the feature you pass in the feature list.
    :param timestamp_for_filtering: name of the column to filter by, can be str for all the feature sets or a
                                    dictionary ({<feature set name>: <timestamp column name>, ...})
                                    that indicates the timestamp column name for each feature set. Optional.
                                    By default, the filter executes on the timestamp_key of each feature set.
                                    Note: the time filtering is performed on each feature set before the
                                    merge process using start_time and end_time params.
    :param additional_filters: List of additional_filter conditions as tuples.
                                Each tuple should be in the format (column_name, operator, value).
                                Supported operators: "=", ">=", "<=", ">", "<".
                                Example: [("Product", "=", "Computer")]
                                For all supported filters, please see:
                                https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html


    """
    return _get_offline_features(
        feature_vector,
        entity_rows,
        entity_timestamp_column,
        target,
        run_config,
        drop_columns,
        start_time,
        end_time,
        with_indexes,
        update_stats,
        engine,
        engine_args,
        query,
        order_by,
        spark_service,
        timestamp_for_filtering,
        additional_filters,
    )


def _get_offline_features(
    feature_vector: Union[str, FeatureVector],
    entity_rows=None,
    entity_timestamp_column: str = None,
    target: DataTargetBase = None,
    run_config: RunConfig = None,
    drop_columns: list[str] = None,
    start_time: Union[str, datetime] = None,
    end_time: Union[str, datetime] = None,
    with_indexes: bool = False,
    update_stats: bool = False,
    engine: str = None,
    engine_args: dict = None,
    query: str = None,
    order_by: Union[str, list[str]] = None,
    spark_service: str = None,
    timestamp_for_filtering: Union[str, dict[str, str]] = None,
    additional_filters=None,
) -> Union[OfflineVectorResponse, RemoteVectorResponse]:
    if entity_rows is None and entity_timestamp_column is not None:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "entity_timestamp_column param "
            "can not be specified without entity_rows param"
        )
    if isinstance(target, BaseStoreTarget) and not target.support_pandas:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"get_offline_features does not support targets that do not support pandas engine."
            f" Target kind: {target.kind}"
        )

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
            entity_timestamp_column=entity_timestamp_column,
            run_config=run_config,
            drop_columns=drop_columns,
            with_indexes=with_indexes,
            query=query,
            order_by=order_by,
            start_time=start_time,
            end_time=end_time,
            timestamp_for_filtering=timestamp_for_filtering,
            additional_filters=additional_filters,
        )

    merger = merger_engine(feature_vector, **(engine_args or {}))
    return merger.start(
        entity_rows,
        entity_timestamp_column,
        target=target,
        drop_columns=drop_columns,
        start_time=start_time,
        end_time=end_time,
        timestamp_for_filtering=timestamp_for_filtering,
        with_indexes=with_indexes,
        update_stats=update_stats,
        query=query,
        order_by=order_by,
        additional_filters=additional_filters,
    )


@deprecated(
    version="1.6.0",
    reason="get_online_feature_service() will be removed in 1.8.0, please instead use "
    "get_feature_vector('store://feature_vector_name').get_online_feature_service()",
    category=FutureWarning,
)
def get_online_feature_service(
    feature_vector: Union[str, FeatureVector],
    run_config: RunConfig = None,
    fixed_window_type: FixedWindowType = FixedWindowType.LastClosedWindow,
    impute_policy: dict = None,
    update_stats: bool = False,
    entity_keys: list[str] = None,
):
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

                with get_online_feature_service(vector_uri, entity_keys=['id'],
                                                impute_policy={"*": "$mean", "amount": 0)) as svc:
                    resp = svc.get([{"id": "C123487"}])

        2. as simple function, note that in that option you need to close the session.

            Example::

                svc = get_online_feature_service(vector_uri, entity_keys=["ticker"])
                try:
                    resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
                    print(resp)
                    resp = svc.get([{"ticker": "AAPL"}], as_list=True)
                    print(resp)

                finally:
                    svc.close()

            Example with imputing::

                svc = get_online_feature_service(vector_uri, entity_keys=['id'],
                                                 impute_policy={"*": "$mean", "amount": 0))
                try:
                    resp = svc.get([{"id": "C123487"}])
                except Exception as e:
                    handling exception...
                finally:
                    svc.close()

    :param feature_vector:      feature vector uri or FeatureVector object. passing feature vector obj requires update
                                permissions.
    :param run_config:          function and/or run configuration for remote jobs/services
    :param impute_policy:       a dict with `impute_policy` per feature, the dict key is the feature name and the dict
                                value indicate which value will be used in case the feature is NaN/empty, the replaced
                                value can be fixed number for constants or $mean, $max, $min, $std, $count
                                for statistical
                                values. "*" is used to specify the default for all features, example: `{"*": "$mean"}`
    :param fixed_window_type:   determines how to query the fixed window values which were previously inserted by ingest
    :param update_stats:        update features statistics from the requested feature sets on the vector.
                                Default: False.
    :param entity_keys:         Entity list of the first feature_set in the vector.
                                The indexes that are used to query the online service.
    :return:                    Initialize the `OnlineVectorService`.
                                Will be used in subclasses where `support_online=True`.
    """
    return _get_online_feature_service(
        feature_vector,
        run_config,
        fixed_window_type,
        impute_policy,
        update_stats,
        entity_keys,
    )


def _get_online_feature_service(
    feature_vector: Union[str, FeatureVector],
    run_config: RunConfig = None,
    fixed_window_type: FixedWindowType = FixedWindowType.LastClosedWindow,
    impute_policy: dict = None,
    update_stats: bool = False,
    entity_keys: list[str] = None,
) -> OnlineVectorService:
    if isinstance(feature_vector, FeatureVector):
        update_stats = True
    feature_vector = _features_to_vector_and_check_permissions(
        feature_vector, update_stats
    )

    # Impute policies rely on statistics in many cases, so verifying that the fvec has stats in it
    if impute_policy and not feature_vector.status.stats:
        update_stats = True

    engine_args = {"impute_policy": impute_policy}
    merger_engine = get_merger("storey")
    # todo: support remote service (using remote nuclio/mlrun function if run_config)

    merger = merger_engine(feature_vector, **engine_args)

    return merger.init_online_vector_service(
        entity_keys, fixed_window_type, update_stats=update_stats
    )


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
    featureset: Union[FeatureSet, str] = None,
    source=None,
    targets: list[DataTargetBase] = None,
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
        config = RunConfig(image="mlrun/mlrun")
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
    :param infer_options: schema (for discovery of entities, features in featureset), index, stats,
                          histogram and preview infer options (:py:class:`~mlrun.feature_store.InferOptions`)
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
    if mlrun_context is None:
        deprecated(
            version="1.6.0",
            reason="Calling 'ingest' with mlrun_context=None is deprecated and will be removed in 1.8.0,\
            use 'FeatureSet.ingest()' instead",
            category=FutureWarning,
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
    targets: list[DataTargetBase] = None,
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


@deprecated(
    version="1.6.0",
    reason="'preview' will be removed in 1.8.0, use 'FeatureSet.preview()' instead",
    category=FutureWarning,
)
def preview(
    featureset: FeatureSet,
    source,
    entity_columns: list = None,
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
        )

    :param featureset:     feature set object or uri
    :param source:         source dataframe or csv/parquet file path
    :param entity_columns: list of entity (index) column names
    :param namespace:      namespace or module containing graph classes
    :param options:        schema (for discovery of entities, features in featureset), index, stats,
                           histogram and preview infer options (:py:class:`~mlrun.feature_store.InferOptions`)
    :param verbose:        verbose log
    :param sample_size:    num of rows to sample from the dataset (for large datasets)
    """
    return _preview(
        featureset,
        source,
        entity_columns,
        namespace,
        options,
        verbose,
        sample_size,
    )


def _preview(
    featureset: FeatureSet,
    source,
    entity_columns: list = None,
    namespace=None,
    options: InferOptions = None,
    verbose: bool = False,
    sample_size: int = None,
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
    targets: list[DataTargetBase] = None,
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


@deprecated(
    version="1.6.0",
    reason="'deploy_ingestion_service_v2' will be removed in 1.8.0, "
    "use 'FeatureSet.deploy_ingestion_service()' instead",
    category=FutureWarning,
)
def deploy_ingestion_service_v2(
    featureset: Union[FeatureSet, str],
    source: DataSource = None,
    targets: list[DataTargetBase] = None,
    name: str = None,
    run_config: RunConfig = None,
    verbose=False,
) -> tuple[str, BaseRuntime]:
    """Start real-time ingestion service using nuclio function

    Deploy a real-time function implementing feature ingestion pipeline
    the source maps to Nuclio event triggers (http, kafka, v3io stream, etc.)

    the `run_config` parameter allow specifying the function and job configuration,
    see: :py:class:`~mlrun.feature_store.RunConfig`

    example::

        source = HTTPSource()
        func = mlrun.code_to_function("ingest", kind="serving").apply(mount_v3io())
        config = RunConfig(function=func)
        deploy_ingestion_service_v2(my_set, source, run_config=config)

    :param featureset:    feature set object or uri
    :param source:        data source object describing the online or offline source
    :param targets:       list of data target objects
    :param name:          name for the job/function
    :param run_config:    service runtime configuration (function object/uri, resources, etc..)
    :param verbose:       verbose log

    :return: URL to access the deployed ingestion service, and the function that was deployed (which will
             differ from the function passed in via the run_config parameter).
    """
    return _deploy_ingestion_service_v2(
        featureset,
        source,
        targets,
        name,
        run_config,
        verbose,
    )


def _deploy_ingestion_service_v2(
    featureset: Union[FeatureSet, str],
    source: DataSource = None,
    targets: list[DataTargetBase] = None,
    name: str = None,
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
    targets: list[BaseStoreTarget] = None,
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
    targets: list[DataTargetBase] = None,
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
