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
from typing import List, Optional, Union
from urllib.parse import urlparse

import pandas as pd

import mlrun
import mlrun.errors

from ..data_types import InferOptions, get_infer_interface
from ..datastore.store_resources import parse_store_uri
from ..datastore.targets import (
    TargetTypes,
    get_default_targets,
    get_target_driver,
    validate_target_list,
)
from ..db import RunDBError
from ..model import DataSource, DataTargetBase
from ..runtimes import RuntimeKinds
from ..runtimes.function_reference import FunctionReference
from ..utils import get_caller_globals, logger
from .common import RunConfig, get_feature_set_by_uri, get_feature_vector_by_uri
from .feature_set import FeatureSet
from .feature_vector import FeatureVector, OfflineVectorResponse, OnlineVectorService
from .ingestion import (
    context_to_ingestion_params,
    init_featureset_graph,
    run_ingestion_job,
    run_spark_graph,
)
from .retrieval import LocalFeatureMerger, init_feature_vector_graph, run_merge_job

_v3iofs = None
spark_transform_handler = "transform"


def _features_to_vector(features):
    if isinstance(features, str):
        vector = get_feature_vector_by_uri(features)
    elif isinstance(features, FeatureVector):
        vector = features
        if not vector.metadata.name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "feature vector name must be specified"
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
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> OfflineVectorResponse:
    """retrieve offline feature vector results

    specify a feature vector object/uri and retrieve the desired features, their metadata
    and statistics. returns :py:class:`~mlrun.feature_store.OfflineVectorResponse`,
    results can be returned as a dataframe or written to a target

    example::

        features = [
            "stock-quotes.bid",
            "stock-quotes.asks_sum_5h",
            "stock-quotes.ask as mycol",
            "stocks.*",
        ]
        vector = FeatureVector(features=features)
        resp = get_offline_features(
            vector, entity_rows=trades, entity_timestamp_column="time"
        )
        print(resp.to_dataframe())
        print(vector.get_stats_table())
        resp.to_parquet("./out.parquet")

    :param feature_vector: feature vector uri or FeatureVector object
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
    """
    feature_vector = _features_to_vector(feature_vector)

    entity_timestamp_column = (
        entity_timestamp_column or feature_vector.spec.timestamp_field
    )
    if run_config:
        return run_merge_job(
            feature_vector,
            target,
            entity_rows,
            timestamp_column=entity_timestamp_column,
            run_config=run_config,
            drop_columns=drop_columns,
        )

    if (start_time or end_time) and not entity_timestamp_column:
        raise TypeError(
            "entity_timestamp_column or feature_vector.spec.timestamp_field is required when passing start/end time"
        )
    merger = LocalFeatureMerger(feature_vector)
    return merger.start(
        entity_rows,
        entity_timestamp_column,
        target=target,
        drop_columns=drop_columns,
        start_time=start_time,
        end_time=end_time,
    )


def get_online_feature_service(
    feature_vector: Union[str, FeatureVector], run_config: RunConfig = None,
) -> OnlineVectorService:
    """initialize and return online feature vector service api,
    returns :py:class:`~mlrun.feature_store.OnlineVectorService`

    example::

        svc = get_online_feature_service(vector_uri)
        resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
        print(resp)
        resp = svc.get([{"ticker": "AAPL"}], as_list=True)
        print(resp)

    :param feature_vector:  feature vector uri or FeatureVector object
    :param run_config:   function and/or run configuration for remote jobs/services
    """
    feature_vector = _features_to_vector(feature_vector)
    graph, index_columns = init_feature_vector_graph(feature_vector)
    service = OnlineVectorService(feature_vector, graph, index_columns)

    # todo: support remote service (using remote nuclio/mlrun function if run_config)
    return service


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
    overwrite=True,
) -> pd.DataFrame:
    """Read local DataFrame, file, URL, or source into the feature store
    Ingest reads from the source, run the graph transformations, infers  metadata and stats
    and writes the results to the default of specified targets

    when targets are not specified data is stored in the configured default targets
    (will usually be NoSQL for real-time and Parquet for offline).

    example::

        stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
        stocks = pd.read_csv("stocks.csv")
        df = ingest(stocks_set, stocks, infer_options=fstore.InferOptions.default())

        # for running as remote job
        config = RunConfig(image='mlrun/mlrun').apply(mount_v3io())
        df = ingest(stocks_set, stocks, run_config=config)

        # specify source and targets
        source = CSVSource("mycsv", path="measurements.csv")
        targets = [CSVTarget("mycsv", path="./mycsv.csv")]
        ingest(measurements, source, targets)

    :param featureset:    feature set object or featureset.uri. (uri must be of a feature set that is in the DB,
                          call `.save()` if it's not)
    :param source:        source dataframe or file path
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
                          (default: True. deletes the targets that are about to be ingested)
    """
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

    if run_config:
        # remote job execution
        run_config = run_config.copy() if run_config else RunConfig()
        source, run_config.parameters = set_task_params(
            featureset, source, targets, run_config.parameters, infer_options
        )
        name = f"{featureset.metadata.name}_ingest"
        return run_ingestion_job(
            name, featureset, run_config, source.schedule, spark_context
        )

    if mlrun_context:
        # extract ingestion parameters from mlrun context
        if featureset or source is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot specify mlrun_context with feature set or source"
            )
        featureset, source, targets, infer_options = context_to_ingestion_params(
            mlrun_context
        )
        if not source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "data source was not specified"
            )
        mlrun_context.logger.info(f"starting ingestion task to {featureset.uri}")
        return_df = False

    namespace = namespace or get_caller_globals()

    purge_targets = targets or featureset.spec.targets or get_default_targets()
    if overwrite:
        validate_target_list(targets=purge_targets)
        purge_target_names = [
            t if isinstance(t, str) else t.name for t in purge_targets
        ]
        featureset.purge_targets(target_names=purge_target_names, silent=True)
    else:
        for target in purge_targets:
            overwrite_supported_targets = [TargetTypes.parquet, TargetTypes.nosql]
            if target.kind not in overwrite_supported_targets:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Only some targets ({0}) support overwrite=False ingestion".format(
                        ",".join(overwrite_supported_targets)
                    )
                )
            if hasattr(target, "is_single_file") and target.is_single_file():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Overwriting isn't supported in single files. Please use folder path."
                )

    if spark_context and featureset.spec.engine != "spark":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "featureset.spec.engine must be set to 'spark' to ingest with spark"
        )
    if featureset.spec.engine == "spark":
        # use local spark session to ingest
        return _ingest_with_spark(
            spark_context,
            featureset,
            source,
            targets,
            infer_options=infer_options,
            mlrun_context=mlrun_context,
            namespace=namespace,
        )

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    schema_options = InferOptions.get_common_options(
        infer_options, InferOptions.schema()
    )
    if schema_options:
        preview(
            featureset, source, options=schema_options, namespace=namespace,
        )
    infer_stats = InferOptions.get_common_options(
        infer_options, InferOptions.all_stats()
    )
    return_df = return_df or infer_stats != InferOptions.Null
    featureset.save()

    targets = targets or featureset.spec.targets or get_default_targets()
    df = init_featureset_graph(
        source, featureset, namespace, targets=targets, return_df=return_df,
    )
    infer_from_static_df(df, featureset, options=infer_stats)
    _post_ingestion(mlrun_context, featureset, spark_context)

    return df


def preview(
    featureset: FeatureSet,
    source,
    entity_columns=None,
    timestamp_key=None,
    namespace=None,
    options: InferOptions = None,
    verbose=False,
) -> pd.DataFrame:
    """run the ingestion pipeline with local DataFrame/file data and infer features schema and stats

    example::

        quotes_set = FeatureSet("stock-quotes", entities=[Entity("ticker")])
        quotes_set.add_aggregation("asks", "ask", ["sum", "max"], ["1h", "5h"], "10m")
        quotes_set.add_aggregation("bids", "bid", ["min", "max"], ["1h"], "10m")
        df = preview(
            quotes_set,
            quotes_df,
            entity_columns=["ticker"],
            timestamp_key="time",
        )

    :param featureset:     feature set object or uri
    :param source:         source dataframe or csv/parquet file path
    :param entity_columns: list of entity (index) column names
    :param timestamp_key:  timestamp column name
    :param namespace:      namespace or module containing graph classes
    :param options:        schema and stats infer options (:py:class:`~mlrun.feature_store.InferOptions`)
    :param verbose:        verbose log
    """
    options = options if options is not None else InferOptions.default()
    if timestamp_key is not None:
        featureset.spec.timestamp_key = timestamp_key

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

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
            infer_from_static_df(
                source,
                featureset,
                entity_columns,
                InferOptions.get_common_options(options, InferOptions.Entities),
            )
        source = init_featureset_graph(
            source, featureset, namespace, return_df=True, verbose=verbose
        )

    df = infer_from_static_df(source, featureset, entity_columns, options)
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

    example::

        source = HTTPSource()
        func = mlrun.code_to_function("ingest", kind="serving").apply(mount_v3io())
        config = RunConfig(function=func)
        fs.deploy_ingestion_service(my_set, source, run_config=config)

    :param featureset:    feature set object or uri
    :param source:        data source object describing the online or offline source
    :param targets:       list of data target objects
    :param name:          name name for the job/function
    :param run_config:    service runtime configuration (function object/uri, resources, etc..)
    :param verbose:       verbose log
    """
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    run_config = run_config.copy() if run_config else RunConfig()
    source, run_config.parameters = set_task_params(
        featureset, source, targets, run_config.parameters
    )

    name = name or f"{featureset.metadata.name}-ingest"
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

    # todo: add trigger (from source object)

    function.spec.graph = featureset.spec.graph
    function.spec.parameters = run_config.parameters
    function.spec.graph_initializer = (
        "mlrun.feature_store.ingestion.featureset_initializer"
    )
    function.verbose = function.verbose or verbose
    if run_config.local:
        return function.to_mock_server(namespace=get_caller_globals())
    return function.deploy()


def _ingest_with_spark(
    spark=None,
    featureset: Union[FeatureSet, str] = None,
    source: DataSource = None,
    targets: List[DataTargetBase] = None,
    infer_options: InferOptions = InferOptions.default(),
    mlrun_context=None,
    namespace=None,
):
    try:
        if spark is None or spark is True:
            # create spark context
            from pyspark.sql import SparkSession

            if mlrun_context:
                session_name = f"{mlrun_context.name}-{mlrun_context.uid}"
            else:
                session_name = (
                    f"{featureset.metadata.project}-{featureset.metadata.name}"
                )

            spark = SparkSession.builder.appName(session_name).getOrCreate()

        if isinstance(source, pd.DataFrame):
            df = spark.createDataFrame(source)
        else:
            df = source.to_spark_df(spark)
        if featureset.spec.graph and featureset.spec.graph.steps:
            df = run_spark_graph(df, featureset, namespace, spark)
        infer_from_static_df(df, featureset, options=infer_options)

        key_column = featureset.spec.entities[0].name
        timestamp_key = featureset.spec.timestamp_key
        if not targets:
            if not featureset.spec.targets:
                featureset.set_targets()
            targets = featureset.spec.targets
            targets = [get_target_driver(target, featureset) for target in targets]

        for target in targets or []:
            if target.path and urlparse(target.path).scheme == "":
                if mlrun_context:
                    mlrun_context.logger.error(
                        "Paths for spark ingest must contain schema, i.e v3io, s3, az"
                    )
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Paths for spark ingest must contain schema, i.e v3io, s3, az"
                )
            spark_options = target.get_spark_options(key_column, timestamp_key)
            logger.info(
                f"writing to target {target.name}, spark options {spark_options}"
            )
            df.write.mode("overwrite").save(**spark_options)
            target.set_resource(featureset)
            target.update_resource_status("ready")
        _post_ingestion(mlrun_context, featureset, spark)
    finally:
        if spark:
            spark.stop()
    return df


def _post_ingestion(context, featureset, spark=None):
    featureset.save()
    if context:
        context.logger.info("ingestion task completed, targets:")
        context.logger.info(f"{featureset.status.targets.to_dict()}")
        context.log_result("featureset", featureset.uri)


def infer_from_static_df(
    df, featureset, entity_columns=None, options: InferOptions = InferOptions.default()
):
    """infer feature-set schema & stats from static dataframe (without pipeline)"""
    if hasattr(df, "to_dataframe"):
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
        featureset.status.stats = inferer.get_stats(df, options)
    if InferOptions.get_common_options(options, InferOptions.Preview):
        featureset.status.preview = inferer.get_preview(df)
    return df


def set_task_params(
    featureset: FeatureSet,
    source: DataSource = None,
    targets: List[DataTargetBase] = None,
    parameters: dict = None,
    infer_options: InferOptions = InferOptions.Null,
):
    """convert ingestion parameters to dict, return source + params dict"""
    source = source or featureset.spec.source
    parameters = parameters or {}
    parameters["infer_options"] = infer_options
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

    :param uri:  a feature set uri([{project}/{name}[:version])
    :param project:  project name if not specified in uri or not using the current/default
    """
    return get_feature_set_by_uri(uri, project)


def get_feature_vector(uri, project=None):
    """get feature vector object from the db

    :param uri:  a feature vector uri([{project}/{name}[:version])
    :param project:  project name if not specified in uri or not using the current/default
    """
    return get_feature_vector_by_uri(uri, project)


def delete_feature_set(name, project="", tag=None, uid=None, force=False):
    """ Delete a :py:class:`~mlrun.feature_store.FeatureSet` object from the DB.
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
    """ Delete a :py:class:`~mlrun.feature_store.FeatureVector` object from the DB.
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
