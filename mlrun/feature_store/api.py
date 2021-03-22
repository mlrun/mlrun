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
from typing import List, Union

import pandas as pd

import mlrun

from ..data_types import InferOptions, get_infer_interface
from ..datastore.targets import get_default_targets, get_target_driver
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
)
from .retrieval import LocalFeatureMerger, init_feature_vector_graph, run_merge_job

_v3iofs = None
spark_transform_handler = "transform"


def _features_to_vector(features):
    if isinstance(features, str):
        vector = get_feature_vector_by_uri(features)
    elif isinstance(features, FeatureVector):
        vector = features
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"illegal features value/type ({type(features)})"
        )
    return vector


def get_offline_features(
    feature_vector: Union[str, FeatureVector],
    entity_rows=None,
    entity_timestamp_column: str = None,
    batch: bool = False,
    target: DataTargetBase = None,
    run_config: RunConfig = None,
    drop_columns: List[str] = None,
) -> OfflineVectorResponse:
    """retrieve offline feature vector results

    specify list of features or feature vector object/uri and retrieve the desired features,
    their metadata and statistics. results can be returned as a dataframe or written to a target

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
    :param batch:          run as a remote (cluster) batch job
    :param target:         where to write the results to
    :param drop_columns:   list of columns to drop from the final result
    :param entity_timestamp_column: timestamp column name in the entity rows dataframe
    :param run_config:     function and/or run configuration
    """
    feature_vector = _features_to_vector(feature_vector)

    entity_timestamp_column = (
        entity_timestamp_column or feature_vector.spec.timestamp_field
    )
    if batch:
        return run_merge_job(
            feature_vector,
            target,
            entity_rows,
            timestamp_column=entity_timestamp_column,
            run_config=run_config,
            drop_columns=drop_columns,
        )

    merger = LocalFeatureMerger(feature_vector)
    return merger.start(
        entity_rows, entity_timestamp_column, target=target, drop_columns=drop_columns,
    )


def get_online_feature_service(
    feature_vector: Union[str, FeatureVector], run_config: RunConfig = None,
) -> OnlineVectorService:
    """initialize and return online feature vector service api

    example::

        svc = get_online_feature_service(vector_uri)
        resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
        print(resp)
        resp = svc.get([{"ticker": "AAPL"}], as_list=True)
        print(resp)

    :param feature_vector:  feature vector uri or FeatureVector object
    :param function:     optional, mlrun FunctionReference object, serverless function template
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
    transformer=None,  # temporary, will be merged into the graph
) -> pd.DataFrame:
    """Read local DataFrame, file, or URL into the feature store

    example::

        stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
        stocks = pd.read_csv("stocks.csv")
        df = ingest(stocks_set, stocks, infer_options=fs.InferOptions.default())

        # for running as remote job
        config = RunConfig(image='mlrun/mlrun').apply(mount_v3io())
        df = ingest(stocks_set, stocks, run_config=config)

    :param featureset:    feature set object or uri
    :param source:        source dataframe or file path
    :param targets:       optional list of data target objects
    :param namespace:     namespace or module containing graph classes
    :param return_df:     indicate if to return a dataframe with the graph results
    :param infer_options: schema and stats infer options
    :param run_config:    function and/or run configuration for remote jobs
    :param mlrun_context: mlrun context (when running as a job), for internal use !
    :param spark_context: local spark session or True to create one
    :param transformer:   custom transformation function
    """
    if not mlrun_context and (not featureset or source is None):
        raise mlrun.errors.MLRunInvalidArgumentError(
            "feature set and source must be specified"
        )
    if featureset and isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    if run_config:
        # remote job execution
        run_config = run_config.copy() if run_config else RunConfig()
        source, run_config.parameters = set_task_params(
            featureset, source, targets, run_config.parameters, infer_options
        )
        name = f"{featureset.metadata.name}_ingest"
        return run_ingestion_job(name, featureset, run_config, source.schedule)

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

    if spark_context:
        # use local spark session to ingest
        return _ingest_with_spark(
            spark_context,
            featureset,
            source,
            targets,
            infer_options=infer_options,
            transformer=transformer,
            mlrun_context=mlrun_context,
        )

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    schema_options = InferOptions.get_common_options(
        infer_options, InferOptions.schema()
    )
    if schema_options:
        infer_metadata(
            featureset, source, options=schema_options, namespace=namespace,
        )
    infer_stats = InferOptions.get_common_options(
        infer_options, InferOptions.all_stats()
    )
    return_df = return_df or infer_stats != InferOptions.Null
    featureset.save()

    targets = targets or featureset.spec.targets or get_default_targets()
    graph = init_featureset_graph(
        source, featureset, namespace, targets=targets, return_df=return_df
    )
    df = graph.wait_for_completion()
    infer_from_static_df(df, featureset, options=infer_stats)
    _post_ingestion(mlrun_context, featureset, spark_context)
    return df


def infer(
    featureset: FeatureSet,
    source,
    entity_columns=None,
    timestamp_key=None,
    namespace=None,
    options: InferOptions = None,
) -> pd.DataFrame:
    """Infer features schema and stats from a local DataFrame

    example::

        quotes_set = FeatureSet("stock-quotes", entities=[Entity("ticker")])
        quotes_set.add_aggregation("asks", "ask", ["sum", "max"], ["1h", "5h"], "10m")
        quotes_set.add_aggregation("bids", "bid", ["min", "max"], ["1h"], "10m")
        df = infer_metadata(
            quotes_set,
            quotes_df,
            entity_columns=["ticker"],
            timestamp_key="time",
            options=fs.InferOptions.default(),
        )

    :param featureset:     feature set object or uri
    :param source:         source dataframe or file path
    :param entity_columns: list of entity (index) column names
    :param timestamp_key:  timestamp column name
    :param namespace:      namespace or module containing graph classes
    :param options:        schema and stats infer options
    """
    options = options if options is not None else InferOptions.default()
    if timestamp_key is not None:
        featureset.spec.timestamp_key = timestamp_key

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    namespace = namespace or get_caller_globals()
    if featureset.spec.require_processing():
        # find/update entities schema
        if len(featureset.spec.entities) == 0:
            infer_from_static_df(
                source,
                featureset,
                entity_columns,
                InferOptions.get_common_options(options, InferOptions.Entities),
            )
        graph = init_featureset_graph(source, featureset, namespace, return_df=True)
        source = graph.wait_for_completion()

    df = infer_from_static_df(source, featureset, entity_columns, options)
    return df


# keep for backwards compatibility
infer_metadata = infer


def _run_ingestion_job(
    featureset: Union[FeatureSet, str],
    source: DataSource = None,
    targets: List[DataTargetBase] = None,
    name: str = None,
    infer_options: InferOptions = InferOptions.default(),
    run_config: RunConfig = None,
):
    """Start batch ingestion task using remote MLRun job or spark function

    Deploy and run batch job implementing feature ingestion pipeline
    sources will deploy mlrun python or spark jobs (use the `engine` attribute to select spark),
    for scheduled jobs set the schedule attribute in the offline source.

    example::

        source = CSVSource("mycsv", path="measurements.csv")
        targets = [CSVTarget("mycsv", path="./mycsv.csv")]
        run_ingestion_job(measurements, source, targets, name="tst_ingest")

    :param featureset:    feature set object or uri
    :param source:        data source object describing the online or offline source
    :param targets:       list of data target objects
    :param name:          name name for the job/function
    :param infer_options: schema and stats infer options
    :param run_config:    (function object/uri, resources, etc..)
    """
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
    infer_options: InferOptions = InferOptions.Null,
    run_config: RunConfig = None,
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
    :param infer_options: schema and stats infer options
    :param run_config:    service runtime configuration (function object/uri, resources, etc..)
    """
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    run_config = run_config.copy() if run_config else RunConfig()
    source, run_config.parameters = set_task_params(
        featureset, source, targets, run_config.parameters, infer_options
    )

    name = name or f"{featureset.metadata.name}_ingest"
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
    function.verbose = True
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
    transformer=None,
    namespace=None,
):
    """Start ingestion task using Spark

    example::

        # custom transformation function
        def transform(spark, context, df):
            df.filter("age > 40")
            return df

        spark = SparkSession.builder.appName("Spark function").getOrCreate()
        featureset = fs.FeatureSet("iris", entities=[fs.Entity("length")])
        source = CSVSource('mydata', 'v3io:///projects/iris/mycsv.csv')
        targets = [CSVTarget('out', 'v3io:///projects/iris/dataout')]

        df = spark_ingestion(spark, featureset, source, targets,
                             fs.InferOptions.all(), transformer=transform)

    :param spark:         spark session
    :param featureset:    feature set object or uri
    :param source:        data source object describing the online or offline source
    :param targets:       list of data target objects
    :param infer_options: schema and stats infer options
    :param mlrun_context: mlrun context (when running as a job)
    :param transformer:   custom transformation function
    :param namespace:      namespace or module containing graph classes
    """
    if spark is None or spark is True:
        # create spark context
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName(
            f"{mlrun_context.name}-{mlrun_context.uid}"
        ).getOrCreate()

    transformer = transformer or namespace.get(spark_transform_handler, None)
    df = source.to_spark_df(spark)
    infer_from_static_df(df, featureset, options=infer_options)

    if transformer:
        df = transformer(spark, mlrun_context, df)

    key_column = featureset.spec.entities[0].name
    timestamp_key = featureset.spec.timestamp_key
    if not targets:
        if not featureset.spec.targets:
            featureset.set_targets()
        targets = featureset.spec.targets
        targets = [get_target_driver(target, featureset) for target in targets]

    for target in targets or []:
        spark_options = target.get_spark_options(key_column, timestamp_key)
        logger.info(f"writing to target {target.name}, spark options {spark_options}")
        df.write.mode("overwrite").save(**spark_options)
        target.set_resource(featureset)
        target.update_resource_status("ready", is_dir=True)

    _post_ingestion(mlrun_context, featureset, spark)
    return df


def _post_ingestion(context, featureset, spark=None):
    featureset.save()
    if context:
        context.logger.info("ingestion task completed, targets:")
        context.logger.info(f"{featureset.status.targets.to_dict()}")
        context.log_result("featureset", featureset.uri)
        if spark:
            spark.stop()


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


def delete_feature_set(name, project=""):
    """ Delete a :py:class:`~mlrun.feature_store.FeatureSet` object from the DB.
    :param name: Name of the object to delete
    :param project: Name of the object's project
    """
    db = mlrun.get_run_db()
    return db.delete_feature_set(name=name, project=project)


def delete_feature_vector(name, project=""):
    """ Delete a :py:class:`~mlrun.feature_store.FeatureVector` object from the DB.
    :param name: Name of the object to delete
    :param project: Name of the object's project
    """
    db = mlrun.get_run_db()
    return db.delete_feature_vector(name=name, project=project)
