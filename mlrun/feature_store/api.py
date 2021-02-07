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

from typing import List, Union, Dict
import mlrun
import pandas as pd
from .common import get_feature_vector_by_uri, get_feature_set_by_uri
from .model.base import DataTargetBase, DataSource
from .retrieval import LocalFeatureMerger, init_feature_vector_graph
from .ingestion import init_featureset_graph, deploy_ingestion_function
from .model import FeatureVector, FeatureSet, OnlineVectorService, OfflineVectorResponse
from .targets import get_default_targets
from ..runtimes.function_reference import FunctionReference
from ..utils import get_caller_globals
from ..features import infer_schema_from_df, InferOptions, get_df_stats, get_df_preview

_v3iofs = None


try:
    # add v3io:// path prefix support to pandas & dask, todo: move to datastores
    from v3iofs import V3ioFS

    _v3iofs = V3ioFS()
except Exception:
    pass


def _features_to_vector(features):
    if isinstance(features, str):
        vector = get_feature_vector_by_uri(features)
    elif isinstance(features, list):
        vector = FeatureVector(features=features)
    elif isinstance(features, FeatureVector):
        vector = features
    else:
        raise mlrun.errors.MLRunInvalidArgumentError("illegal features value/type")
    return vector


def get_offline_features(
    features: Union[str, List[str], FeatureVector],
    entity_rows=None,
    entity_timestamp_column: str = None,
    watch: bool = True,
    store_target: DataTargetBase = None,
    engine: str = None,
) -> OfflineVectorResponse:
    """retrieve offline feature vector results

    example::

        features = [
            "stock-quotes.bid",
            "stock-quotes.asks_sum_5h",
            "stock-quotes.ask as mycol",
            "stocks.*",
        ]

        resp = get_offline_features(
            features, entity_rows=trades, entity_timestamp_column="time"
        )
        print(resp.to_dataframe())
        print(resp.vector.get_stats_table())
        resp.to_parquet("./out.parquet")

    :param features:     list of features or feature vector uri or FeatureVector object
    :param entity_rows:  dataframe with entity rows to join with
    :param watch:        indicate we want to wait for the result
    :param store_target: where to write the results to
    :param engine:       join/merge engine (local, job, spark)
    :param entity_timestamp_column: timestamp column name in the entity rows dataframe
    """
    vector = _features_to_vector(features)
    entity_timestamp_column = entity_timestamp_column or vector.spec.timestamp_field
    # todo: support different merger engines (job, spark)
    merger = LocalFeatureMerger(vector)
    return merger.start(entity_rows, entity_timestamp_column, store_target)


def get_online_feature_service(
    features: Union[str, List[str], FeatureVector], function=None
) -> OnlineVectorService:
    """initialize and return online feature vector service api

    example::

        svc = get_online_feature_service(vector_uri)
        resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
        print(resp)
        resp = svc.get([{"ticker": "AAPL"}])
        print(resp)

    :param features:     list of features or feature vector uri or FeatureVector object
    :param function:     optional, mlrun FunctionReference object, serverless function template
    """
    vector = _features_to_vector(features)
    graph = init_feature_vector_graph(vector)
    service = OnlineVectorService(vector, graph)

    # todo: support remote service (using remote nuclio/mlrun function)
    return service


def ingest(
    featureset: Union[FeatureSet, str],
    source,
    targets: List[DataTargetBase] = None,
    namespace=None,
    return_df: bool = True,
    infer_options: InferOptions = InferOptions.Null,
) -> pd.DataFrame:
    """Read local DataFrame, file, or URL into the feature store

    example::

        stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
        stocks = pd.read_csv("stocks.csv")
        df = ingest(stocks_set, stocks, infer_options=fs.InferOptions.default())

    :param featureset:    feature set object or uri
    :param source:        source dataframe or file path
    :param targets:       optional list of data target objects
    :param namespace:     namespace or module containing graph classes
    :param return_df:     indicate if to return a dataframe with the graph results
    :param infer_options: schema and stats infer options
    """
    namespace = namespace or get_caller_globals()
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    if infer_options & InferOptions.schema():
        infer_metadata(
            featureset,
            source,
            options=InferOptions.get_common_options(
                infer_options, InferOptions.schema()
            ),
            namespace=namespace,
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
    _infer_from_df(df, featureset, options=infer_stats)
    featureset.save()
    return df


def infer_metadata(
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
        _infer_from_df(
            source,
            featureset,
            entity_columns,
            InferOptions.get_common_options(options, InferOptions.schema()),
        )
        graph = init_featureset_graph(source, featureset, namespace, return_df=True)
        source = graph.wait_for_completion()

    _infer_from_df(source, featureset, entity_columns, options)
    return source


def run_ingestion_task(
    featureset: Union[FeatureSet, str],
    source: DataSource = None,
    targets: List[DataTargetBase] = None,
    name: str = None,
    infer_options: InferOptions = InferOptions.Null,
    parameters: Dict[str, Union[str, list, dict]] = None,
    function: FunctionReference = None,
    local=False,
    watch=True,
):
    """Start ingestion task using MLRun job or nuclio function

    example::

        source = CSVSource("mycsv", path="measurements.csv")
        targets = [CSVTarget("mycsv", path="./mycsv.csv")]
        run_ingestion_task(measurements, source, targets, name="tst_ingest")

    :param featureset:    feature set object or uri
    :param source:        data source object describing the online or offline source
    :param targets:       list of data target objects
    :param name:          name name for the job/function
    :param infer_options: schema and stats infer options
    :param parameters:    extra parameter dictionary which is passed to the graph context
    :param function:      reference to custom ingestion function (points to function/code/..)
    :param local:         run local emulation using mock_server() or run_local()
    :param watch:         wait for job completion, set to False if you dont want to wait
    """
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    source = source or featureset.spec.source
    parameters = parameters or {}
    parameters["infer_options"] = infer_options
    parameters["featureset"] = featureset.uri
    if not source.online:
        parameters["source"] = source.to_dict()
    if targets:
        parameters["targets"] = [target.to_dict() for target in targets]
    elif not featureset.spec.targets:
        featureset.set_targets()
    featureset.save()

    deploy_ingestion_function(
        name,
        source,
        featureset,
        parameters,
        function_ref=function,
        local=local,
        watch=watch,
    )
    return


def _infer_from_df(
    df, featureset, entity_columns=None, options: InferOptions = InferOptions.Null
):
    if InferOptions.get_common_options(options, InferOptions.schema()):
        featureset.spec.timestamp_key = infer_schema_from_df(
            df,
            featureset.spec.features,
            featureset.spec.entities,
            featureset.spec.timestamp_key,
            entity_columns,
            options=options,
        )
    if InferOptions.get_common_options(options, InferOptions.Stats):
        featureset.status.stats = get_df_stats(df, options)
    if InferOptions.get_common_options(options, InferOptions.Preview):
        featureset.status.preview = get_df_preview(df)
