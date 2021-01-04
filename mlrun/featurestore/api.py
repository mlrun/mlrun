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
import mlrun
from .common import get_feature_vector_by_uri, get_feature_set_by_uri
from .infer import (
    InferOptions,
    infer_from_source,
)
from .model.base import DataTargetSpec
from .retrieval import LocalFeatureMerger, init_feature_vector_graph
from .ingestion import init_featureset_graph
from .model import FeatureVector, FeatureSet, OnlineVectorService
from .targets import get_default_targets
from ..utils import get_caller_globals

_v3iofs = None


try:
    # add v3io:// path prefix support to pandas & dask, todo: move to datastores
    from v3iofs import V3ioFS

    _v3iofs = V3ioFS()
except Exception:
    pass


def _features_to_vector(features, name):
    if isinstance(features, str):
        vector = get_feature_vector_by_uri(features)
    elif isinstance(features, list):
        vector = FeatureVector(features=features)
    elif isinstance(features, FeatureVector):
        vector = features
    else:
        raise ValueError("illegal features value/type")

    if name:
        vector.metadata.name = name
    return vector


def get_offline_features(
    features: Union[str, List[str], FeatureVector],
    entity_rows=None,
    entity_timestamp_column: str = None,
    name: str = None,
    watch: bool = True,
    store_target: DataTargetSpec = None,
):
    """retrieve offline feature vector

    example:
        features = [
            "stock-quotes#bid",
            "stock-quotes#asks_sum_5h",
            "stock-quotes#ask as mycol",
            "stocks#*",
        ]

        resp = fs.get_offline_features(
            features, entity_rows=trades, entity_timestamp_column="time"
        )
        print(resp.vector.to_yaml())
        print(resp.to_dataframe())
        resp.to_parquet("./xx.parquet")

    :param features:     - list of features or feature vector uri or FeatureVector object
    :param entity_rows:  - dataframe with entity rows to join with
    :param name:         - name to use for the generated feature vector
    :param watch:        - indicate we want to wait for the result
    :param store_target: - where to write the results to
    :param entity_timestamp_column: - timestamp column name in the entity rows dataframe
    """
    vector = _features_to_vector(features, name)
    entity_timestamp_column = entity_timestamp_column or vector.spec.timestamp_field
    merger = LocalFeatureMerger(vector)
    return merger.start(entity_rows, entity_timestamp_column, store_target)


def get_online_feature_service(
    features: Union[str, List[str], FeatureVector], name: str = None, function=None
):
    """initialize and return the feature vector online client

    :param features:     - list of features or feature vector uri or FeatureVector object
    :param name:         - name to use for the generated feature vector
    :param function:     - optional, mlrun FunctionReference object, serverless function template
    """
    vector = _features_to_vector(features, name)
    controller = init_feature_vector_graph(vector)
    service = OnlineVectorService(vector, controller)
    return service


def ingest(
    featureset: Union[FeatureSet, str],
    source,
    targets=None,
    namespace=None,
    return_df=True,
    infer_options: InferOptions = InferOptions.Null,
):
    """Read local DataFrame, file, or URL into the feature store"""
    namespace = namespace or get_caller_globals()
    if isinstance(featureset, str):
        featureset = get_feature_set_by_uri(featureset)

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    if infer_options & InferOptions.Schema:
        infer_metadata(
            featureset,
            source,
            options=infer_options & InferOptions.Schema,
            namespace=namespace,
        )
    infer_stats = infer_options & InferOptions.AllStats
    return_df = return_df or infer_stats != InferOptions.Null
    featureset.save()

    targets = targets or featureset.spec.targets or get_default_targets()
    controller = init_featureset_graph(
        source, featureset, namespace, targets=targets, return_df=return_df
    )
    df = controller.await_termination()
    infer_from_source(df, featureset, options=infer_stats)
    featureset.save()
    return df


def infer_metadata(
    featureset,
    source,
    entity_columns=None,
    timestamp_key=None,
    namespace=None,
    options: InferOptions = None,
):
    """Infer features schema and stats from a local DataFrame"""
    options = options if options is not None else InferOptions.default()
    if timestamp_key is not None:
        featureset.spec.timestamp_key = timestamp_key

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    namespace = namespace or get_caller_globals()
    if featureset.spec.require_processing():
        # find/update entities schema
        infer_from_source(
            source, featureset, entity_columns, options & InferOptions.Schema
        )
        controller = init_featureset_graph(
            source, featureset, namespace, return_df=True
        )
        source = controller.await_termination()

    infer_from_source(source, featureset, entity_columns, options)
    return source


def run_ingestion_task(
    featureset, source, targets=None, parameters=None, function=None
):
    """Start MLRun ingestion job or nuclio function to load data into the feature store"""
    pass
