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
from .infer import (
    get_df_stats,
    get_df_preview,
    infer_schema_from_df,
    InferOptions,
    infer_from_source,
)
from .retrieval import LocalFeatureMerger, init_feature_vector_graph
from .ingestion import init_featureset_graph
from .model import FeatureVector, FeatureSet, OnlineVectorService
from ..config import config
from ..utils import parse_function_uri, get_caller_globals

_v3iofs = None


try:
    # add v3io:// path prefix support to pandas & dask, todo: move to datastores
    from v3iofs import V3ioFS

    _v3iofs = V3ioFS()
except Exception:
    pass


def get_feature_set(uri):
    """get feature set from db"""

    db = mlrun.get_run_db()
    project, name, tag, uid = parse_function_uri(uri, config.default_project)
    obj = db.get_feature_set(name, project, tag, uid)
    return FeatureSet.from_dict(obj)


def get_feature_vector(uri):
    """get feature vector from db"""

    db = mlrun.get_run_db()
    project, name, tag, uid = parse_function_uri(uri, config.default_project)
    obj = db.get_feature_vector(name, project, tag, uid)
    return FeatureVector.from_dict(obj)


def list_feature_sets(
    name: str = None,
    project: str = None,
    tag: str = None,
    state: str = None,
    labels: List[str] = None,
):
    """list feature sets with optional filter"""

    project = project or config.default_project
    db = mlrun.get_run_db()
    resp = db.list_feature_sets(project, name, tag, state, labels=labels)
    if resp:
        return [FeatureSet.from_dict(obj) for obj in resp]


def list_feature_vectors(
    name: str = None,
    project: str = None,
    tag: str = None,
    state: str = None,
    labels: List[str] = None,
):
    """list feature vectors with optional filter"""

    project = project or config.default_project
    db = mlrun.get_run_db()
    resp = db.list_feature_vectors(project, name, tag, state, labels=labels)
    if resp:
        return [FeatureVector.from_dict(obj) for obj in resp]


def _features_to_vector(features, name):
    if isinstance(features, str):
        vector = get_feature_vector(features)
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
    features,
    entity_rows=None,
    entity_timestamp_column=None,
    name=None,
    watch=True,
    store_target=None,
):
    vector = _features_to_vector(features, name)
    entity_timestamp_column = entity_timestamp_column or vector.spec.timestamp_field
    merger = LocalFeatureMerger(vector)
    return merger.start(entity_rows, entity_timestamp_column, store_target)


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
        featureset = get_feature_set(featureset)

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

    controller = init_featureset_graph(
        source, featureset, namespace, with_targets=True, return_df=return_df
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
    label_column=None,
    namespace=None,
    options: InferOptions = InferOptions.Null,
):
    """Infer features schema and stats from a local DataFrame"""
    options = options if options is not None else InferOptions.default()
    if timestamp_key is not None:
        featureset.spec.timestamp_key = timestamp_key

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = mlrun.store_manager.object(url=source).as_df()

    namespace = namespace or get_caller_globals()
    df = None
    if featureset.spec.require_processing():
        # find/update entities schema
        infer_from_source(
            source, featureset, entity_columns, options & InferOptions.Schema
        )
        controller = init_featureset_graph(
            source, featureset, namespace, with_targets=False, return_df=True
        )
        df = controller.await_termination()

    infer_from_source(source, featureset, entity_columns, options)
    if label_column:
        featureset.spec.label_column = label_column
    return df


def get_online_feature_service(features, name=None, function=None):
    vector = _features_to_vector(features, name)
    controller = init_feature_vector_graph(vector)
    service = OnlineVectorService(vector, controller)
    return service


def run_ingestion_job(
    featureset, source_path, targets=None, parameters=None, function=None
):
    """Start MLRun ingestion job to load data into the feature store"""
    pass


def deploy_ingestion_service(
    featureset, source_path, targets=None, parameters=None, function=None
):
    """Start real-time Nuclio function which loads data into the feature store"""
    pass


def get_features_metadata(features):
    """return metadata (schema & stats) for requested features"""
    pass
