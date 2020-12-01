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
from urllib.parse import urlparse
from storey import Table, V3ioDriver, NoopDriver

from mlrun import get_run_db
from mlrun.datastore import store_manager
from .infer import get_df_stats, get_df_preview
from .pipeline import ingest_from_df

from .vector import (
    OfflineVectorResponse,
    OnlineVectorService,
    FeatureVector,
)
from mlrun.featurestore.mergers.local import LocalFeatureMerger
from .featureset import FeatureSet
from .model import TargetTypes, FeatureClassKind, DataTarget
from ..serving.server import MockContext
from ..utils import get_caller_globals, parse_function_uri


def store_client(
    project=None, secrets=None, context=None, data_prefixes={}, api_address=None
):
    return FeatureStoreClient(project, secrets, context, data_prefixes, api_address)


class FeatureStoreClient:
    def __init__(
        self,
        project=None,
        secrets=None,
        context=None,
        data_prefixes={},
        api_address=None,
    ):
        self.nosql_path_prefix = ""
        self.default_prefixes = data_prefixes or {"parquet": "./store"}
        self.project = project
        self.parameters = {}
        self.default_feature_set = ""
        self.context = context or MockContext()
        setattr(self.context, "get_table", self._get_table)
        setattr(self.context, "client", self)
        try:
            # add v3io:// path prefix support to pandas & dask
            from v3iofs import V3ioFS

            self._v3iofs = V3ioFS()
        except Exception:
            pass

        self._api_address = api_address
        self._db_conn = None
        self._secrets = secrets
        self._data_stores = store_manager.set(secrets)
        self._fs = {}
        self._tabels = {}
        self._default_ingest_targets = [TargetTypes.parquet]

    def _init_featureset_targets(self, featureset, targets):
        if TargetTypes.nosql in targets:
            target_path = self._get_target_path(TargetTypes.nosql, featureset)
            target_path = nosql_path(target_path)
            target = DataTarget("nosql", TargetTypes.nosql, target_path, online=True)
            featureset.status.update_target(target)
            table = Table(target_path, V3ioDriver())
            self._tabels[featureset.uri()] = table

        if TargetTypes.parquet in targets:
            target_path = self._get_target_path(
                TargetTypes.parquet, featureset, ".parquet"
            )
            target = DataTarget("parquet", TargetTypes.parquet, target_path)
            featureset.status.update_target(target)

        if TargetTypes.tsdb in targets:
            target_path = self._get_target_path(
                TargetTypes.tsdb, featureset
            )
            target = DataTarget("tsdb", TargetTypes.tsdb, target_path)
            featureset.status.update_target(target)

    def _get_table(self, name):
        if name in self._tabels:
            return self._tabels[name]

        if name == "":
            table = Table("", NoopDriver())
            self._tabels[name] = table
            return table

        raise ValueError(f"table name={name} not set")

    def _get_db(self):
        if not self._db_conn:
            self._db_conn = get_run_db(self._api_address).connect(self._secrets)
        return self._db_conn

    def get_data_stores(self):
        return self._data_stores

    def _get_db_path(self, kind, name, project=None, version=None):
        project = project or self.project or "default"
        if version:
            name = f"{name}-{version}"
        return f".store/{project}/{kind}/{name}"

    def _get_target_path(self, kind, featureset, suffix=""):
        name = featureset.metadata.name
        version = featureset.metadata.tag
        project = featureset.metadata.project or self.project or "default"
        data_prefix = self.default_prefixes[kind]
        if version:
            name = f"{name}-{version}"
        return f"{data_prefix}/{project}/{kind}/{name}{suffix}"

    def ingest(
        self,
        featureset: Union[FeatureSet, str],
        source,
        targets=None,
        namespace=None,
        return_df=True,
        infer_schema=False,
        with_stats=False,
        with_histogram=False,
        with_preview=False,
    ):
        """Read local DataFrame, file, or URL into the feature store"""
        targets = targets or self._default_ingest_targets
        namespace = namespace or get_caller_globals()
        if isinstance(featureset, str):
            featureset = self.get_feature_set(featureset)

        if isinstance(source, str):
            # if source is a path/url convert to DataFrame
            source = self.get_data_stores().object(url=source).as_df()

        if infer_schema:
            featureset.infer_from_df(source)
        self._init_featureset_targets(featureset, targets)
        return_df = return_df or with_stats or with_preview
        self.save_object(featureset)
        df = ingest_from_df(
            self.context, featureset, source, targets, namespace, return_df=return_df
        ).await_termination()
        if with_stats:
            featureset.status.stats = get_df_stats(df, with_histogram)
        if with_preview:
            featureset.status.preview = get_df_preview(df)
        self.save_object(featureset)
        return df

    def run_ingestion_job(
        self, featureset, source_path, targets=None, parameters=None, function=None
    ):
        """Start MLRun ingestion job to load data into the feature store"""
        pass

    def deploy_ingestion_service(
        self, featureset, source_path, targets=None, parameters=None, function=None
    ):
        """Start real-time Nuclio function which loads data into the feature store"""
        pass

    def get_features_metadata(self, features):
        """return metadata (schema & stats) for requested features"""
        pass

    def get_offline_features(
        self,
        features,
        entity_rows=None,
        entity_timestamp_column=None,
        watch=True,
        store_target=None,
    ):

        merger = LocalFeatureMerger()
        vector = FeatureVector(features=features)
        vector.parse_features(self)
        featuresets, feature_dfs = vector.load_featureset_dfs()
        merger.merge(entity_rows, entity_timestamp_column, featuresets, feature_dfs)
        return OfflineVectorResponse(self, merger)

    def get_online_feature_service(self, features):
        vector = FeatureVector(features=features)
        vector.parse_features(self)
        for featureset in vector.feature_set_objects.values():
            target_path = featureset.status.targets["nosql"].path
            table = Table(target_path, V3ioDriver())
            self._tabels[featureset.uri()] = table

        service = OnlineVectorService(self, vector)
        service.init()
        return service

    def get_feature_set(self, uri, use_cache=False):
        if not uri and self.default_feature_set:
            uri = self.default_feature_set
        if not uri:
            raise ValueError("name or client.default_feature_set must be set")

        if use_cache and uri in self._fs:
            return self._fs[uri]
        project, name, tag, uid = parse_function_uri(uri)
        project = project or self.project or ""
        obj = self._get_db().get_feature_set(name, project, tag, uid)
        fs = FeatureSet.from_dict(obj)
        self._fs[uri] = fs
        return fs

    def list_feature_sets(
        self,
        name: str = None,
        project: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
    ):
        """list feature sets with optional filter"""
        project = project or self.project
        resp = self._get_db().list_feature_sets(
            project, name, tag, state, labels=labels
        )
        print(resp.dict())
        if resp:
            return [FeatureSet.from_dict(obj) for obj in resp.dict()["feature_sets"]]

    def get_feature_vector(self, name, project=None):
        raise NotImplementedError("api not yet not supported")

    def save_object(self, obj, versioned=False):
        """save feature set/vector or other definitions into the DB"""
        db = self._get_db()
        if obj.kind == FeatureClassKind.FeatureSet:
            obj.metadata.project = obj.metadata.project or self.project
            obj_dict = obj.to_dict()
            # obj_dict['metadata']['labels'] = obj_dict['metadata'].get('labels', {})  # bypass DB bug
            db.store_feature_set(obj_dict, tag=obj.metadata.tag, versioned=versioned)
        elif obj.kind == FeatureClassKind.FeatureVector:
            # TODO: write to mlrun db
            target = self._get_db_path(
                obj.kind, obj.metadata.name, obj.metadata.project
            )
            self._data_stores.object(url=target + ".yaml").put(obj.to_yaml())
        else:
            raise NotImplementedError(f"object kind not supported ({obj.kind})")


def nosql_path(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme.lower()
    if scheme != "v3io":
        raise ValueError("url must start with v3io://[host]/{container}/{path}")

    endpoint = parsed_url.hostname
    if parsed_url.port:
        endpoint += ":{}".format(parsed_url.port)
    # todo: use endpoint
    return parsed_url.path.strip("/") + "/"
