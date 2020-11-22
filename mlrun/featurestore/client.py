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
import yaml

from mlrun import get_run_db
from mlrun.datastore import store_manager
from .infer import get_df_stats, get_df_preview
from .pipeline import create_ingestion_pipeline

from .vector import (
    OfflineVectorResponse,
    OnlineVectorService,
    FeatureVector,
)
from mlrun.featurestore.mergers.local import LocalFeatureMerger
from .featureset import FeatureSet
from .model import TargetTypes, FeatureClassKind
from ..utils import get_caller_globals, parse_function_uri


def store_client(data_prefix="", project=None, secrets=None, api_address=None):
    return FeatureStoreClient(data_prefix, project, secrets, api_address)


class FeatureStoreClient:
    def __init__(self, data_prefix="", project=None, secrets=None, api_address=None):
        self.data_prefix = data_prefix or "./store"
        self.nosql_path_prefix = ""
        self.project = project
        self.parameters = {}
        self.default_feature_set = ''
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
        self._default_ingest_targets = [TargetTypes.parquet]

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
        return f"{self.data_prefix}/{project}/{kind}/{name}"

    def _get_target_path(self, kind, featureset, suffix=""):
        name = featureset.metadata.name
        version = featureset.metadata.tag
        project = featureset.metadata.project or self.project or "default"
        if kind == TargetTypes.nosql:
            data_prefix = nosql_path(self.nosql_path_prefix or self.data_prefix)
        else:
            data_prefix = self.data_prefix
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
        entity_list = list(featureset.spec.entities.keys())
        if not entity_list:
            raise ValueError("Entity columns are not defined for this feature set")

        if isinstance(source, str):
            # if source is a path/url convert to DataFrame
            source = self.get_data_stores().object(url=source).as_df()

        if infer_schema:
            featureset.infer_from_df(source)
        df = create_ingestion_pipeline(self, featureset, source, targets, namespace).await_termination()
        if with_stats:
            featureset.status.stats = get_df_stats(df, with_histogram)
        if with_preview:
            featureset.status.preview = get_df_preview(df)
        self.save_object(featureset)
        if return_df:
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
        vector = FeatureVector(self, features=features)
        vector.parse_features()
        featuresets, feature_dfs = vector.load_featureset_dfs()
        df = merger.merge(
            entity_rows, entity_timestamp_column, featuresets, feature_dfs
        )
        return OfflineVectorResponse(self, merger)

    def get_online_feature_service(self, features):
        vector = FeatureVector(self, features=features)
        vector.parse_features()
        service = OnlineVectorService(self, vector)
        service.init()
        return service

    def get_feature_set(self, uri, use_cache=False):
        if not uri and self.default_feature_set:
            uri = self.default_feature_set
        if not uri:
            raise ValueError('name or client.default_feature_set must be set')

        if use_cache and uri in self._fs:
            return self._fs[uri]
        project, name, tag, uid = parse_function_uri(uri)
        project = project or self.project
        obj = self._get_db().get_feature_set(name, project, tag, uid)
        fs = FeatureSet.from_dict(obj.dict())
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
        resp = self._get_db().list_feature_sets(project, name, tag, state, labels=labels)
        print(resp.dict())
        if resp:
            return [FeatureSet.from_dict(obj) for obj in resp.dict()['feature_sets']]

    def get_feature_vector(self, name, project=None):
        raise NotImplementedError("api not yet not supported")

    def save_object(self, obj, versioned=False):
        """save feature set/vector or other definitions into the DB"""
        db = self._get_db()
        if obj.kind == FeatureClassKind.FeatureSet:
            obj.metadata.project = obj.metadata.project or self.project
            obj_dict = obj.to_dict()
            obj_dict['metadata']['labels'] = obj_dict['metadata'].get('labels', {})  # bypass DB bug
            db.store_feature_set(obj.metadata.name, obj_dict, obj.metadata.project,
                                 tag=obj.metadata.tag, versioned=versioned)
            #db.create_feature_set(objdeict, obj.metadata.project, versioned=False)
        elif obj.kind == FeatureClassKind.FeatureVector:
            # TODO: write to mlrun db
            target = self._get_db_path(obj.kind, obj.metadata.name, obj.metadata.project)
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
    return parsed_url.path.strip("/")
