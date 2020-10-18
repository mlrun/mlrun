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

import yaml
from mlrun.datastore import store_manager

from .vector import FeatureVectorSpec, OfflineVectorResponse, OnlineVectorService
from .mergers import LocalFeatureMerger
from .featureset import FeatureSet
from .model import DataTarget, TargetTypes, FeatureClassKind
from .ingest import write_to_target_store


def store_client(data_prefix="", project=None, secrets=None):
    return FeatureStoreClient(data_prefix, project, secrets)


class FeatureStoreClient:
    def __init__(self, data_prefix="", project=None, secrets=None):
        self._api = None
        self.data_prefix = data_prefix or "./store"
        self.nosql_path_prefix = ""
        self._data_stores = store_manager.set(secrets)
        self._fs = {}
        self._default_ingest_targets = [TargetTypes.parquet]
        self.project = project
        self.parameters = {}

    def get_data_stores(self):
        return self._data_stores

    def _get_target_path(self, kind, name, project=None, version=None):
        project = project or self.project or "default"
        data_prefix = (
            self.nosql_path_prefix if kind == TargetTypes.nosql else self.data_prefix
        )
        if version:
            name = f"{name}-{version}"
        return f"{data_prefix}/{project}/{kind}/{name}"

    def ingest(self, featureset: FeatureSet, source, targets=None):
        """Read local DataFrame, file, or URL into the feature store"""
        targets = targets or self._default_ingest_targets
        if not targets:
            raise ValueError("ingestion target(s) were not specified")
        for target in targets:
            target_path = self._get_target_path(
                target, featureset.metadata.name, featureset.metadata.project
            )
            target_path = write_to_target_store(
                self, target, source, target_path, featureset
            )
            target = DataTarget(target, target_path)
            featureset.status.update_target(target)
        self.save_object(featureset)

    def run_ingestion_job(
        self, featureset, source_path, targets=None, parameters=None, function=None
    ):
        """Start MLRun ingestion job to load data into the feature store"""
        pass

    def deploy_ingestion_service(
        self, featureset, source_path, argets=None, parameters=None, function=None
    ):
        """Start real-time Nuclio function which loads data into the feature store"""
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
        vector = FeatureVectorSpec(self, features)
        vector.parse_features()
        featuresets, feature_dfs = vector.load_featureset_dfs()
        df = merger.merge(
            entity_rows, entity_timestamp_column, featuresets, feature_dfs
        )
        return OfflineVectorResponse(self, df=df)

    def get_online_feature_service(self, features, store_kind):
        vector = FeatureVectorSpec(self, features)
        return OnlineVectorService(self, vector)

    def get_feature_set(self, name, project=None):
        target = self._get_target_path(FeatureClassKind.FeatureSet, name, project)
        body = self._data_stores.object(url=target + ".yaml").get()
        obj = yaml.load(body, Loader=yaml.FullLoader)
        return FeatureSet.from_dict(obj)

    def save_object(self, obj):
        """save featureset or other definitions into the DB"""
        if obj.kind != FeatureClassKind.FeatureSet:
            raise NotImplementedError('only support FeatureSet for now')
        target = self._get_target_path(
            FeatureClassKind.FeatureSet,
            obj.metadata.name,
            obj.metadata.project,
        )
        self._data_stores.object(url=target + ".yaml").put(obj.to_yaml())
