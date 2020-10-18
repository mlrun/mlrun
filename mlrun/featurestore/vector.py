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

from typing import List

from mlrun.model import ModelObj
from mlrun.run import get_dataitem

from .featureset import FeatureSet
from .model import DataTarget, get_offline_store


class FeatureVectorSpec(ModelObj):
    def __init__(self, client=None, features=None):
        self.features: List[str] = features or []
        self._client = client

        self.entity_source = None
        self._entity_df = None
        self._feature_set_fields = {}
        self._processed_features = {}
        self.feature_set_objects = {}

    def parse_features(self):
        self._processed_features = {}
        self.feature_set_objects = {}
        self._feature_set_fields = {}

        def add_feature(name, alias, feature_set_object):
            if alias in self._processed_features.keys():
                raise ValueError(
                    f"feature name/alias {alias} already specified,"
                    " use another alias (feature-set:name[@alias])"
                )
            feature = feature_set_object[name]
            self._processed_features[alias or name] = (feature_set_object, feature)
            featureset_name = feature_set_object.metadata.name
            if featureset_name in self._feature_set_fields.keys():
                value = self._feature_set_fields[featureset_name]
                value.append((name, alias))
                self._feature_set_fields[featureset_name] = value
            else:
                self._feature_set_fields[featureset_name] = [(name, alias)]

        for feature in self.features:
            feature_set, feature_name, alias = _parse_feature_string(feature)
            if feature_set not in self.feature_set_objects.keys():
                self.feature_set_objects[feature_set] = self._client.get_feature_set(
                    feature_set
                )
            feature_set_object = self.feature_set_objects[feature_set]

            feature_set_fields = feature_set_object.spec.get_features_map().keys()
            if feature_name == "*":
                for field in feature_set_fields:
                    if field != feature_set_object.spec.timestamp_key:
                        if alias:
                            add_feature(field, alias + "_" + field, feature_set_object)
                        else:
                            add_feature(field, field, feature_set_object)
            else:
                if feature_name not in feature_set_fields:
                    raise ValueError(
                        f"feature {feature} not found in feature set {feature_set}"
                    )
                add_feature(feature_name, alias, feature_set_object)

    def load_featureset_dfs(self):
        feature_sets = []
        dfs = []
        for name, columns in self._feature_set_fields.items():
            fs = self.feature_set_objects[name]
            column_names = [name for name, alias in columns]
            if fs.spec.timestamp_key:
                column_names = [fs.spec.timestamp_key] + column_names
            feature_sets.append(fs)
            df = _featureset_to_df(fs, column_names)
            df.rename(
                columns={name: alias for name, alias in columns if alias}, inplace=True
            )
            dfs.append(df)
        return feature_sets, dfs


class OfflineVectorResponse:
    def __init__(self, client, run_url=None, df=None):
        self._client = client
        self._df = df

    @property
    def status(self):
        return "ready"

    def to_dataframe(self):
        return self._df


class OnlineVectorService:
    def __init__(self, client, vector):
        self._client = client
        self._vector = vector

    @property
    def status(self):
        return "ready"

    def get(self, entity_rows):
        return entity_rows


def _parse_feature_string(feature):
    if ":" not in feature:
        raise ValueError(
            f"feature {feature} must be in the form feature-set:name[@alias]"
        )
    splitted = feature.split(":")
    feature_set = splitted[0]
    feature_name = splitted[1]
    splitted = feature_name.split("@")
    if len(splitted) > 1:
        return feature_set, splitted[0], splitted[1]
    return feature_set, feature_name, None


def _featureset_to_df(
    featureset: FeatureSet, columns=None, target_name=None, df_module=None
):
    targets_map = featureset.status.get_targets_map()
    target_name = get_offline_store(targets_map.keys(), target_name)
    target: DataTarget = targets_map[target_name]
    columns = list(featureset.spec.get_entities_map().keys()) + columns
    return get_dataitem(target.path).as_df(columns=columns, df_module=df_module)
