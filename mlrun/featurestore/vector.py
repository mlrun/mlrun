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
import os
from tempfile import mktemp

import mlrun
import pandas as pd

from mlrun.model import ModelObj
import v3io.dataplane

from .model import (
    FeatureSetMetadata,
    FeatureVectorSpec,
    FeatureVectorStatus,
)
from .pipeline import init_feature_vector_graph
from ..config import config as mlconf


class FeatureVectorError(Exception):
    """ feature vector error. """

    def __init__(self, *args, **kwargs):
        pass


class FeatureVector(ModelObj):
    """Feature Vector"""

    kind = "FeatureVector"
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self, name=None, features=None, description=None):
        self._spec: FeatureVectorSpec = None
        self._metadata = None
        self._status = None
        self._api_client = None

        self.spec = FeatureVectorSpec(description=description, features=features)
        self.metadata = FeatureSetMetadata(name=name)
        self.status = None

        self._entity_df = None
        self._feature_set_fields = {}
        self.feature_set_objects = {}

    @property
    def spec(self) -> FeatureVectorSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FeatureVectorSpec)

    @property
    def metadata(self) -> FeatureSetMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", FeatureSetMetadata)

    @property
    def status(self) -> FeatureVectorStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureVectorStatus)

    def get_stats_table(self):
        if self.status.stats:
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def parse_features(self):
        processed_features = {}  # dict of name to (featureset, feature object)
        feature_set_objects = {}  # cache of used feature set objects
        feature_set_fields = {}  # list of field (name, alias) per featureset

        def add_feature(name, alias, feature_set_object):
            if alias in processed_features.keys():
                raise ValueError(
                    f"feature name/alias {alias} already specified,"
                    " use another alias (feature-set:name[@alias])"
                )
            feature = feature_set_object[name]
            processed_features[alias or name] = (feature_set_object, feature)
            featureset_name = feature_set_object.metadata.name
            if featureset_name in feature_set_fields.keys():
                value = feature_set_fields[featureset_name]
                value.append((name, alias))
                feature_set_fields[featureset_name] = value
            else:
                feature_set_fields[featureset_name] = [(name, alias)]

        for feature in self._spec.features:
            feature_set, feature_name, alias = _parse_feature_string(feature)
            if feature_set not in feature_set_objects.keys():
                feature_set_objects[feature_set] = mlrun.featurestore.get_feature_set(
                    feature_set
                )
            feature_set_object = feature_set_objects[feature_set]

            feature_fields = feature_set_object.spec.features.keys()
            if feature_name == "*":
                for field in feature_fields:
                    if field != feature_set_object.spec.timestamp_key:
                        if alias:
                            add_feature(field, alias + "_" + field, feature_set_object)
                        else:
                            add_feature(field, field, feature_set_object)
            else:
                if feature_name not in feature_fields:
                    raise ValueError(
                        f"feature {feature} not found in feature set {feature_set}"
                    )
                add_feature(feature_name, alias, feature_set_object)

        for feature_set_name, fields in feature_set_fields.items():
            feature_set = feature_set_objects[feature_set_name]
            for name, alias in fields:
                field_name = alias or name
                if field_name in feature_set.status.stats:
                    self.status.stats[field_name] = feature_set.status.stats[name]
                if field_name in feature_set.spec.features.keys():
                    self.status.features[field_name] = feature_set.spec.features[name]

        return feature_set_objects, feature_set_fields

    def save(self, tag="", versioned=False):
        db = mlrun.get_db_connection()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag
        as_dict = self.to_dict()
        db.store_feature_vector(as_dict, tag=tag, versioned=versioned)


def print_event(event):
    print("EVENT:", str(event.key))
    print(str(event.body))
    return event


class OnlineVectorService:
    def __init__(self, vector):
        self.vector = vector
        self._controller = None

    @property
    def status(self):
        return "ready"

    def start(self):
        self._controller = init_feature_vector_graph(self.vector)

    def get(self, entity_rows: list):
        results = []
        futures = []
        for row in entity_rows:
            futures.append(self._controller.emit(row, return_awaitable_result=True))
        for future in futures:
            result = future.await_result()
            results.append(result.body)

        return results

    def close(self):
        self._controller.terminate()


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
