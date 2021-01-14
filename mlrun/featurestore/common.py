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

import mlrun
from mlrun.utils import parse_versioned_object_uri
from ..config import config

feature_separator = "."


def parse_features(vector):
    """parse and validate feature list (from vector) and add metadata from feature sets """
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

    for feature in vector.spec.features:
        feature_set, feature_name, alias = _parse_feature_string(feature)
        if feature_set not in feature_set_objects.keys():
            feature_set_objects[feature_set] = get_feature_set_by_uri(
                feature_set, vector.metadata.project
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
            if name in feature_set.status.stats:
                vector.status.stats[field_name] = feature_set.status.stats[name]
            if name in feature_set.spec.features.keys():
                vector.status.features[field_name] = feature_set.spec.features[name]

    return feature_set_objects, feature_set_fields


def _parse_feature_string(feature):
    """parse feature string into feature set name, feature name, alias"""
    # expected format: <feature-set>.<name|*>[ as alias]
    if feature_separator not in feature:
        raise ValueError(
            f"feature {feature} must be in the form feature-set.feature[ as alias]"
        )
    splitted = feature.split(feature_separator)
    feature_set = splitted[0]
    feature_name = splitted[1]
    splitted = feature_name.split(" as ")
    if len(splitted) > 1:
        return feature_set.strip(), splitted[0].strip(), splitted[1].strip()
    return feature_set.strip(), feature_name.strip(), None


def get_feature_set_by_uri(uri, project):
    """get feature set object from db by uri"""
    db = mlrun.get_run_db()
    default_project = project or config.default_project
    project, name, tag, uid = parse_versioned_object_uri(uri, default_project)
    return db.get_feature_set(name, project, tag, uid)


def get_feature_vector_by_uri(uri):
    """get feature vector object from db by uri"""
    db = mlrun.get_run_db()
    project, name, tag, uid = parse_versioned_object_uri(uri, config.default_project)
    return db.get_feature_vector(name, project, tag, uid)
