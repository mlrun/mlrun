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
expected_message = f"in the form feature-set{feature_separator}feature[ as alias]"


def parse_feature_string(feature):
    """parse feature string into feature set name, feature name, alias"""
    # expected format: <feature-set>.<name|*>[ as alias]
    if feature_separator not in feature:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"feature {feature} must be {expected_message}"
        )
    splitted = feature.split(feature_separator)
    if len(splitted) > 2:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"feature {feature} must be {expected_message}, cannot have more than one '.'"
        )
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
