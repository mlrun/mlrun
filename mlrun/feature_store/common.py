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
from copy import copy

import mlrun
from mlrun.runtimes.function_reference import FunctionReference
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


def get_feature_set_by_uri(uri, project=None):
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


class RunConfig:
    def __init__(
        self,
        function=None,
        local=None,
        image=None,
        kind=None,
        handler=None,
        parameters=None,
    ):
        self._function = None
        self._modifiers = []
        self.secret_sources = []

        self.function = function
        self.local = local
        self.image = image
        self.kind = kind
        self.handler = handler
        self.parameters = parameters or {}
        self.watch = True

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        if function and not (
            isinstance(function, (str, FunctionReference))
            or hasattr(self.function, "apply")
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "function must be a uri (string) or mlrun function object/reference"
            )
        self._function = function

    def apply(self, modifier):
        self._modifiers.append(modifier)
        return self

    def with_secret(self, kind, source):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in jobs, example::

            run_config.with_secrets('file', 'file.txt')
            run_config.with_secrets('inline', {'key': 'val'})
            run_config.with_secrets('env', 'ENV1,ENV2')
            run_config.with_secrets('vault', ['secret1', 'secret2'...])

        :param kind:   secret type (file, inline, env, vault)
        :param source: secret data or link (see example)

        :returns: This (self) object
        """

        self.secret_sources.append({"kind": kind, "source": source})
        return self

    def to_function(self, default_kind=None, default_image=None):
        if isinstance(self.function, FunctionReference):
            function = self.function.to_function(default_kind)
        elif hasattr(self.function, "apply"):
            function = copy(self.function)
        else:
            function = FunctionReference(
                self.function, image=self.image, kind=self.kind
            ).to_function(default_kind)

        function.spec.image = function.spec.image or default_image
        for modifier in self._modifiers:
            function.apply(modifier)
        return function

    def copy(self):
        return copy(self)
