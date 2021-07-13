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
import mlrun.errors
from mlrun.runtimes.function_reference import FunctionReference
from mlrun.utils import StorePrefix, parse_versioned_object_uri

from ..config import config

project_separator = "/"
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


def parse_project_name_from_feature_string(feature):
    """parse feature string into project name and feature"""
    # expected format: <project-name>/<feature>
    if project_separator not in feature:
        return None, feature

    splitted = feature.split(project_separator)
    if len(splitted) > 2:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"feature {feature} must be {expected_message}, cannot have more than one '/'"
        )
    project_name = splitted[0]
    feature_name = splitted[1]
    return project_name.strip(), feature_name.strip()


def get_feature_set_by_uri(uri, project=None):
    """get feature set object from db by uri"""
    db = mlrun.get_run_db()
    default_project = project or config.default_project

    # parse store://.. uri
    if mlrun.datastore.is_store_uri(uri):
        prefix, new_uri = mlrun.datastore.parse_store_uri(uri)
        if prefix != StorePrefix.FeatureSet:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"provided store uri ({uri}) does not represent a feature set (prefix={prefix})"
            )
        uri = new_uri

    project, name, tag, uid = parse_versioned_object_uri(uri, default_project)
    return db.get_feature_set(name, project, tag, uid)


def get_feature_vector_by_uri(uri, project=None):
    """get feature vector object from db by uri"""
    db = mlrun.get_run_db()
    default_project = project or config.default_project

    # parse store://.. uri
    if mlrun.datastore.is_store_uri(uri):
        prefix, new_uri = mlrun.datastore.parse_store_uri(uri)
        if prefix != StorePrefix.FeatureVector:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"provided store uri ({uri}) does not represent a feature vector (prefix={prefix})"
            )
        uri = new_uri

    project, name, tag, uid = parse_versioned_object_uri(uri, default_project)
    return db.get_feature_vector(name, project, tag, uid)


class RunConfig:
    """remote job/service run configuration

    when running feature ingestion or merging tasks we use the RunConfig class to pass
    the desired function and job configuration.
    the apply() method is used to set resources like volumes, the with_secret() method adds secrets

    Parameters:
        function:      this can be function uri or function object or path to function code (.py/.ipynb) or
                       a :py:class:`~mlrun.runtimes.function_reference.FunctionReference`
                       the function define the code, dependencies, and resources
        image (str):   function container image
        kind (str):    mlrun function kind (job, serving, remote-spark, ..), required when function points to code
        handler (str): the function handler to execute
        local (bool):  use True to simulate local job run or mock service
        watch (bool):  in batch jobs will wait for the job completion and print job logs to the console
        parameters (dict): optional parameters
    """

    def __init__(
        self,
        function=None,
        local=None,
        image=None,
        kind=None,
        handler=None,
        parameters=None,
        watch=None,
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
        self.watch = True if watch is None else watch

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        if function and not (
            isinstance(function, (str, FunctionReference)) or hasattr(function, "apply")
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "function must be a uri (string) or mlrun function object/reference"
            )
        self._function = function

    def apply(self, modifier):
        """apply a modifier to add/set function resources like volumes

        example::

            run_config.apply(mlrun.platforms.auto_mount())
        """
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
