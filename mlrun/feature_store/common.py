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
import typing
from copy import copy

import mlrun
import mlrun.errors
from mlrun.api.schemas import AuthorizationVerificationInput
from mlrun.runtimes import BaseRuntime
from mlrun.runtimes.function_reference import FunctionReference
from mlrun.runtimes.utils import enrich_function_from_dict
from mlrun.utils import StorePrefix, logger, mlconf, parse_versioned_object_uri

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


def parse_feature_set_uri(uri, project=None):
    """get feature set object from db by uri"""
    default_project = project or config.default_project

    # parse store://.. uri
    if mlrun.datastore.is_store_uri(uri):
        prefix, new_uri = mlrun.datastore.parse_store_uri(uri)
        if prefix != StorePrefix.FeatureSet:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"provided store uri ({uri}) does not represent a feature set (prefix={prefix})"
            )
        uri = new_uri

    return parse_versioned_object_uri(uri, default_project)


def get_feature_set_by_uri(uri, project=None):
    """get feature set object from db by uri"""
    db = mlrun.get_run_db()
    project, name, tag, uid = parse_feature_set_uri(uri, project)
    resource = (
        mlrun.api.schemas.AuthorizationResourceTypes.feature_set.to_resource_string(
            project, "feature-set"
        )
    )

    auth_input = AuthorizationVerificationInput(
        resource=resource, action=mlrun.api.schemas.AuthorizationAction.read
    )
    db.verify_authorization(auth_input)

    return db.get_feature_set(name, project, tag, uid)


def get_feature_vector_by_uri(uri, project=None, update=True):
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

    resource = (
        mlrun.api.schemas.AuthorizationResourceTypes.feature_vector.to_resource_string(
            project, "feature-vector"
        )
    )

    if update:
        auth_input = AuthorizationVerificationInput(
            resource=resource, action=mlrun.api.schemas.AuthorizationAction.update
        )
    else:
        auth_input = AuthorizationVerificationInput(
            resource=resource, action=mlrun.api.schemas.AuthorizationAction.read
        )

    db.verify_authorization(auth_input)

    return db.get_feature_vector(name, project, tag, uid)


def verify_feature_set_permissions(
    feature_set, action: mlrun.api.schemas.AuthorizationAction
):
    project, _, _, _ = parse_feature_set_uri(feature_set.uri)

    resource = (
        mlrun.api.schemas.AuthorizationResourceTypes.feature_set.to_resource_string(
            project, "feature-set"
        )
    )
    db = feature_set._get_run_db()

    auth_input = AuthorizationVerificationInput(resource=resource, action=action)
    db.verify_authorization(auth_input)


def verify_feature_set_exists(feature_set):
    db = feature_set._get_run_db()
    project, uri, tag, _ = parse_feature_set_uri(feature_set.uri)

    try:
        fset = db.get_feature_set(feature_set.metadata.name, project, tag)
        if not fset.spec.features:
            raise mlrun.errors.MLRunNotFoundError(f"feature set {uri} is empty")
    except mlrun.errors.MLRunNotFoundError:
        raise mlrun.errors.MLRunNotFoundError(f"feature set {uri} does not exist")


def verify_feature_vector_permissions(
    feature_vector, action: mlrun.api.schemas.AuthorizationAction
):
    project = feature_vector._metadata.project or mlconf.default_project

    resource = (
        mlrun.api.schemas.AuthorizationResourceTypes.feature_vector.to_resource_string(
            project, "feature-vector"
        )
    )

    db = mlrun.get_run_db()
    auth_input = AuthorizationVerificationInput(resource=resource, action=action)
    db.verify_authorization(auth_input)


class RunConfig:
    """class for holding function and run specs for jobs and serving functions"""

    def __init__(
        self,
        function: typing.Union[str, FunctionReference, BaseRuntime] = None,
        local: bool = None,
        image: str = None,
        kind: str = None,
        handler: str = None,
        parameters: dict = None,
        watch: bool = None,
        owner=None,
        credentials: typing.Optional[mlrun.model.Credentials] = None,
        code: str = None,
        requirements: typing.Union[str, typing.List[str]] = None,
        extra_spec: dict = None,
        auth_info=None,
    ):
        """class for holding function and run specs for jobs and serving functions

        when running feature ingestion or merging tasks we use the RunConfig class to pass
        the desired function and job configuration.
        the apply() method is used to set resources like volumes, the with_secret() method adds secrets

        Most attributes are optional, if not specified a proper default value will be set

        examples::

            # config for local run emulation
            config = RunConfig(local=True)

            # config for using empty/default code
            config = RunConfig()

            # config for using .py/.ipynb file with image and extra package requirements
            config = RunConfig("mycode.py", image="mlrun/mlrun", requirements=["spacy"])

            # config for using function object
            function = mlrun.import_function("hub://some_function")
            config = RunConfig(function)

        :param function:    this can be function uri or function object or path to function code (.py/.ipynb)
                            or a :py:class:`~mlrun.runtimes.function_reference.FunctionReference`
                            the function define the code, dependencies, and resources
        :param local:       use True to simulate local job run or mock service
        :param image:       function container image
        :param kind:        function runtime kind (job, serving, spark, ..), required when function points to code
        :param handler:     the function handler to execute (for jobs or nuclio)
        :param parameters:  job parameters
        :param watch:       in batch jobs will wait for the job completion and print job logs to the console
        :param owner:       job owner
        :param credentials: job credentials
        :param code:        function source code (as string)
        :param requirements: python requirements file path or list of packages
        :param extra_spec:  additional dict with function spec fields/values to add to the function
        :param auth_info:   authentication info. *For internal use* when running on server
        """
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
        self.owner = owner
        self.credentials = credentials
        self.code = code or ""
        self.requirements = requirements
        self.extra_spec = extra_spec
        self.auth_info = auth_info

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
        """internal, generate function object"""
        if isinstance(self.function, FunctionReference):
            if self.code:
                self.function.code = self.code
            if self.requirements:
                self.function.requirements = self.requirements
            if self.extra_spec:
                self.function.spec = self.extra_spec
            function = self.function.to_function(default_kind, default_image)
        elif hasattr(self.function, "apply"):
            function = copy(self.function)
            if self.code:
                function.with_code(body=self.code)
            if self.requirements:
                self.function.with_requirements(self.requirements)
            if self.extra_spec:
                self.function = enrich_function_from_dict(
                    self.function, self.extra_spec
                )
            function.spec.image = function.spec.image or default_image
        else:
            function = FunctionReference(
                self.function,
                image=self.image,
                kind=self.kind,
                code=self.code,
                requirements=self.requirements,
                spec=self.extra_spec,
            ).to_function(default_kind, default_image)

        if not function.is_deployed():
            # todo: handle build for job functions
            logger.warn("cannot run function, it must be built/deployed first")

        for modifier in self._modifiers:
            function.apply(modifier)
        function.metadata.credentials = self.credentials
        return function

    def copy(self):
        return copy(self)
