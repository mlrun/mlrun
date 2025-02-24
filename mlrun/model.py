# Copyright 2023 Iguazio
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

import inspect
import json
import pathlib
import re
import time
import typing
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from os import environ
from typing import Any, Optional, Union

import pydantic.v1.error_wrappers

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas.notification
import mlrun.utils.regex

from .utils import (
    dict_to_json,
    dict_to_yaml,
    get_artifact_target,
    logger,
    template_artifact_path,
)

# Changing {run_id} will break and will not be backward compatible.
RUN_ID_PLACE_HOLDER = "{run_id}"  # IMPORTANT: shouldn't be changed.


class ModelObj:
    _dict_fields = []
    # Bellow attributes are used in to_dict method
    # Fields to strip from the object by default if strip=True
    _default_fields_to_strip = []
    # Fields that will be serialized by the object's _serialize_field method
    _fields_to_serialize = []
    # Fields that will be enriched by the object's _enrich_field method
    _fields_to_enrich = []
    # Fields that will be ignored by the object's _is_valid_field_value_for_serialization method
    _fields_to_skip_validation = []

    @staticmethod
    def _verify_list(param, name):
        if not isinstance(param, list):
            raise ValueError(f"Parameter {name} must be a list")

    @staticmethod
    def _verify_dict(param, name, new_type=None):
        if (
            param is not None
            and not isinstance(param, dict)
            and not hasattr(param, "to_dict")
        ):
            raise ValueError(f"Parameter {name} must be a dict or object")
        if new_type and (isinstance(param, dict) or param is None):
            return new_type.from_dict(param)
        return param

    @mlrun.utils.filter_warnings("ignore", FutureWarning)
    def to_dict(
        self,
        fields: Optional[list] = None,
        exclude: Optional[list] = None,
        strip: bool = False,
    ) -> dict:
        """
        Convert the object to a dict

        :param fields:  A list of fields to include in the dictionary. If not provided, the default value is taken
            from `self._dict_fields` or from the object __init__ params.
        :param exclude: A list of fields to exclude from the dictionary.
        :param strip:  If True, the object's `_default_fields_to_strip` attribute is appended to the exclude list.
            Strip purpose is to remove fields that are context / environment specific and not required for actually
            define the object.

        :return: A dictionary representation of the object.
        """
        struct = {}

        fields = self._resolve_initial_to_dict_fields(fields)
        fields_to_exclude = exclude or []
        if strip:
            fields_to_exclude += self._default_fields_to_strip

        # fields_to_save is built from the fields list minus the fields to exclude minus the fields that requires
        # serialization and enrichment (because they will be added later to the struct)
        fields_to_save = (
            set(fields)
            - set(fields_to_exclude)
            - set(self._fields_to_serialize)
            - set(self._fields_to_enrich)
        )

        # Iterating over the fields to save and adding them to the struct
        for field_name in fields_to_save:
            field_value = getattr(self, field_name, None)
            if self._is_valid_field_value_for_serialization(
                field_name, field_value, strip
            ):
                # If the field value has attribute to_dict, we call it.
                # If one of the attributes is a third party object that has to_dict method (such as k8s objects), then
                # add it to the object's _fields_to_serialize attribute and handle it in the _serialize_field method.
                if hasattr(field_value, "to_dict"):
                    # TODO: Allow passing fields to exclude from the parent object to the child object
                    #  e.g.: run.to_dict(exclude=["status.artifacts"])
                    field_value = field_value.to_dict(strip=strip)
                    if self._is_valid_field_value_for_serialization(
                        field_name, field_value, strip
                    ):
                        struct[field_name] = field_value
                else:
                    struct[field_name] = field_value

        # Subtracting the fields_to_exclude from the fields_to_serialize because if we want to exclude a field there
        # is no need to serialize it.
        fields_to_serialize = list(
            set(self._fields_to_serialize) - set(fields_to_exclude)
        )
        self._resolve_field_value_by_method(
            struct, self._serialize_field, fields_to_serialize, strip
        )

        # Subtracting the fields_to_exclude from the fields_to_enrich because if we want to exclude a field there
        # is no need to enrich it.
        fields_to_enrich = list(set(self._fields_to_enrich) - set(fields_to_exclude))
        self._resolve_field_value_by_method(
            struct, self._enrich_field, fields_to_enrich, strip
        )

        self._apply_enrichment_before_to_dict_completion(struct, strip=strip)
        return struct

    def _resolve_initial_to_dict_fields(self, fields: Optional[list] = None) -> list:
        """
        Resolve fields to be used in to_dict method.
        If fields is None, use `_dict_fields` attribute of the object.
        If fields is None and `_dict_fields` is empty, use the object's __init__ parameters.
        :param fields: List of fields to iterate over.

        :return: List of fields to iterate over.
        """
        return (
            fields
            or self._dict_fields
            or list(inspect.signature(self.__init__).parameters.keys())
        )

    def _is_valid_field_value_for_serialization(
        self, field_name: str, field_value: str, strip: bool = False
    ) -> bool:
        """
        Check if the field value is valid for serialization.
        If field name is in `_fields_to_skip_validation` attribute, skip validation and return True.
        If strip is False skip validation and return True.
        If field value is None or empty dict/list, then no need to store it.
        :param field_name:  Field name.
        :param field_value: Field value.

        :return: True if the field value is valid for serialization, False otherwise.
        """
        if field_name in self._fields_to_skip_validation:
            return True
        # TODO: remove when Runtime initialization will be refactored and enrichment will be moved to BE
        # if not strip:
        #     return True

        return field_value is not None and not (
            (isinstance(field_value, dict) or isinstance(field_value, list))
            and not field_value
        )

    def _resolve_field_value_by_method(
        self,
        struct: dict,
        method: typing.Callable,
        fields: Optional[typing.Union[list, set]] = None,
        strip: bool = False,
    ) -> dict:
        for field_name in fields:
            field_value = method(struct=struct, field_name=field_name, strip=strip)
            if self._is_valid_field_value_for_serialization(
                field_name, field_value, strip
            ):
                struct[field_name] = field_value
        return struct

    def _serialize_field(
        self, struct: dict, field_name: Optional[str] = None, strip: bool = False
    ) -> typing.Any:
        # We pull the field from self and not from struct because it was excluded from the struct when looping over
        # the fields to save.
        return getattr(self, field_name, None)

    def _enrich_field(
        self, struct: dict, field_name: Optional[str] = None, strip: bool = False
    ) -> typing.Any:
        # We first try to pull from struct because the field might have been already serialized and if not,
        # we pull from self
        return struct.get(field_name, None) or getattr(self, field_name, None)

    def _apply_enrichment_before_to_dict_completion(
        self, struct: dict, strip: bool = False
    ) -> dict:
        return struct

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: Optional[dict] = None
    ):
        """create an object from a python dictionary"""
        struct = {} if struct is None else struct
        deprecated_fields = deprecated_fields or {}
        fields = fields or cls._dict_fields
        if not fields:
            fields = list(inspect.signature(cls.__init__).parameters.keys())
        new_obj = cls()
        if struct:
            # we are looping over the fields to save the same order and behavior in which the class
            # initialize the attributes
            for field in fields:
                # we want to set the field only if the field exists in struct
                if field in struct:
                    field_val = struct.get(field, None)
                    if field not in deprecated_fields:
                        setattr(new_obj, field, field_val)

            for deprecated_field, new_field in deprecated_fields.items():
                field_value = struct.get(new_field) or struct.get(deprecated_field)
                if field_value:
                    setattr(new_obj, new_field, field_value)

        return new_obj

    def to_yaml(self, exclude=None, strip: bool = False) -> str:
        """convert the object to yaml

        :param exclude: list of fields to exclude from the yaml
        :param strip:   if True, strip fields that are not required for actually define the object
        """
        return dict_to_yaml(self.to_dict(exclude=exclude, strip=strip))

    def to_json(self, exclude=None, strip: bool = False):
        """convert the object to json

        :param exclude: list of fields to exclude from the json
        :param strip:   if True, strip fields that are not required for actually define the object
        """
        return dict_to_json(self.to_dict(exclude=exclude, strip=strip))

    def to_str(self):
        """convert the object to string (with dict layout)"""
        return self.__str__()

    def __str__(self):
        return str(self.to_dict())

    def copy(self):
        """create a copy of the object"""
        return deepcopy(self)


# model class for building ModelObj dictionaries
class ObjectDict:
    kind = "object_dict"

    def __init__(self, classes_map, default_kind=""):
        self._children = OrderedDict()
        self._default_kind = default_kind
        self._classes_map = classes_map

    def values(self):
        return self._children.values()

    def keys(self):
        return self._children.keys()

    def items(self):
        return self._children.items()

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        yield from self._children.keys()

    def __getitem__(self, name):
        return self._children[name]

    def __setitem__(self, key, item):
        self._children[key] = self._get_child_object(item, key)

    def __delitem__(self, key):
        del self._children[key]

    def update(self, key, item):
        child = self._get_child_object(item, key)
        self._children[key] = child
        return child

    def to_dict(self, strip: bool = False):
        return {k: v.to_dict(strip=strip) for k, v in self._children.items()}

    @classmethod
    def from_dict(cls, classes_map: dict, children=None, default_kind=""):
        if children is None:
            return cls(classes_map, default_kind)
        if not isinstance(children, dict):
            raise ValueError("children must be a dict")

        new_obj = cls(classes_map, default_kind)
        for name, child in children.items():
            obj_name = name
            if hasattr(child, "name") and child.name is not None:
                obj_name = child.name
            elif isinstance(child, dict) and "name" in child:
                obj_name = child["name"]
            child_obj = new_obj._get_child_object(child, obj_name)
            new_obj._children[name] = child_obj

        return new_obj

    def _get_child_object(self, child, name):
        if hasattr(child, "kind") and child.kind in self._classes_map.keys():
            child.name = name
            return child
        elif isinstance(child, dict):
            kind = child.get("kind", self._default_kind)
            if kind not in self._classes_map.keys():
                raise ValueError(f"illegal object kind {kind}")
            child_obj = self._classes_map[kind].from_dict(child)
            child_obj.name = name
            return child_obj
        else:
            raise ValueError(f"illegal child (should be dict or child kind), {child}")

    def to_yaml(self):
        return dict_to_yaml(self.to_dict())

    def to_json(self):
        return dict_to_json(self.to_dict())

    def to_str(self):
        return self.__str__()

    def __str__(self):
        return str(self.to_dict())

    def copy(self):
        return deepcopy(self)


class ObjectList:
    def __init__(self, child_class):
        self._children = OrderedDict()
        self._child_class = child_class

    def values(self):
        return self._children.values()

    def keys(self):
        return self._children.keys()

    def items(self):
        return self._children.items()

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        yield from self._children.values()

    def __getitem__(self, name):
        if isinstance(name, int):
            return list(self._children.values())[name]
        return self._children[name]

    def __setitem__(self, key, item):
        self.update(item, key)

    def __delitem__(self, key):
        del self._children[key]

    def to_dict(self, strip: bool = False):
        # method used by ModelObj class to serialize the object to nested dict
        return [t.to_dict(strip=strip) for t in self._children.values()]

    @classmethod
    def from_list(cls, child_class, children=None):
        if children is None:
            return cls(child_class)
        if not isinstance(children, list):
            raise ValueError("states must be a list")

        new_obj = cls(child_class)
        for child in children:
            name, child_obj = new_obj._get_child_object(child)
            new_obj._children[name] = child_obj
        return new_obj

    def _get_child_object(self, child):
        if isinstance(child, self._child_class):
            return child.name, child
        elif isinstance(child, dict):
            if "name" not in child.keys():
                raise ValueError("illegal object no 'name' field")
            child_obj = self._child_class.from_dict(child)
            return child_obj.name, child_obj
        else:
            raise ValueError(f"illegal child (should be dict or child kind), {child}")

    def update(self, child, name=None):
        object_name, child_obj = self._get_child_object(child)
        child_obj.name = name or object_name
        self._children[child_obj.name] = child_obj
        return child_obj

    def move_to_end(self, child, last=True):
        self._children.move_to_end(child, last)

    def update_list(self, object_list: "ObjectList", push_at_start: bool = False):
        if push_at_start:
            self._children = OrderedDict(
                list(object_list._children.items()) + list(self._children.items())
            )
        else:
            self._children = OrderedDict(
                list(self._children.items()) + list(object_list._children.items())
            )


class Credentials(ModelObj):
    generate_access_key = "$generate"
    secret_reference_prefix = "$ref:"

    def __init__(
        self,
        access_key: Optional[str] = None,
    ):
        self.access_key = access_key


class BaseMetadata(ModelObj):
    _default_fields_to_strip = ModelObj._default_fields_to_strip + [
        "hash",
        "uid",
        # Below are environment specific fields, no need to keep when stripping
        "namespace",
        "project",
        "labels",
        "annotations",
        "credentials",
        # Below are state fields, no need to keep when stripping
        "updated",
    ]

    def __init__(
        self,
        name=None,
        tag=None,
        hash=None,
        namespace=None,
        project=None,
        labels=None,
        annotations=None,
        categories=None,
        updated=None,
        credentials=None,
        uid=None,
    ):
        self.name = name
        self.tag = tag
        self.hash = hash
        self.uid = uid
        self.namespace = namespace
        self.project = project or ""
        self.labels = labels or {}
        self.categories = categories or []
        self.annotations = annotations or {}
        self.updated = updated
        self._credentials = None
        self.credentials = credentials

    @property
    def credentials(self) -> Credentials:
        return self._credentials

    @credentials.setter
    def credentials(self, credentials):
        self._credentials = self._verify_dict(credentials, "credentials", Credentials)


class ImageBuilder(ModelObj):
    """An Image builder"""

    def __init__(
        self,
        functionSourceCode=None,  # noqa: N803 - should be "snake_case", kept for BC
        source=None,
        image=None,
        base_image=None,
        commands=None,
        extra=None,
        secret=None,
        code_origin=None,
        registry=None,
        load_source_on_run=None,
        origin_filename=None,
        with_mlrun=None,
        auto_build=None,
        requirements: Optional[list] = None,
        extra_args=None,
        builder_env=None,
        source_code_target_dir=None,
    ):
        self.functionSourceCode = functionSourceCode  #: functionSourceCode
        self.codeEntryType = ""  #: codeEntryType
        self.codeEntryAttributes = ""  #: codeEntryAttributes
        self.source = source  #: source
        self.code_origin = code_origin  #: code_origin
        self.origin_filename = origin_filename
        self.image = image  #: image
        self.base_image = base_image  #: base_image
        self.commands = commands or []  #: commands
        self.extra = extra  #: extra
        self.extra_args = extra_args  #: extra args
        self.builder_env = builder_env  #: builder env
        self.secret = secret  #: secret
        self.registry = registry  #: registry
        self.load_source_on_run = load_source_on_run  #: load_source_on_run
        self.with_mlrun = with_mlrun  #: with_mlrun
        self.auto_build = auto_build  #: auto_build
        self.build_pod = None
        self.requirements = requirements or []  #: pip requirements
        self.source_code_target_dir = source_code_target_dir or None

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        if source and not (
            source.startswith("git://")
            # lenient check for file extension because we support many file types locally and remotely
            or pathlib.Path(source).suffix
            or source in [".", "./"]
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"source ({source}) must be a compressed (tar.gz / zip) file, a git repo, "
                f"a file path or in the project's context (.)"
            )

        self._source = source

    def build_config(
        self,
        image="",
        base_image=None,
        commands: Optional[list] = None,
        secret=None,
        source=None,
        extra=None,
        load_source_on_run=None,
        with_mlrun=None,
        auto_build=None,
        requirements=None,
        requirements_file=None,
        overwrite=False,
        builder_env=None,
        extra_args=None,
        source_code_target_dir=None,
    ):
        if image:
            self.image = image
        if base_image:
            self.base_image = base_image
        if commands:
            self.with_commands(commands, overwrite=overwrite)
        if requirements or requirements_file:
            self.with_requirements(requirements, requirements_file, overwrite=overwrite)
        if extra:
            self.extra = extra
        if secret is not None:
            self.secret = secret
        if source:
            self.source = source
        if load_source_on_run:
            self.load_source_on_run = load_source_on_run
        if with_mlrun is not None:
            self.with_mlrun = with_mlrun
        if auto_build:
            self.auto_build = auto_build
        if builder_env:
            self.builder_env = builder_env
        if extra_args:
            self.extra_args = extra_args
        if source_code_target_dir:
            self.source_code_target_dir = source_code_target_dir

    def with_commands(
        self,
        commands: list[str],
        overwrite: bool = False,
    ):
        """add commands to build spec.

        :param commands:  list of commands to run during build
        :param overwrite: whether to overwrite the existing commands or add to them (the default)

        :return: function object
        """
        if not isinstance(commands, list) or not all(
            isinstance(item, str) for item in commands
        ):
            raise ValueError("commands must be a string list")
        if not self.commands or overwrite:
            self.commands = commands
        else:
            # add commands to existing build commands
            for command in commands:
                if command not in self.commands:
                    self.commands.append(command)
            # using list(set(x)) won't retain order,
            # solution inspired from https://stackoverflow.com/a/17016257/8116661
            self.commands = list(dict.fromkeys(self.commands))

    def with_requirements(
        self,
        requirements: Optional[list[str]] = None,
        requirements_file: str = "",
        overwrite: bool = False,
    ):
        """add package requirements from file or list to build spec.

        :param requirements:        a list of python packages
        :param requirements_file:   path to a python requirements file
        :param overwrite:           overwrite existing requirements,
                                    when False (default) will append to existing requirements
        :return: function object
        """
        requirements = requirements or []
        self._verify_list(requirements, "requirements")
        resolved_requirements = self._resolve_requirements(
            requirements, requirements_file
        )
        requirements = self.requirements or [] if not overwrite else []

        # make sure we don't append the same line twice
        for requirement in resolved_requirements:
            if requirement not in requirements:
                requirements.append(requirement)

        self.requirements = requirements

    @staticmethod
    def _resolve_requirements(requirements: list, requirements_file: str = "") -> list:
        requirements = requirements or []
        requirements_to_resolve = []

        # handle the requirements_file argument
        if requirements_file:
            with open(requirements_file) as fp:
                requirements_to_resolve.extend(fp.read().splitlines())

        # handle the requirements argument
        requirements_to_resolve.extend(requirements)

        requirements = []
        for requirement in requirements_to_resolve:
            # clean redundant leading and trailing whitespaces
            requirement = requirement.strip()

            # ignore empty lines
            # ignore comments
            if not requirement or requirement.startswith("#"):
                continue

            # ignore inline comments as well
            inline_comment = requirement.split(" #")
            if len(inline_comment) > 1:
                requirement = inline_comment[0].strip()

            requirements.append(requirement)

        return requirements


class Notification(ModelObj):
    """Notification object

    :param kind: notification implementation kind - slack, webhook, etc. See
        :py:class:`mlrun.common.schemas.notification.NotificationKind`
    :param name: for logging and identification
    :param message: message content in the notification
    :param severity: severity to display in the notification
    :param when: list of statuses to trigger the notification: 'running', 'completed', 'error'
    :param condition: optional condition to trigger the notification, a jinja2 expression that can use run data
                      to evaluate if the notification should be sent in addition to the 'when' statuses.
                      e.g.: '{{ run["status"]["results"]["accuracy"] < 0.9}}'
    :param params: Implementation specific parameters for the notification implementation (e.g. slack webhook url,
                   git repository details, etc.)
    :param secret_params: secret parameters for the notification implementation, same as params but will be stored
                          in a k8s secret and passed as a secret reference to the implementation.
    :param status: notification status - pending, sent, error
    :param sent_time: time the notification was sent
    :param reason: failure reason if the notification failed to send
    """

    def __init__(
        self,
        kind: mlrun.common.schemas.notification.NotificationKind = (
            mlrun.common.schemas.notification.NotificationKind.slack
        ),
        name=None,
        message=None,
        severity: mlrun.common.schemas.notification.NotificationSeverity = (
            mlrun.common.schemas.notification.NotificationSeverity.INFO
        ),
        when=None,
        condition=None,
        secret_params=None,
        params=None,
        status=None,
        sent_time=None,
        reason=None,
    ):
        self.kind = kind
        self.name = name or ""
        self.message = message or ""
        self.severity = severity
        self.when = when or ["completed"]
        self.condition = condition or ""
        self.secret_params = secret_params or {}
        self.params = params or {}
        self.status = status
        self.sent_time = sent_time
        self.reason = reason

        self.validate_notification()

    def validate_notification(self):
        try:
            mlrun.common.schemas.notification.Notification(**self.to_dict())
        except pydantic.v1.error_wrappers.ValidationError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid notification object"
            ) from exc

        # validate that size of notification secret_params doesn't exceed 1 MB,
        # due to k8s default secret size limitation.
        # a buffer of 100 KB is added to the size to account for the size of the secret metadata
        if (
            len(json.dumps(self.secret_params))
            > mlrun.common.schemas.notification.NotificationLimits.max_params_size.value
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Notification params size exceeds max size of 1 MB"
            )

    def validate_notification_params(self, default_notification_params=None):
        default_notification_params = default_notification_params or {}
        notification_type = mlrun.utils.notifications.NotificationTypes(self.kind)
        notification_class = notification_type.get_notification()
        secret_params = self.secret_params or {}
        params = self.params or {}
        default_params = default_notification_params.get(notification_type, {})
        params = notification_class.enrich_default_params(params, default_params)
        # if the secret_params are already masked - no need to validate
        params_secret = secret_params.get("secret", "")
        if params_secret:
            if len(secret_params) > 1:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "When the 'secret' key is present, 'secret_params' should not contain any other keys."
                )
            return

        if not secret_params and not params:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Both 'secret_params' and 'params' are empty, at least one must be defined."
            )

        notification_class.validate_params(secret_params | params)

    def enrich_unmasked_secret_params_from_project_secret(self):
        """
        Fill the notification secret params from the project secret.
        We are using this function instead of unmask_secret_params_from_project_secret when we run inside the
        workflow runner pod that doesn't have access to the k8s secrets (but have access to the project secret)
        """
        secret = self.secret_params.get("secret")
        if secret:
            secret_value = mlrun.get_secret_or_env(secret)
            if secret_value:
                try:
                    self.secret_params = json.loads(secret_value)
                except ValueError as exc:
                    raise mlrun.errors.MLRunValueError(
                        "Failed to parse secret value"
                    ) from exc

    @staticmethod
    def validate_notification_uniqueness(notifications: list["Notification"]):
        """Validate that all notifications in the list are unique by name"""
        names = [notification.name for notification in notifications]
        if len(names) != len(set(names)):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Notification names must be unique"
            )


class RunMetadata(ModelObj):
    """Run metadata"""

    def __init__(
        self,
        uid=None,
        name=None,
        project=None,
        labels=None,
        annotations=None,
        iteration=None,
    ):
        self.uid = uid
        self._iteration = iteration
        self.name = name
        self.project = project
        self.labels = labels or {}
        self.annotations = annotations or {}

    @property
    def iteration(self):
        return self._iteration or 0

    @iteration.setter
    def iteration(self, iteration):
        self._iteration = iteration

    def is_workflow_runner(self):
        if not self.labels:
            return False
        return (
            self.labels.get(mlrun_constants.MLRunInternalLabels.job_type, "")
            == "workflow-runner"
        )


class HyperParamStrategies:
    grid = "grid"
    list = "list"
    random = "random"
    custom = "custom"

    @staticmethod
    def all():
        return [
            HyperParamStrategies.grid,
            HyperParamStrategies.list,
            HyperParamStrategies.random,
            HyperParamStrategies.custom,
        ]


class HyperParamOptions(ModelObj):
    """Hyper Parameter Options

    Parameters:
        param_file (str):                   hyper params input file path/url, instead of inline
        strategy (HyperParamStrategies):    hyper param strategy - grid, list or random
        selector (str):                     selection criteria for best result ([min|max.]<result>), e.g. max.accuracy
        stop_condition (str):               early stop condition e.g. "accuracy > 0.9"
        parallel_runs (int):                number of param combinations to run in parallel (over Dask)
        dask_cluster_uri (str):             db uri for a deployed dask cluster function, e.g. db://myproject/dask
        max_iterations (int):               max number of runs (in random strategy)
        max_errors (int):                   max number of child runs errors for the overall job to fail
        teardown_dask (bool):               kill the dask cluster pods after the runs
    """

    def __init__(
        self,
        param_file=None,
        strategy: typing.Optional[HyperParamStrategies] = None,
        selector=None,
        stop_condition=None,
        parallel_runs=None,
        dask_cluster_uri=None,
        max_iterations=None,
        max_errors=None,
        teardown_dask=None,
    ):
        self.param_file = param_file
        self.strategy = strategy
        self.selector = selector
        self.stop_condition = stop_condition
        self.max_iterations = max_iterations
        self.max_errors = max_errors
        self.parallel_runs = parallel_runs
        self.dask_cluster_uri = dask_cluster_uri
        self.teardown_dask = teardown_dask

    def validate(self):
        if self.strategy and self.strategy not in HyperParamStrategies.all():
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"illegal hyper param strategy, use {','.join(HyperParamStrategies.all())}"
            )
        if self.max_iterations and self.strategy != HyperParamStrategies.random:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "max_iterations is only valid in random strategy"
            )


class RunSpec(ModelObj):
    """Run specification"""

    _fields_to_serialize = ModelObj._fields_to_serialize + [
        "handler",
    ]

    def __init__(
        self,
        parameters=None,
        hyperparams=None,
        param_file=None,
        selector=None,
        handler=None,
        inputs=None,
        outputs=None,
        input_path=None,
        output_path=None,
        function=None,
        secret_sources=None,
        data_stores=None,
        strategy=None,
        verbose=None,
        scrape_metrics=None,
        hyper_param_options=None,
        allow_empty_resources=None,
        inputs_type_hints=None,
        returns=None,
        notifications=None,
        state_thresholds=None,
        reset_on_run=None,
        node_selector=None,
    ):
        # A dictionary of parsing configurations that will be read from the inputs the user set. The keys are the inputs
        # keys (parameter names) and the values are the type hint given in the input keys after the colon.
        # Notice: We set it first as empty dictionary as setting the inputs will set it as well in case the type hints
        # were passed in the input keys.
        self._inputs_type_hints = {}

        self._hyper_param_options = None

        # Initialize the inputs and returns properties first and then use their setter methods:
        self._inputs = None
        self.inputs = inputs
        if inputs_type_hints:
            # Override the empty dictionary only if the user passed the parameter:
            self._inputs_type_hints = inputs_type_hints
        self._returns = None
        self.returns = returns

        self._outputs = outputs
        self.hyper_param_options = hyper_param_options
        self.parameters = parameters or {}
        self.hyperparams = hyperparams or {}
        self.param_file = param_file
        self.strategy = strategy
        self.selector = selector
        self.handler = handler
        self.input_path = input_path
        self.output_path = output_path
        self.function = function
        self._secret_sources = secret_sources or []
        self._data_stores = data_stores
        self.verbose = verbose
        self.scrape_metrics = scrape_metrics
        self.allow_empty_resources = allow_empty_resources
        self._notifications = notifications or []
        self.state_thresholds = state_thresholds or {}
        self.reset_on_run = reset_on_run
        self.node_selector = node_selector or {}

    def _serialize_field(
        self, struct: dict, field_name: Optional[str] = None, strip: bool = False
    ) -> Optional[str]:
        # We pull the field from self and not from struct because it was excluded from the struct
        if field_name == "handler":
            if self.handler and isinstance(self.handler, str):
                return self.handler
            return None
        return super()._serialize_field(struct, field_name, strip)

    def is_hyper_job(self):
        param_file = self.param_file or self.hyper_param_options.param_file
        return param_file or self.hyperparams

    @property
    def inputs(self) -> dict[str, str]:
        """
        Get the inputs dictionary. A dictionary of parameter names as keys and paths as values.

        :return: The inputs dictionary.
        """
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: dict[str, str]):
        """
        Set the given inputs in the spec. Inputs can include a type hint string in their keys following a colon, meaning
        following this structure: "<input key : type hint>".

        :exmaple:

        >>> run_spec.inputs = {
        ...     "my_input": "...",
        ...     "my_hinted_input : pandas.DataFrame": "...",
        ... }

        :param inputs: The inputs to set.
        """
        # Check if None, then set and return:
        if inputs is None:
            self._inputs = None
            return

        # Verify it's a dictionary:
        self._inputs = self._verify_dict(inputs, "inputs")

    @property
    def inputs_type_hints(self) -> dict[str, str]:
        """
        Get the input type hints. A dictionary of parameter names as keys and their type hints as values.

        :return: The input type hints dictionary.
        """
        return self._inputs_type_hints

    @inputs_type_hints.setter
    def inputs_type_hints(self, inputs_type_hints: dict[str, str]):
        """
        Set the inputs type hints to parse during a run.

        :param inputs_type_hints: The type hints to set.
        """
        # Verify the given value is a dictionary or None:
        self._inputs_type_hints = self._verify_dict(
            inputs_type_hints, "inputs_type_hints"
        )

    @property
    def returns(self):
        """
        Get the returns list. A list of log hints for returning values.

        :return: The returns list.
        """
        return self._returns

    @returns.setter
    def returns(self, returns: list[Union[str, dict[str, str]]]):
        """
        Set the returns list to log the returning values at the end of a run.

        :param returns: The return list to set.

        :raise MLRunInvalidArgumentError: In case one of the values in the list is invalid.
        """
        # This import is located in the method due to circular imports error.
        from mlrun.package.utils import LogHintUtils

        if returns is None:
            self._returns = None
            return
        self._verify_list(returns, "returns")

        # Validate:
        for log_hint in returns:
            LogHintUtils.parse_log_hint(log_hint=log_hint)

        # Store the results:
        self._returns = returns

    @property
    def hyper_param_options(self) -> HyperParamOptions:
        return self._hyper_param_options

    @hyper_param_options.setter
    def hyper_param_options(self, hyper_param_options):
        self._hyper_param_options = self._verify_dict(
            hyper_param_options, "hyper_param_options", HyperParamOptions
        )

    @property
    def outputs(self) -> list[str]:
        """
        Get the expected outputs. The list is constructed from keys of both the `outputs` and `returns` properties.

        :return: The expected outputs list.
        """
        return self.join_outputs_and_returns(
            outputs=self._outputs, returns=self.returns
        )

    @outputs.setter
    def outputs(self, outputs):
        """
        Set the expected outputs list.

        :param outputs: A list of expected output keys.
        """
        self._verify_list(outputs, "outputs")
        self._outputs = outputs

    @property
    def secret_sources(self):
        return self._secret_sources

    @secret_sources.setter
    def secret_sources(self, secret_sources):
        self._verify_list(secret_sources, "secret_sources")
        self._secret_sources = secret_sources

    @property
    def data_stores(self):
        return self._data_stores

    @data_stores.setter
    def data_stores(self, data_stores):
        self._verify_list(data_stores, "data_stores")
        self._data_stores = data_stores

    @property
    def handler_name(self):
        if self.handler:
            if inspect.isfunction(self.handler):
                return self.handler.__name__
            else:
                return str(self.handler)
        return ""

    @property
    def notifications(self):
        return self._notifications

    @notifications.setter
    def notifications(self, notifications):
        if isinstance(notifications, list):
            self._notifications = ObjectList.from_list(Notification, notifications)
        elif isinstance(notifications, ObjectList):
            self._notifications = notifications
        else:
            raise ValueError("Notifications must be a list")

    @property
    def state_thresholds(self):
        return self._state_thresholds

    @state_thresholds.setter
    def state_thresholds(self, state_thresholds: dict[str, str]):
        """
        Set the dictionary of k8s resource states to thresholds time strings.
        The state will be matched against the pod's status. The threshold should be a time string that conforms
        to timelength python package standards and is at least 1 minute (-1 for infinite). If the phase is active
        for longer than the threshold, the run will be marked as aborted and the pod will be deleted.
        See mlconf.function.spec.state_thresholds for the state options and default values.

        example:
            {"image_pull_backoff": "1h", "executing": "1d 2 hours"}

        :param state_thresholds: The state-thresholds dictionary.
        """
        self._verify_dict(state_thresholds, "state_thresholds")
        self._state_thresholds = state_thresholds

    def extract_type_hints_from_inputs(self):
        """
        This method extracts the type hints from the input keys in the input dictionary.

        As a result, after the method ran the inputs dictionary - a dictionary of parameter names as keys and paths as
        values, will be cleared from type hints and the extracted type hints will be saved in the spec's inputs type
        hints dictionary - a dictionary of parameter names as keys and their type hints as values. If a parameter is
        not in the type hints dictionary, its type hint will be `mlrun.DataItem` by default.
        """
        # Validate there are inputs to read:
        if self.inputs is None:
            return

        # Prepare dictionaries to hold the cleared inputs and type hints:
        cleared_inputs = {}
        extracted_inputs_type_hints = {}

        # Clear the inputs from parsing configurations:
        for input_key, input_value in self.inputs.items():
            # Look for type hinted in input key:
            if ":" in input_key:
                # Separate the user input by colon:
                input_key, input_type = RunSpec._separate_type_hint_from_input_key(
                    input_key=input_key
                )
                # Collect the type hint:
                extracted_inputs_type_hints[input_key] = input_type
            # Collect the cleared input key:
            cleared_inputs[input_key] = input_value

        # Set the now configuration free inputs and extracted type hints:
        self.inputs = cleared_inputs
        self.inputs_type_hints = extracted_inputs_type_hints

    @staticmethod
    def join_outputs_and_returns(
        outputs: list[str], returns: list[Union[str, dict[str, str]]]
    ) -> list[str]:
        """
        Get the outputs set in the spec. The outputs are constructed from both the 'outputs' and 'returns' properties
        that were set by the user.

        :param outputs: A spec outputs property - list of output keys.
        :param returns: A spec returns property - list of key and configuration of how to log returning values.

        :return: The joined 'outputs' and 'returns' list.
        """
        # Collect the 'returns' property keys:
        cleared_returns = []
        if returns:
            for return_value in returns:
                # Check if the return entry is a configuration dictionary or a key-type structure string (otherwise its
                # just a key string):
                if isinstance(return_value, dict):
                    # Set it to the artifact key:
                    return_value = return_value["key"]
                elif ":" in return_value:
                    # Take only the key name (returns values pattern is validated when set in the spec):
                    return_value = return_value.replace(" ", "").split(":")[0]
                # Collect it:
                cleared_returns.append(return_value)

        # Use `set` join to combine the two lists without duplicates:
        outputs = list(set(outputs if outputs else []) | set(cleared_returns))

        return outputs

    @staticmethod
    def _separate_type_hint_from_input_key(input_key: str) -> tuple[str, str]:
        """
        An input key in the `inputs` dictionary parameter of a task (or `Runtime.run` method) or the docs setting of a
        `Runtime` handler can be provided with a colon to specify its type hint in the following structure:
        "<parameter_key> : <type_hint>".

        This method parses the provided value by the user.

        :param input_key: A string entry in the inputs dictionary keys.

        :return: The value as key and type hint tuple.

        :raise MLRunInvalidArgumentError: If an incorrect pattern was provided.
        """
        # Validate correct pattern:
        if input_key.count(":") > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Incorrect input pattern. Input keys can have only a single ':' in them to specify the desired type "
                f"the input will be parsed as. Given: {input_key}."
            )

        # Split into key and type:
        value_key, value_type = input_key.replace(" ", "").split(":")

        return value_key, value_type


class RunStatus(ModelObj):
    """Run status"""

    _default_fields_to_strip = ModelObj._default_fields_to_strip + ["artifacts"]

    def __init__(
        self,
        state=None,
        error=None,
        host=None,
        commit=None,
        status_text=None,
        results=None,
        artifacts=None,
        start_time=None,
        end_time=None,
        last_update=None,
        iterations=None,
        ui_url=None,
        reason: Optional[str] = None,
        notifications: Optional[dict[str, Notification]] = None,
        artifact_uris: Optional[dict[str, str]] = None,
    ):
        self.state = state or "created"
        self.status_text = status_text
        self.error = error
        self.host = host
        self.commit = commit
        self.results = results
        self._artifacts = artifacts
        self.start_time = start_time
        self.end_time = end_time
        self.last_update = last_update
        self.iterations = iterations
        self.ui_url = ui_url
        self.reason = reason
        self.notifications = notifications or {}
        # Artifact key -> URI mapping, since the full artifacts are not stored in the runs DB table
        self._artifact_uris = artifact_uris or {}

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: Optional[dict] = None
    ):
        deprecated_fields = {
            # Set artifacts as deprecated for lazy loading
            "artifacts": "artifact_uris"
        }
        return super().from_dict(
            struct, fields=fields, deprecated_fields=deprecated_fields
        )

    @property
    def artifacts(self):
        """
        Artifacts are lazy loaded to reduce memory consumption.
        We keep artifact_uris (key -> store URI dictionary) to be able to get the run artifacts easily.
        If the artifact is not already in the cache, we get it from the store (DB).
        :return: List of artifact dictionaries
        """
        self._artifacts = self._artifacts or []
        existing_artifact_keys = {
            artifact["metadata"]["key"] for artifact in self._artifacts
        }
        for key, uri in self.artifact_uris.items():
            if key not in existing_artifact_keys:
                artifact = mlrun.datastore.get_store_resource(uri)
                self._artifacts.append(artifact.to_dict())
        return self._artifacts

    @artifacts.setter
    def artifacts(self, artifacts):
        self._artifacts = artifacts

    @property
    def artifact_uris(self):
        return self._artifact_uris

    @artifact_uris.setter
    def artifact_uris(self, artifact_uris):
        resolved_artifact_uris = {}
        if isinstance(artifact_uris, list):
            # artifact_uris is the deprecated list of artifacts - convert to new form
            for artifact in artifact_uris:
                if isinstance(artifact, dict):
                    artifact = mlrun.artifacts.dict_to_artifact(artifact)
                resolved_artifact_uris[artifact.key] = artifact.uri
        else:
            resolved_artifact_uris = artifact_uris

        self._artifact_uris = resolved_artifact_uris

    def is_failed(self) -> Optional[bool]:
        """
        This method returns whether a run has failed.
        Returns none if state has yet to be defined. callee is responsible for handling None.
        (e.g wait for state to be defined)
        """
        if not self.state:
            return None
        return self.state.casefold() in [
            mlrun.run.RunStatuses.failed.casefold(),
            mlrun.run.RunStatuses.error.casefold(),
        ]


class RunTemplate(ModelObj):
    """Run template"""

    def __init__(self, spec: RunSpec = None, metadata: RunMetadata = None):
        self._spec = None
        self._metadata = None
        self.spec = spec
        self.metadata = metadata

    @property
    def spec(self) -> RunSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", RunSpec)

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", RunMetadata)

    def with_params(self, **kwargs):
        """set task parameters using key=value, key2=value2, .."""
        self.spec.parameters = kwargs
        return self

    def with_input(self, key, path):
        """set task data input, path is an Mlrun global DataItem uri

        examples::

            task.with_input("data", "/file-dir/path/to/file")
            task.with_input("data", "s3://<bucket>/path/to/file")
            task.with_input("data", "v3io://<data-container>/path/to/file")
        """
        if not self.spec.inputs:
            self.spec.inputs = {}
        self.spec.inputs[key] = path
        return self

    def with_hyper_params(
        self,
        hyperparams,
        selector=None,
        strategy: HyperParamStrategies = None,
        **options,
    ):
        """set hyper param values and configurations,
        see parameters in: :py:class:`HyperParamOptions`

        example::

            grid_params = {"p1": [2, 4, 1], "p2": [10, 20]}
            task = mlrun.new_task("grid-search")
            task.with_hyper_params(grid_params, selector="max.accuracy")
        """
        self.spec.hyperparams = hyperparams
        self.spec.hyper_param_options = options
        self.spec.hyper_param_options.selector = selector
        self.spec.hyper_param_options.strategy = strategy
        self.spec.hyper_param_options.validate()
        return self

    def with_param_file(
        self,
        param_file,
        selector=None,
        strategy: HyperParamStrategies = None,
        **options,
    ):
        """set hyper param values (from a file url) and configurations,
        see parameters in: :py:class:`HyperParamOptions`

        example::

            grid_params = "s3://<my-bucket>/path/to/params.json"
            task = mlrun.new_task("grid-search")
            task.with_param_file(grid_params, selector="max.accuracy")
        """
        self.spec.hyper_param_options = options
        self.spec.hyper_param_options.param_file = param_file
        self.spec.hyper_param_options.selector = selector
        self.spec.hyper_param_options.strategy = strategy
        self.spec.hyper_param_options.validate()
        return self

    def with_secrets(self, kind, source):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in workflows, example::

            task.with_secrets('file', 'file.txt')
            task.with_secrets('inline', {'key': 'val'})
            task.with_secrets('env', 'ENV1,ENV2')

            task.with_secrets('vault', ['secret1', 'secret2'...])

            # If using with k8s secrets, the k8s secret is managed by MLRun, through the project-secrets
            # mechanism. The secrets will be attached to the running pod as environment variables.
            task.with_secrets('kubernetes', ['secret1', 'secret2'])

            # If using an empty secrets list [] then all accessible secrets will be available.
            task.with_secrets('vault', [])

            # To use with Azure key vault, a k8s secret must be created with the following keys:
            # kubectl -n <namespace> create secret generic azure-key-vault-secret \\
            #     --from-literal=tenant_id=<service principal tenant ID> \\
            #     --from-literal=client_id=<service principal client ID> \\
            #     --from-literal=secret=<service principal secret key>

            task.with_secrets('azure_vault', {
                'name': 'my-vault-name',
                'k8s_secret': 'azure-key-vault-secret',
                # An empty secrets list may be passed ('secrets': []) to access all vault secrets.
                'secrets': ['secret1', 'secret2'...]
            })

        :param kind:   secret type (file, inline, env)
        :param source: secret data or link (see example)

        :returns: The RunTemplate object
        """

        if kind == "vault" and isinstance(source, list):
            source = {"project": self.metadata.project, "secrets": source}

        self.spec.secret_sources.append({"kind": kind, "source": source})
        return self

    def set_label(self, key, value):
        """set a key/value label for the task"""
        self.metadata.labels[key] = str(value)
        return self

    def to_env(self):
        environ["MLRUN_EXEC_CONFIG"] = self.to_json()


class RunObject(RunTemplate):
    """A run"""

    def __init__(
        self,
        spec: RunSpec = None,
        metadata: RunMetadata = None,
        status: RunStatus = None,
    ):
        super().__init__(spec, metadata)
        self._status = None
        self.status = status
        self.outputs_wait_for_completion = True

    @classmethod
    def from_template(cls, template: RunTemplate):
        return cls(template.spec, template.metadata)

    def to_json(self, exclude=None, **kwargs):
        # Since the `params` attribute within each notification object can be large,
        # it has the potential to cause errors and is unnecessary for the notification functionality.
        # Therefore, in this section, we remove the `params` attribute from each notification object.
        if (
            exclude_notifications_params := kwargs.get("exclude_notifications_params")
        ) and exclude_notifications_params:
            if self.spec.notifications:
                # Extract and remove 'params' from each notification
                extracted_params = []
                for notification in self.spec.notifications:
                    extracted_params.append(notification.params)
                    del notification.params
                # Generate the JSON representation, excluding specified fields
                json_obj = super().to_json(exclude=exclude)
                # Restore 'params' back to the notifications
                for notification, params in zip(
                    self.spec.notifications, extracted_params
                ):
                    notification.params = params
                return json_obj
        return super().to_json(exclude=exclude)

    @property
    def status(self) -> RunStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", RunStatus)

    @property
    def error(self) -> str:
        """error string if failed"""
        if (
            self.status
            and self.status.state
            in mlrun.common.runtimes.constants.RunStates.error_and_abortion_states()
        ):
            unknown_error = ""
            if (
                self.status.state
                in mlrun.common.runtimes.constants.RunStates.abortion_states()
            ):
                unknown_error = "Run was aborted"

            elif (
                self.status.state
                in mlrun.common.runtimes.constants.RunStates.error_states()
            ):
                unknown_error = "Unknown error"

            return (
                self.status.error
                or self.status.status_text
                or self.status.reason
                or unknown_error
            )
        return ""

    def output(self, key: str):
        """
        Return the value of a specific result or artifact by key.

        This method waits for the outputs to complete and retrieves the value corresponding to the provided key.
        If the key exists in the results, it returns the corresponding result value.
        If not found in results, it attempts to fetch the artifact by key (cached in the run status).
        If the artifact is not found, it tries to fetch the artifact URI by key.
        If no artifact or result is found for the key, returns None.

        :param key: The key of the result or artifact to retrieve.
        :return: The value of the result or the artifact URI corresponding to the key, or None if not found.
        """
        self._outputs_wait_for_completion()

        # Check if the key exists in results and return the result value
        if self.status.results and key in self.status.results:
            return self.status.results[key]

        # Artifacts are usually cached in the run object under `status.artifacts`. However, the artifacts are not
        # stored in the DB as part of the run. The server may enrich the run with the artifacts or provide
        # `status.artifact_uris` instead. See mlrun.common.formatters.run.RunFormat.
        # When running locally - `status.artifact_uri` does not exist in the run.
        # When listing runs - `status.artifacts` does not exist in the run.
        artifact = self._artifact(key)
        if artifact:
            return get_artifact_target(artifact, self.metadata.project)

        if self.status.artifact_uris and key in self.status.artifact_uris:
            return self.status.artifact_uris[key]

        return None

    @property
    def ui_url(self) -> str:
        """UI URL (for relevant runtimes)"""
        self.refresh()
        if not self._status.ui_url:
            print(f"UI currently not available (status={self._status.state})")
        return self._status.ui_url

    @property
    def outputs(self):
        """
        Return a dictionary of outputs, including result values and artifact URIs.

        This method waits for the outputs to complete and combines result values
        and artifact URIs into a single dictionary. If there are multiple artifacts
        for the same key, only include the artifact that does not have the "latest" tag.
        If there is no other tag, include the "latest" tag as a fallback.

        :return: Dictionary containing result values and artifact URIs.
        """
        self._outputs_wait_for_completion()
        outputs = {}

        # Add results if available
        if self.status.results:
            outputs.update(self.status.results)

        # Artifacts are usually cached in the run object under `status.artifacts`. However, the artifacts are not
        # stored in the DB as part of the run. The server may enrich the run with the artifacts or provide
        # `status.artifact_uris` instead. See mlrun.common.formatters.run.RunFormat.
        # When running locally - `status.artifact_uri` does not exist in the run.
        # When listing runs - `status.artifacts` does not exist in the run.
        if self.status.artifacts:
            outputs.update(self._process_artifacts(self.status.artifacts))
        elif self.status.artifact_uris:
            outputs.update(self.status.artifact_uris)

        return outputs

    def artifact(self, key: str) -> typing.Optional["mlrun.DataItem"]:
        """Return artifact DataItem by key.

        This method waits for the outputs to complete, searches for the artifact matching the given key,
        and returns a DataItem if the artifact is found.

        :param key: The key of the artifact to find.
        :return: A DataItem corresponding to the artifact with the given key, or None if no such artifact is found.
        """
        self._outputs_wait_for_completion()
        artifact = self._artifact(key)
        if not artifact:
            return None
        uri = get_artifact_target(artifact, self.metadata.project)
        return mlrun.get_dataitem(uri) if uri else None

    def _outputs_wait_for_completion(
        self,
        show_logs=False,
    ):
        """
        Wait for the run to complete fetching the run outputs.
        When running a function with watch=False, and passing the outputs to another function,
        the outputs will not be available until the run is completed.
        :param show_logs: default False, avoid spamming unwanted logs of the run when the user asks for outputs
        """
        if self.outputs_wait_for_completion:
            self.wait_for_completion(
                show_logs=show_logs,
            )

    def _artifact(self, key):
        """
        Return the last artifact DataItem that matches the given key.

        If multiple artifacts with the same key exist, return the last one in the list.
        If there are artifacts with different tags, the method will return the one with a tag other than 'latest'
        if available.
        If no artifact with the given key is found, return None.

        :param key: The key of the artifact to retrieve.
        :return: The last artifact DataItem with the given key, or None if no such artifact is found.
        """
        if not self.status.artifacts and not self.status.artifact_uris:
            return None

        # Collect artifacts that match the key
        matching_artifacts = [
            artifact
            for artifact in self.status.artifacts
            if artifact["metadata"].get("key") == key
        ]

        if not matching_artifacts:
            if key not in self.status.artifact_uris:
                return None

            # Get artifact by store URI sanity (should have been enriched by now in status.artifacts property)
            artifact_uri = self.status.artifact_uris[key]
            return mlrun.datastore.get_store_resource(artifact_uri)

        # Sort matching artifacts by creation date in ascending order.
        # The last element in the list will be the one created most recently.
        # In case the `created` field does not exist in the artifact, that artifact will appear first in the sorted list
        matching_artifacts.sort(
            key=lambda artifact: artifact["metadata"].get("created", datetime.min)
        )

        # Filter out artifacts with 'latest' tag
        non_latest_artifacts = [
            artifact
            for artifact in matching_artifacts
            if artifact["metadata"].get("tag") != "latest"
        ]

        # Return the last non-'latest' artifact if available, otherwise return the last artifact
        # In the case of only one tag, `status.artifacts` includes [v1, latest]. In that case, we want to return v1.
        # In the case of multiple tags, `status.artifacts` includes [v1, latest, v2, v3].
        # In that case, we need to return the last one (v3).
        return (non_latest_artifacts or matching_artifacts)[-1]

    def _process_artifacts(self, artifacts):
        artifacts_by_key = {}

        # Organize artifacts by key
        for artifact in artifacts:
            key = artifact["metadata"]["key"]
            if key not in artifacts_by_key:
                artifacts_by_key[key] = []
            artifacts_by_key[key].append(artifact)

        outputs = {}
        for key, artifacts in artifacts_by_key.items():
            # Sort matching artifacts by creation date in ascending order.
            # The last element in the list will be the one created most recently.
            # In case the `created` field does not exist in the artifactthat artifact will appear
            # first in the sorted list
            artifacts.sort(
                key=lambda artifact: artifact["metadata"].get("created", datetime.min)
            )

            # Filter out artifacts with 'latest' tag
            non_latest_artifacts = [
                artifact
                for artifact in artifacts
                if artifact["metadata"].get("tag") != "latest"
            ]

            # Save the last non-'latest' artifact if available, otherwise save the last artifact
            # In the case of only one tag, `artifacts` includes [v1, latest], in that case, we want to save v1.
            # In the case of multiple tags, `artifacts` includes [v1, latest, v2, v3].
            # In that case, we need to save the last one (v3).
            artifact_to_save = (non_latest_artifacts or artifacts)[-1]
            outputs[key] = get_artifact_target(artifact_to_save, self.metadata.project)

        return outputs

    def uid(self):
        """run unique id"""
        return self.metadata.uid

    def state(self):
        """current run state"""
        if (
            self.status.state
            in mlrun.common.runtimes.constants.RunStates.terminal_states()
        ):
            return self.status.state
        self.refresh()
        return self.status.state or "unknown"

    def refresh(self):
        """refresh run state from the db"""
        db = mlrun.get_run_db()
        run = db.read_run(
            uid=self.metadata.uid,
            project=self.metadata.project,
            iter=self.metadata.iteration,
        )
        if run:
            run_status = run.get("status", {})
            # Artifacts are not stored in the DB, so we need to preserve them here
            run_status["artifacts"] = self.status.artifacts
            self.status = RunStatus.from_dict(run_status)
            return self

    def show(self):
        """show the current status widget, in jupyter notebook"""
        db = mlrun.get_run_db()
        db.list_runs(uid=self.metadata.uid, project=self.metadata.project).show()

    def logs(self, watch=True, db=None, offset=0):
        """return or watch on the run logs"""
        if not db:
            db = mlrun.get_run_db()

        if not db:
            logger.warning("DB is not configured, cannot show logs")
            return None

        state, new_offset = db.watch_log(
            self.metadata.uid, self.metadata.project, watch=watch, offset=offset
        )
        if state:
            logger.debug("Run reached terminal state", state=state)

        return state, new_offset

    def wait_for_completion(
        self,
        sleep=3,
        timeout=0,
        raise_on_failure=True,
        show_logs=None,
        logs_interval=None,
    ):
        """
        Wait for remote run to complete.
        Default behavior is to wait until reached terminal state or timeout passed, if timeout is 0 then wait forever
        It pulls the run status from the db every sleep seconds.
        If show_logs is not False and logs_interval is not None, it will print the logs when run reached terminal state
        If show_logs is not False and logs_interval is defined, it will print the logs every logs_interval seconds
        if show_logs is False it will not print the logs, will still pull the run state until it reaches terminal state
        """
        # TODO: rename sleep to pull_state_interval
        total_time = 0
        offset = 0
        last_pull_log_time = None
        logs_enabled = show_logs is not False
        state = self.state()
        if state not in mlrun.common.runtimes.constants.RunStates.terminal_states():
            logger.info(
                f"run {self.metadata.name} is not completed yet, waiting for it to complete",
                current_state=state,
            )
        while True:
            state = self.state()
            if (
                logs_enabled
                and logs_interval
                and state
                not in mlrun.common.runtimes.constants.RunStates.terminal_states()
                and (
                    last_pull_log_time is None
                    or (datetime.now() - last_pull_log_time).seconds > logs_interval
                )
            ):
                last_pull_log_time = datetime.now()
                state, offset = self.logs(watch=False, offset=offset)

            if state in mlrun.common.runtimes.constants.RunStates.terminal_states():
                if logs_enabled and logs_interval:
                    self.logs(watch=False, offset=offset)
                break
            time.sleep(sleep)
            total_time += sleep
            if timeout and total_time > timeout:
                raise mlrun.errors.MLRunTimeoutError(
                    "Run did not reach terminal state on time"
                )
        if logs_enabled and not logs_interval:
            self.logs(watch=False)
        if (
            raise_on_failure
            and state != mlrun.common.runtimes.constants.RunStates.completed
        ):
            raise mlrun.errors.MLRunRuntimeError(
                f"Task {self.metadata.name} did not complete (state={state})"
            )

        return state

    def abort(self):
        """abort the run"""
        db = mlrun.get_run_db()
        db.abort_run(self.metadata.uid, self.metadata.project)

    @staticmethod
    def create_uri(project: str, uid: str, iteration: Union[int, str], tag: str = ""):
        if tag:
            tag = f":{tag}"
        iteration = str(iteration)
        return f"{project}@{uid}#{iteration}{tag}"

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str, str, str]:
        """Parse the run's uri

        :param uri: run uri in the format of <project>@<uid>#<iteration>[:tag]
        :return: project, uid, iteration, tag
        """
        uri_pattern = mlrun.utils.regex.run_uri_pattern
        match = re.match(uri_pattern, uri)
        if not match:
            raise ValueError(
                "Uri not in supported format <project>@<uid>#<iteration>[:tag]"
            )
        group_dict = match.groupdict()
        return (
            group_dict["project"],
            group_dict["uid"],
            group_dict["iteration"],
            group_dict["tag"],
        )


class EntrypointParam(ModelObj):
    def __init__(
        self,
        name="",
        type=None,
        default=None,
        doc="",
        required=None,
        choices: Optional[list] = None,
    ):
        self.name = name
        self.type = type
        self.default = default
        self.doc = doc
        self.required = required
        self.choices = choices


class FunctionEntrypoint(ModelObj):
    def __init__(
        self,
        name="",
        doc="",
        parameters=None,
        outputs=None,
        lineno=-1,
        has_varargs=None,
        has_kwargs=None,
    ):
        self.name = name
        self.doc = doc
        self.parameters = [] if parameters is None else parameters
        self.outputs = [] if outputs is None else outputs
        self.lineno = lineno
        self.has_varargs = has_varargs
        self.has_kwargs = has_kwargs


def new_task(
    name=None,
    project=None,
    handler=None,
    params=None,
    hyper_params=None,
    param_file=None,
    selector=None,
    hyper_param_options=None,
    inputs=None,
    outputs=None,
    in_path=None,
    out_path=None,
    artifact_path=None,
    secrets=None,
    base=None,
    returns=None,
) -> RunTemplate:
    """Creates a new task

    :param name:            task name
    :param project:         task project
    :param handler:         code entry-point/handler name
    :param params:          input parameters (dict)
    :param hyper_params:    dictionary of hyper parameters and list values, each
                            hyper param holds a list of values, the run will be
                            executed for every parameter combination (GridSearch)
    :param param_file:      a csv file with parameter combinations, first row hold
                            the parameter names, following rows hold param values
    :param selector:        selection criteria for hyper params e.g. "max.accuracy"
    :param hyper_param_options:   hyper parameter options, see: :py:class:`HyperParamOptions`
    :param inputs:          dictionary of input objects + optional paths (if path is
                            omitted the path will be the in_path/key)
    :param outputs:         dictionary of input objects + optional paths (if path is
                            omitted the path will be the out_path/key)
    :param in_path:         default input path/url (prefix) for inputs
    :param out_path:        default output path/url (prefix) for artifacts
    :param artifact_path:   default artifact output path
    :param secrets:         extra secrets specs, will be injected into the runtime
                            e.g. ['file=<filename>', 'env=ENV_KEY1,ENV_KEY2']
    :param base:            task instance to use as a base instead of a fresh new task instance
    :param returns:         List of log hints - configurations for how to log the returning values from the handler's
                            run (as artifacts or results). The list's length must be equal to the amount of returning
                            objects. A log hint may be given as:

                            * A string of the key to use to log the returning value as result or as an artifact. To
                              specify The artifact type, it is possible to pass a string in the following structure:
                              "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no
                              artifact type is specified, the object's default artifact type will be used.
                            * A dictionary of configurations to use when logging. Further info per object type and
                              artifact type can be given there. The artifact key must appear in the dictionary as
                              "key": "the_key".
    """

    if base:
        run = deepcopy(base)
    else:
        run = RunTemplate()
    run.metadata.name = name or run.metadata.name
    run.metadata.project = project or run.metadata.project
    run.spec.handler = handler or run.spec.handler
    run.spec.parameters = params or run.spec.parameters
    run.spec.inputs = inputs or run.spec.inputs
    run.spec.returns = returns or run.spec.returns
    run.spec.outputs = outputs or run.spec.outputs or []
    run.spec.input_path = in_path or run.spec.input_path
    run.spec.output_path = artifact_path or out_path or run.spec.output_path
    run.spec.secret_sources = secrets or run.spec.secret_sources or []

    run.spec.hyperparams = hyper_params or run.spec.hyperparams
    run.spec.hyper_param_options = hyper_param_options or run.spec.hyper_param_options
    run.spec.hyper_param_options.param_file = (
        param_file or run.spec.hyper_param_options.param_file
    )
    run.spec.hyper_param_options.selector = (
        selector or run.spec.hyper_param_options.selector
    )
    return run


class TargetPathObject:
    """Class configuring the target path
    This class will take consideration of a few parameters to create the correct end result path:

    - :run_id: if run_id is provided target will be considered as run_id mode which require to
        contain a {run_id} place holder in the path.
    - :is_single_file: if true then run_id must be the directory containing the output file
        or generated before the file name (run_id/output.file).
    - :base_path: if contains the place holder for run_id, run_id must not be None.
        if run_id passed and place holder doesn't exist the place holder will
        be generated in the correct place.
    """

    def __init__(
        self,
        base_path=None,
        run_id=None,
        is_single_file=False,
    ):
        self.run_id = run_id
        self.full_path_template = base_path
        if run_id is not None:
            if RUN_ID_PLACE_HOLDER not in self.full_path_template:
                if not is_single_file:
                    if self.full_path_template[-1] != "/":
                        self.full_path_template = self.full_path_template + "/"
                    self.full_path_template = (
                        self.full_path_template + RUN_ID_PLACE_HOLDER + "/"
                    )
                else:
                    dir_name_end = len(self.full_path_template)
                    if self.full_path_template[-1] != "/":
                        dir_name_end = self.full_path_template.rfind("/") + 1
                    updated_path = (
                        self.full_path_template[:dir_name_end]
                        + RUN_ID_PLACE_HOLDER
                        + "/"
                        + self.full_path_template[dir_name_end:]
                    )
                    self.full_path_template = updated_path
            else:
                if self.full_path_template[-1] != "/":
                    if self.full_path_template.endswith(RUN_ID_PLACE_HOLDER):
                        self.full_path_template = self.full_path_template + "/"
        else:
            if RUN_ID_PLACE_HOLDER in self.full_path_template:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Error when trying to create TargetPathObject with place holder '{run_id}' but no value."
                )

    def get_templated_path(self):
        return self.full_path_template

    def get_absolute_path(self, project_name=None):
        path = template_artifact_path(
            artifact_path=self.full_path_template,
            project=project_name,
        )
        return path.format(run_id=self.run_id) if self.run_id else path


class DataSource(ModelObj):
    """online or offline data source spec"""

    _dict_fields = [
        "kind",
        "name",
        "path",
        "attributes",
        "key_field",
        "time_field",
        "schedule",
        "online",
        "workers",
        "max_age",
        "start_time",
        "end_time",
        "credentials_prefix",
    ]
    kind = None

    _fields_to_serialize = ["start_time", "end_time"]

    def __init__(
        self,
        name: Optional[str] = None,
        path: Optional[str] = None,
        attributes: Optional[dict[str, object]] = None,
        key_field: Optional[str] = None,
        time_field: Optional[str] = None,
        schedule: Optional[str] = None,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None,
    ):
        self.name = name
        self.path = str(path) if path is not None else None
        self.attributes = attributes or {}
        self.schedule = schedule
        self.key_field = key_field
        self.time_field = time_field
        self.start_time = start_time
        self.end_time = end_time

        self.online = None
        self.max_age = None
        self.workers = None
        self._secrets = {}

    def set_secrets(self, secrets):
        self._secrets = secrets

    def _serialize_field(
        self, struct: dict, field_name: Optional[str] = None, strip: bool = False
    ) -> typing.Any:
        value = super()._serialize_field(struct, field_name, strip)
        # We pull the field from self and not from struct because it was excluded from the struct when looping over
        # the fields to save.
        if field_name in ("start_time", "end_time") and isinstance(value, datetime):
            return value.isoformat()
        return value


class DataTargetBase(ModelObj):
    """data target spec, specify a destination for the feature set data"""

    _dict_fields = [
        "name",
        "kind",
        "path",
        "after_step",
        "attributes",
        "partitioned",
        "key_bucketing_number",
        "partition_cols",
        "time_partitioning_granularity",
        "max_events",
        "flush_after_seconds",
        "storage_options",
        "run_id",
        "schema",
        "credentials_prefix",
    ]

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: Optional[dict] = None
    ):
        return super().from_dict(struct, fields=fields)

    def get_path(self):
        # polymorphism won't work here, because from_dict always returns an instance of the base type (DataTargetBase)
        if self.kind in ["stream", "kafka"]:
            return TargetPathObject(self.path or "")

        if self.path:
            is_single_file = hasattr(self, "is_single_file") and self.is_single_file()
            return TargetPathObject(self.path, self.run_id, is_single_file)
        else:
            return None

    def __init__(
        self,
        kind: Optional[str] = None,
        name: str = "",
        path=None,
        attributes: Optional[dict[str, str]] = None,
        after_step=None,
        partitioned: bool = False,
        key_bucketing_number: Optional[int] = None,
        partition_cols: Optional[list[str]] = None,
        time_partitioning_granularity: Optional[str] = None,
        max_events: Optional[int] = None,
        flush_after_seconds: Optional[int] = None,
        storage_options: Optional[dict[str, str]] = None,
        schema: Optional[dict[str, Any]] = None,
        credentials_prefix=None,
    ):
        self.name = name
        self.kind: str = kind
        self.path = path
        self.after_step = after_step
        self.attributes = attributes or {}
        self.last_written = None
        self.partitioned = partitioned
        self.key_bucketing_number = key_bucketing_number
        self.partition_cols = partition_cols
        self.time_partitioning_granularity = time_partitioning_granularity
        self.max_events = max_events
        self.flush_after_seconds = flush_after_seconds
        self.storage_options = storage_options
        self.run_id = None
        self.schema = schema
        self.credentials_prefix = credentials_prefix


class FeatureSetProducer(ModelObj):
    """information about the task/job which produced the feature set data"""

    def __init__(self, kind=None, name=None, uri=None, owner=None, sources=None):
        self.kind = kind
        self.name = name
        self.owner = owner
        self.uri = uri
        self.sources = sources or {}


class DataTarget(DataTargetBase):
    """data target with extra status information (used in the feature-set/vector status)"""

    _dict_fields = [
        "name",
        "kind",
        "path",
        "attributes",
        "start_time",
        "online",
        "status",
        "updated",
        "size",
        "last_written",
        "run_id",
        "partitioned",
        "key_bucketing_number",
        "partition_cols",
        "time_partitioning_granularity",
        "credentials_prefix",
    ]

    def __init__(
        self,
        kind: Optional[str] = None,
        name: str = "",
        path=None,
        online=None,
    ):
        super().__init__(kind, name, path)
        self.status = ""
        self.updated = None
        self.size = None
        self.online = online
        self.max_age = None
        self.start_time = None
        self.last_written = None
        self._producer = None
        self.producer = {}
        self.attributes = {}

    @property
    def producer(self) -> FeatureSetProducer:
        return self._producer

    @producer.setter
    def producer(self, producer):
        self._producer = self._verify_dict(producer, "producer", FeatureSetProducer)


class VersionedObjMetadata(ModelObj):
    def __init__(
        self,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        uid: Optional[str] = None,
        project: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        annotations: Optional[dict[str, str]] = None,
        updated=None,
    ):
        self.name = name
        self.tag = tag
        self.uid = uid
        self.project = project
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.updated = updated
