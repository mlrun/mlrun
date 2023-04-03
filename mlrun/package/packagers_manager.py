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
#
import importlib
import inspect
import os
import shutil
from typing import Any, Dict, List, Tuple, Type, Union

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem, is_store_uri, parse_store_uri, store_manager
from mlrun.errors import (
    MLRunInvalidArgumentError,
    MLRunPackagePackagerCollectionError,
    MLRunPackagePackingError,
    MLRunPackageUnpackingError,
)
from mlrun.package.packager import Packager
from mlrun.package.packagers.default_packager import DefaultPackager
from mlrun.utils import StorePrefix, logger

from .common import LogHintKey, TypingUtils


class PackagersManager:
    """
    A packager manager is holding the project's packagers and sending them objects to pack and data items to unpack.

    It prepares the instructions / log hint configurations and then looks for the first packager who fits the task.
    That's why when the manager collects its packagers, it first collects builtin MLRun packagers and only then the
    user's custom packagers, this way user's custom packagers will have higher priority.
    """

    # Mandatory packagers to be collected at initialization time:
    _MLRUN_REQUIREMENTS_PACKAGERS = [
        "python_standard_library",
        "pandas",
        "numpy",
        "mlrun",
    ]
    # Optional packagers to be collected at initialization time:
    _EXTENDED_PACKAGERS = ["matplotlib", "plotly", "bokeh"]

    def __init__(self, default_packager: Packager = None):
        """
        Initialize a packagers manager.

        :param default_packager: The default packager should be a packager that fits to all types. It will be the first
                                 packager in the manager's packagers (meaning it will be used at lowest priority) and it
                                 should be found fitting when all packagers managed by the manager do not fit an
                                 object or data item. Default to ``mlrun.DefaultPackager``.
        """
        # Set the default packager:
        self._default_packager = default_packager or DefaultPackager

        # Initialize the packagers list (with the default packager in it):
        self._packagers: List[Packager] = [self._default_packager]

        # Set an artifacts list and results dictionary to collect all packed objects (will be used later to write extra
        # data if noted by the user using the log hint key "extra_data")
        self._artifacts: List[Artifact] = []
        self._results = {}

        # Collect the builtin standard packagers:
        self._collect_packagers()

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Get the artifacts that were packed by the manager.

        :return: A list of artifacts.
        """
        return self._artifacts

    @property
    def results(self) -> dict:
        """
        Get the results that were packed by the manager.

        :return: A results dictionary.
        """
        return self._results

    def collect_packagers(self, packagers: List[Union[Type, str]]):
        """
        Collect the provided packagers. Packagers passed as module paths will be imported and validated to be of type
        `Packager`. If needed to import all packagers from a module, use the module path with a "*" at the end.

        Notice: Only packagers that are declared in the module will be collected (packagers imported in the module scope
        won't be collected). For example::

            from mlrun import Packager
            from x import XPackager

            class YPackager(Packager):
                pass

        Only "YPackager" will be collected as it is declared in the module, but not "XPackager" which is only imported.

        :param packagers: List of packagers to add.

        :raise MLRunInvalidArgumentError:   In case one of the classes provided was not of type `Packager`.
        :raise MLRunPackageCollectingError: In case the packager could not be collected.
        """
        for packager in packagers:
            # If it's a string, it's the module path of the class, so we import it:
            if isinstance(packager, str):
                # Import the module:
                module_name, class_name = self._split_module_path(module_path=packager)
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError as module_not_found_error:
                    raise MLRunPackagePackagerCollectionError(
                        f"The packager '{class_name}' could not be collected from the module '{module_name}' as it "
                        f"cannot be imported."
                    ) from module_not_found_error
                # Check if needed to import all packagers from the given module:
                if class_name == "*":
                    # Get all the packagers from the module and collect them (this time they will be sent as `Packager`
                    # types to the method):
                    self.collect_packagers(
                        packagers=[
                            member[1]
                            for member in inspect.getmembers(
                                module,
                                lambda member: (
                                    hasattr(member, "__module__")
                                    and member.__module__ == module.__name__
                                    and isinstance(member, Packager)
                                ),
                            )
                        ]
                    )
                    # Collected from the previous call, continue to the next packager in the list:
                    continue
                # Import the packager and continue like as if it was given as a type:
                try:
                    packager = getattr(module, class_name)
                except AttributeError as attribute_error:
                    raise MLRunPackagePackagerCollectionError(
                        f"The packager '{class_name}' could not be collected as it does not exist in the module "
                        f"'{module.__name__}'."
                    ) from attribute_error
            # Validate the class given is a `Packager` type:
            if not isinstance(packager, Packager):
                raise MLRunInvalidArgumentError(
                    f"The packager '{packager.__name__}' could not be collected as it is not a `mlrun.Packager`."
                )
            # Collect the packager (putting him first in the list for highest priority:
            self._packagers.insert(0, packager)
            # For debugging, we'll print the collected packager:
            logger.debug(
                f"The packagers manager collected the packager: {str(packager)}"
            )

    def pack(
        self, obj: Any, log_hint: Dict[str, str]
    ) -> Union[Artifact, dict, None, List[Union[Artifact, dict, None]]]:
        """
        Pack an object using one of the manager's packagers. A `dict` ("**") or `list` ("*") unpacking syntax in the
        log hint key will pack the objects within them in separate packages.

        :param obj:      The object to pack as an artifact.
        :param log_hint: The log hint to use.

        :return: The packaged artifact or result. None is returned if there was a problem while packing the object. If
                 a prefix of dict or list unpacking was provided in the log hint key, a list of all the arbitrary number
                 of packaged objects will be returned.
        """
        # Get the key to see if needed to pack arbitrary number of objects via list or dict prefixes:
        log_hint_key = log_hint[LogHintKey.KEY]
        if log_hint_key.startswith("**"):
            # A dictionary unpacking prefix was given, validate the object is a dictionary and prepare the objects to
            # pack with their keys:
            if not isinstance(obj, dict):
                logger.warn(
                    f"The log hint key '{log_hint_key}' has a dictionary unpacking prefix ('**') to log arbitrary "
                    f"number of objects within the dictionary, but a dictionary was not provided, the given object is "
                    f"of type '{self._get_type_name(type(obj))}'. The object is ignored, to log it, please remove the "
                    f"'**' prefix from the key."
                )
                return None
            objects_to_pack = {
                f"{log_hint_key[len('**'):]}{dict_key}": dict_obj
                for dict_key, dict_obj in obj.items()
            }
        elif log_hint_key.startswith("*"):
            # A list unpacking prefix was given, validate the object is a list and prepare the objects to pack with
            # their keys:
            if not isinstance(obj, list):
                logger.warn(
                    f"The log hint key '{log_hint_key}' has a list unpacking prefix ('*') to log arbitrary number of "
                    f"objects within the list, but a list was not provided, the given object is of type "
                    f"'{self._get_type_name(type(obj))}'. The object is ignored, to log it, please remove the '*' "
                    f"prefix from the key."
                )
                return None
            objects_to_pack = {
                f"{log_hint_key[len('*'):]}{i}": obj[i] for i in range(len(obj))
            }
        else:
            # A single object is required to be packaged:
            objects_to_pack = {log_hint_key: obj}

        # Go over the collected keys and objects and pack them:
        packages = []
        for key, per_key_obj in objects_to_pack.items():
            # Edit the key in the log hint:
            per_key_log_hint = log_hint.copy()
            per_key_log_hint[LogHintKey.KEY] = key
            # Pack and collect the package:
            try:
                packages.append(self._pack(obj=per_key_obj, log_hint=per_key_log_hint))
            except Exception as exception:
                raise MLRunPackagePackingError(
                    f"An exception was raised during the packing of '{per_key_log_hint}'."
                ) from exception

        # If multiple packages were packed, return a list, otherwise return the single package:
        return packages if len(packages) > 1 else packages[0]

    def unpack(self, data_item: DataItem, type_hint: Type) -> Any:
        """
        Unpack an object using one of the manager's packagers. The data item can be unpacked in two options:

        * As a package: If the data item contains a package and the type hint provided is equal to the object
          type noted in the package. Or, if it's a package and a type hint was not provided.
        * As a data item: If the data item is not a package or the type hint provided is not equal to the one noted in
          the package.

        Notice: It is not recommended to use a different packager than the one who originally packed the object to
        unpack it. A warning will be shown in that case.

        :param data_item: The data item holding the package.
        :param type_hint: The type hint to parse the data item as.

        :return: The unpacked object parsed as type hinted.
        """
        # Set variables to hold the manager notes and packager instructions:
        artifact_key = None
        packager_name = None
        object_type = None
        artifact_type = None
        instructions = {}

        # Try to get the notes and instructions (can be found only in artifacts but data item may be a simple path/url):
        is_package = False
        if (
            data_item.artifact_url
            and is_store_uri(url=data_item.artifact_url)
            and parse_store_uri(data_item.artifact_url)[0]
            in [StorePrefix.Artifact, StorePrefix.Dataset, StorePrefix.Model]
        ):
            # Get the artifact object in the data item:
            artifact, _ = store_manager.get_store_artifact(url=data_item.artifact_url)
            # Get the key from the artifact's metadata and instructions from the artifact's spec:
            artifact_key = artifact.metadata.key
            packaging_instructions = artifact.spec.packaging_instructions
            # Extract the manager notes and packager instructions found:
            if packaging_instructions:
                is_package = True  # Mark that it is a package.
                packager_name = packaging_instructions[
                    self._InstructionsNotesKey.PACKAGER_NAME
                ]
                object_type = self._get_type_from_name(
                    type_name=packaging_instructions[
                        self._InstructionsNotesKey.OBJECT_TYPE
                    ]
                )
                artifact_type = packaging_instructions[
                    self._InstructionsNotesKey.ARTIFACT_TYPE
                ]
                instructions = packaging_instructions[
                    self._InstructionsNotesKey.INSTRUCTIONS
                ]

        # Check how to unpack the data item according to the collected info (if it's a package or a simple data item):
        # Notice: we will always prefer to unpack a package by the packager who packaged it, but the user may provide a
        # different type hint. This is the only scenario we'll ignore the packager name noted in the package and try to
        # unpack with a different one according to the type hint provided (`object_type` and `type_hint` are not equal).
        if is_package:
            # It's a package, continue according to the provided type hint:
            if type_hint is None:
                # User count on the type noted in the package so we unpack it as is:
                unpack_as_package = True
            else:
                # A type hint is provided, check if the type hint is equal to the one noted in the package:
                matching_object_and_type_hint = False
                hinted_types = [type_hint]
                while len(hinted_types) > 0:
                    if object_type in hinted_types:
                        matching_object_and_type_hint = True
                        break
                    hinted_types = TypingUtils.reduce_type_hint(type_hint=hinted_types)
                if matching_object_and_type_hint:
                    # They are equal, so we will unpack the package as is:
                    unpack_as_package = True
                else:
                    # They are not equal, so we can't count on the original packager noted on the package as the user
                    # require different type, so we unpack as data item:
                    logger.warn(
                        f"{artifact_key} was originally packaged as type '{object_type}' but the type hint given to "
                        f"unpack it as is '{type_hint}'. It is recommended to not parse an object from type to type "
                        f"using a packager as unknown behavior might happen."
                    )
                    unpack_as_package = False
        else:
            # Not a package, unpack as data item:
            unpack_as_package = False

        # Unpack:
        try:
            if unpack_as_package:
                return self._unpack_package(
                    data_item=data_item,
                    artifact_key=artifact_key,
                    artifact_type=artifact_type,
                    packager_name=packager_name,
                    instructions=instructions,
                )
            return self._unpack_data_item(
                data_item=data_item,
                type_hint=type_hint,
            )
        except Exception as exception:
            raise MLRunPackageUnpackingError(
                f"An exception was raised during the unpacking of '{data_item.key}'."
            ) from exception

    def link_packages(
        self,
        additional_artifacts: List[Artifact],
        additional_results: dict,
    ):
        """
        Link packages between each other according to the provided extra data and metrics spec keys. A future link is
        marked with ellipses (...). If no link was found, None will be used and a warning will get printed.

        :param additional_artifacts: Additional artifacts to link (should come from a `mlrun.MLClientCtx`).
        :param additional_results:   Additional results to link (should come from a `mlrun.MLClientCtx`).
        """
        # Join the manager's artifacts and results with the additional ones to look for a link in all of them:
        joined_artifacts = [*additional_artifacts, *self.artifacts]
        joined_results = {**additional_results, **self.results}

        # Go over the artifacts and link:
        for artifact in self.artifacts:
            # Go over the extra data keys:
            for key in artifact.spec.extra_data:
                # Future link is marked with ellipses (...):
                if artifact.spec.extra_data[key] is ...:
                    # Look for an artifact or result with this key to link it:
                    extra_data = self._look_for_extra_data(
                        key=key, artifacts=joined_artifacts, results=joined_results
                    )
                    # Print a warning if a link is missing:
                    if extra_data is None:
                        logger.warn(
                            f"Could not find {key} to link as extra data for {artifact.key}."
                        )
                    # Link it (None will be used in case it was not found):
                    artifact.spec.extra_data[key] = extra_data
            # Go over the metrics keys if available (`ModelArtifactSpec` has a metrics property that may be waiting for
            # values from logged results):
            if hasattr(artifact.spec, "metrics"):
                for key in artifact.spec.metrics:
                    # Future link is marked with ellipses (...):
                    if artifact.spec.metrics[key] is ...:
                        # Link it (None will be used in case it was not found):
                        artifact.spec.metrics[key] = joined_results.get(key, None)

    def clear_packagers_outputs(self):
        """
        Clear the outputs of all packagers. This method should be called at the end of the run after logging all
        artifacts as some will require uploading the files that will be deleted in this method.
        """
        for packager in self._packagers:
            for path in packager.get_clearing_path_list():
                if not os.path.exists(path):
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    class _InstructionsNotesKey:
        """
        Library of keys for the packager instructions to be added to the packed artifact's spec.
        """

        PACKAGER_NAME = "packager_name"
        OBJECT_TYPE = "object_type"
        ARTIFACT_TYPE = "artifact_type"
        INSTRUCTIONS = "instructions"

    def _try_to_collect_packagers_from_module(
        self, module_name: str, packagers: List[str]
    ):
        """
        Collect a packagers of a given module only if it was successfully imported.

        :param module_name: The module name to try to import.
        :param packagers:   The packagers to collect.
        """
        try:
            importlib.import_module(module_name)

            self.collect_packagers(packagers=packagers)
        except ModuleNotFoundError:
            pass

    def _collect_packagers(self):
        """
        Collect MLRun's builtin packagers. In addition, more `mlrun.frameworks` packagers are added if the interpreter
        has the frameworks. The priority will be as follows (from higher to lower priority):

        1. `mlrun.frameworks` packagers
        2. MLRun's optional packagers
        3. MLRun's mandatory packagers (MLRun's requirements)
        """
        # Collect MLRun's requirements packagers (mandatory):
        self.collect_packagers(
            packagers=[
                f"mlrun.package.packagers.{module_name}_packagers.*"
                for module_name in self._MLRUN_REQUIREMENTS_PACKAGERS
            ]
        )

        # Add extra packagers for optional libraries:
        for module_name in self._EXTENDED_PACKAGERS:
            self._try_to_collect_packagers_from_module(
                module_name=module_name,
                packagers=[f"mlrun.package.packagers.{module_name}_packagers.*"],
            )

    def _get_packager_by_name(self, name: str) -> Union[Packager, None]:
        """
        Look for a packager with the given name and return it.

        If a packager was not found None will be returned.

        :param name: The name of the packager to get.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager by exact name:
        for packager in self._packagers:
            if packager.__name__ == name:
                return packager

        # No packager was found:
        logger.warn(f"The packager '{name}' was not found.")
        return None

    def _get_packager_by_type(
        self, object_type: Type, artifact_type: str = None
    ) -> Union[Packager, None]:
        """
        Look for a packager that can handle the provided object type and can also pack it as the provided artifact type.

        If a packager was not found None will be returned.

        :param object_type:   The object type the packager to get should handle.
        :param artifact_type: The artifact type the packager to get should pack and log as.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager for the combination of object type nad artifact type:
        possible_type_hints = [object_type]
        while len(possible_type_hints) > 0:
            for type_hint in possible_type_hints:
                for packager in self._packagers:
                    if packager.is_packable(
                        object_type=type_hint, artifact_type=artifact_type
                    ):
                        return packager
            # Reduce the type hint list and continue:
            possible_type_hints = TypingUtils.reduce_type_hint(
                type_hint=possible_type_hints
            )

        # No packager was found:
        return None

    def _pack(self, obj: Any, log_hint: dict) -> Union[Artifact, dict, None]:
        """
        Pack an object using one of the manager's packagers.

        :param obj:      The object to pack as an artifact.
        :param log_hint: The log hint to use.

        :return: The packaged artifact or result. None is returned if there was a problem while packing the object.
        """
        # Get the artifact type (if user didn't pass any, the packager will use its configured default):
        artifact_type = log_hint.pop(LogHintKey.ARTIFACT_TYPE, None)

        # Get a packager:
        object_type = type(obj)
        packager = self._get_packager_by_type(
            object_type=object_type, artifact_type=artifact_type
        )
        if packager is None:
            logger.warn(
                f"No packager was found for the combination of 'object_type={self._get_type_name(typ=object_type)}' "
                f"and 'artifact_type={artifact_type}'."
            )
            return None

        # Use the packager to pack the object:
        packed_object = packager.pack(
            obj=obj, artifact_type=artifact_type, configurations=log_hint
        )

        # Check if the packaged object is None, meaning there was an error in the process of packing it:
        if packed_object is None:
            return None

        # If the packed object is a result, return it as is:
        if isinstance(packed_object, dict):
            # Collect the result and return:
            self._results.update(packed_object)
            return packed_object

        # It is an artifact, continue with the packaging:
        artifact, instructions = packed_object

        # Prepare the manager's labels:
        packaging_instructions = {
            self._InstructionsNotesKey.PACKAGER_NAME: packager.__name__,
            self._InstructionsNotesKey.OBJECT_TYPE: self._get_type_name(
                typ=object_type
            ),
            self._InstructionsNotesKey.ARTIFACT_TYPE: artifact_type,
            self._InstructionsNotesKey.INSTRUCTIONS: instructions,
        }

        # Set the instructions in the artifact's spec:
        artifact.spec.packaging_instructions = packaging_instructions

        # Collect the artifact and return:
        self._artifacts.append(artifact)
        return artifact

    def _unpack_package(
        self,
        data_item: DataItem,
        artifact_key: str,
        artifact_type: str,
        packager_name: str,
        instructions: dict,
    ) -> Any:
        """
        Unpack a data item as a package using the given notes.

        :param data_item:     The data item to unpack.
        :param artifact_key:  The artifact's key (used only to raise a meaningful error message in case of an error).
        :param artifact_type: The artifact type to unpack as.
        :param packager_name: The packager's name to use for the unpacking.
        :param instructions:  The instructions to pass to the packager's unpack method.

        :return: The unpacked object.

        :raise MLRunPackageUnpackingError: If there is no packager with the given name.
        """
        # Get the packager by the given name:
        packager = self._get_packager_by_name(name=packager_name)
        if packager is None:
            raise MLRunPackageUnpackingError(
                f"{artifact_key} was originally packaged by a packager of type '{packager_name}' but it "
                f"was not found. Custom packagers should be added to the project running the function "
                f"using the `add_custom_packager` method and make sure the function was set in the project "
                f"with the attribute 'with_repo=True`."
            )

        # Unpack:
        return packager.unpack(
            data_item=data_item,
            artifact_type=artifact_type,
            instructions=instructions,
        )

    def _unpack_data_item(self, data_item: DataItem, type_hint: Type):
        """
        Unpack a data item to the desired hinted type.

        :param data_item: The data item to unpack.
        :param type_hint: The type hint to unpack it to.

        :return: The unpacked object.

        :raise MLRunPackageUnpackingError: If there is no packager that supports the provided type hint.
        """
        # Get the packager by the given type:
        packager = self._get_packager_by_type(object_type=type_hint)
        if packager is None:
            raise MLRunPackageUnpackingError(
                f"Could not find a packager that supports the hinted type: '{type_hint}'"
            )
        if packager is self._default_packager:
            logger.info(
                f"Trying to use the default packager to unpack the data item '{data_item.key}'"
            )

        # Unpack:
        return packager.unpack(
            data_item=data_item,
            artifact_type=None,
            instructions={},
        )

    @staticmethod
    def _look_for_extra_data(
        key: str,
        artifacts: List[Artifact],
        results: dict,
    ) -> Union[Artifact, str, int, float, None]:
        """
        Look for an extra data item (artifact or result) by given key. If not found, None is returned.

        :param key:       Key to look for.
        :param artifacts: Artifacts to look in.
        :param results:   Results to look in.

        :return: The artifact or result with the same key or None if not found.
        """
        for artifact in artifacts:
            if key == artifact.key:
                return artifact
        return results.get(key, None)

    @staticmethod
    def _split_module_path(module_path: str) -> Tuple[str, str]:
        """
        Split a module path to the module name and the class name. Notice inner classes are not supported.

        :param module_path: The module path to split.

        :return: A tuple of strings of the module name and the class name.
        """
        # Set the main script module in case there is no module to be found:
        if "." not in module_path:
            module_path = f"__main__.{module_path}"

        # Split and return:
        module_name, class_name = module_path.rsplit(".", 1)
        return module_name, class_name

    @staticmethod
    def _get_type_name(typ: Type) -> str:
        """
        Get an object type full name - its module path. For example, the name of a pandas data frame will be "DataFrame"
        but its full name (module path) is: "pandas.core.frame.DataFrame".

        Notice: Type hints are not an object type. They are as their name suggests, only hints. As such, typing hints
        should not be given to this function (they do not have '__name__' and '__qualname__' attributes for example).

        :param typ: The object's type to get its full name.

        :return: The object's type full name.
        """
        # Get the module name:
        module_name = typ.__module__

        # Get the type's (class) name
        class_name = typ.__qualname__ if hasattr(typ, "__qualname__") else typ.__name__
        return f"{module_name}.{class_name}"

    @staticmethod
    def _get_type_from_name(type_name: str) -> Type:
        """
        Get the type object out of the given module path. The module must be a full module path (for example:
        "pandas.DataFrame" and not "DataFrame") otherwise it assumes to be from the local run module - __main__.

        :param type_name: The type full name (module path) string.

        :return: The represented type as imported from its module.
        """
        module_name, class_name = PackagersManager._split_module_path(
            module_path=type_name
        )
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
