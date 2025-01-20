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
#
import importlib
import inspect
import os
import shutil
import traceback
from typing import Any, Optional, Union

import mlrun.errors
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem, get_store_resource, store_manager
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.utils import logger

from .errors import (
    MLRunPackageCollectionError,
    MLRunPackagePackingError,
    MLRunPackageUnpackingError,
)
from .packager import Packager
from .packagers.default_packager import DefaultPackager
from .utils import LogHintKey, TypeHintUtils


class PackagersManager:
    """
    A packager manager holds the project's packagers and sends them objects to pack, and data items to unpack.

    It prepares the instructions / log hint configurations and then looks for the first packager that fits the task.
    """

    def __init__(self, default_packager: Optional[type[Packager]] = None):
        """
        Initialize a packagers manager.

        :param default_packager: The default packager should be a packager that fits all types. It
                                 should fit any packagers that are managed by the manager that do not fit an
                                 object or data item. Default to ``mlrun.DefaultPackager``.
        """
        # Set the default packager:
        self._default_packager = (default_packager or DefaultPackager)()

        # Initialize the packagers list (with the default packager in it):
        self._packagers: list[Packager] = []

        # Set an artifacts list and results dictionary to collect all packed objects (will be used later to write extra
        # data if noted by the user using the log hint key "extra_data")
        self._artifacts: list[Artifact] = []
        self._results = {}

    @property
    def artifacts(self) -> list[Artifact]:
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

    def collect_packagers(
        self, packagers: list[Union[type[Packager], str]], default_priority: int = 5
    ):
        """
        Collect the provided packagers. Packagers passed as module paths are imported and validated to be of type
        `Packager`. If it's needed to import all packagers from a module, use the module path with an asterisk
        "*" at the end. (A packager with a name that starts with an underscore '_' is not collected.)

        Notice: Only packagers that are declared in the module are collected (packagers imported in the module scope
        aren't collected). For example::

            from mlrun import Packager
            from x import XPackager


            class YPackager(Packager):
                pass

        Only "YPackager" is collected since it is declared in the module, but not "XPackager", which is only imported.

        :param packagers:        List of packagers to add.
        :param default_priority: The default priority for the packagers that don't have a set priority (equals to ...).

        :raise MLRunPackageCollectingError: In case the packager could not be collected.
        """
        # Collect the packagers:
        for packager in packagers:
            # If it's a string, it's the module path of the class, so we import it:
            if isinstance(packager, str):
                # TODO: For supporting Hub packagers, if the string is a hub url, then look in the labels for the
                #       packagers to import and import the function as a module.
                # Import the module:
                module_name, class_name = self._split_module_path(module_path=packager)
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError as module_not_found_error:
                    raise MLRunPackageCollectionError(
                        f"The packager '{class_name}' could not be collected from the module '{module_name}' as it "
                        f"cannot be imported: {module_not_found_error}"
                    ) from module_not_found_error
                # Check if needed to import all packagers from the given module:
                if class_name == "*":
                    # Get all the packagers from the module and collect them (this time they will be sent as `Packager`
                    # types to the method):
                    self.collect_packagers(
                        packagers=[
                            member
                            for _, member in inspect.getmembers(
                                module,
                                lambda m: (
                                    # Validate it is declared in the module:
                                    hasattr(m, "__module__")
                                    and m.__module__ == module.__name__
                                    # Validate it is a `Packager`:
                                    and isinstance(m, type)
                                    and issubclass(m, Packager)
                                    # Validate it is not a "protected" `Packager`:
                                    and not m.__name__.startswith("_")
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
                    raise MLRunPackageCollectionError(
                        f"The packager '{class_name}' could not be collected as it does not exist in the module "
                        f"'{module.__name__}': {attribute_error}"
                    ) from attribute_error
            # Validate the class given is a `Packager` type:
            if not issubclass(packager, Packager):
                raise MLRunPackageCollectionError(
                    f"The packager '{packager.__name__}' could not be collected as it is not a `mlrun.Packager`."
                )
            # Initialize the packager class:
            packager = packager()
            # Set default priority in case it is not set in the packager's class:
            if packager.priority is ...:
                packager.priority = default_priority
            # Collect the packager (putting him first in the list for highest priority:
            self._packagers.insert(0, packager)
            # For debugging, we'll print the collected packager:
            logger.debug(
                f"The packagers manager collected the packager: {str(packager)}"
            )

        # Sort the packagers:
        self._packagers.sort()

    def pack(
        self, obj: Any, log_hint: dict[str, str]
    ) -> Union[Artifact, dict, None, list[Union[Artifact, dict, None]]]:
        """
        Pack an object using one of the manager's packagers. A `dict` ("**") or `list` ("*") unpacking syntax in the
        log hint key packs the objects within them in separate packages.

        :param obj:      The object to pack as an artifact.
        :param log_hint: The log hint to use.

        :return: The packaged artifact or result. None is returned if there was a problem while packing the object. If
                 a prefix of dict or list unpacking was provided in the log hint key, a list of all the arbitrary number
                 of packaged objects is returned.

        :raise MLRunInvalidArgumentError: If the key in the log hint instructs to log an arbitrary number of artifacts
                                          but the object type does not match the "*" or "**" used in the key.
        :raise MLRunPackagePackingError:  If there was an error during the packing.
        """
        # Get the key to see if needed to pack arbitrary number of objects via list or dict prefixes:
        log_hint_key = log_hint[LogHintKey.KEY]
        if log_hint_key.startswith("**"):
            # A dictionary unpacking prefix was given, validate the object is a dictionary and prepare the objects to
            # pack with their keys:
            if not isinstance(obj, dict):
                raise MLRunInvalidArgumentError(
                    f"The log hint key '{log_hint_key}' has a dictionary unpacking prefix ('**') to log arbitrary "
                    f"number of objects within the dictionary, but a dictionary was not provided, the given object is "
                    f"of type '{self._get_type_name(type(obj))}'. The object is ignored, to log it, please remove the "
                    f"'**' prefix from the key."
                )
            objects_to_pack = {
                f"{log_hint_key[len('**'):]}{dict_key}": dict_obj
                for dict_key, dict_obj in obj.items()
            }
        elif log_hint_key.startswith("*"):
            # An iterable unpacking prefix was given, validate the object is iterable and prepare the objects to pack
            # with their keys:
            is_iterable = True
            try:
                for _ in obj:
                    break
            except TypeError:
                is_iterable = False
            if not is_iterable:
                raise MLRunInvalidArgumentError(
                    f"The log hint key '{log_hint_key}' has an iterable unpacking prefix ('*') to log arbitrary number "
                    f"of objects within it (like a `list` or `set`), but an iterable object was not provided, the "
                    f"given object is of type '{self._get_type_name(type(obj))}'. The object is ignored, to log it, "
                    f"please remove the '*' prefix from the key."
                )
            objects_to_pack = {
                f"{log_hint_key[len('*'):]}{i}": obj_i for i, obj_i in enumerate(obj)
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
                    f"An exception was raised during the packing of '{per_key_log_hint}': {exception}"
                ) from exception

        # If multiple packages were packed, return a list, otherwise return the single package:
        return packages if len(packages) > 1 else packages[0]

    def unpack(self, data_item: DataItem, type_hint: type) -> Any:
        """
        Unpack an object using one of the manager's packagers. The data item can be unpacked in two ways:

        * As a package: If the data item contains a package and the type hint provided is equal to the object
          type noted in the package. Or, if it's a package and a type hint was not provided.
        * As a data item: If the data item is not a package or the type hint provided is not equal to the one noted in
          the package.

        If the type hint is a `mlrun.DataItem` then it won't be unpacked.

        Notice: It is not recommended to use a different packager than the one that originally packed the object to
        unpack it. A warning displays in that case.

        :param data_item: The data item holding the package.
        :param type_hint: The type hint to parse the data item as.

        :return: The unpacked object parsed as type hinted.
        """
        # Check if `DataItem` is hinted - meaning the user can expect a data item and do not want to unpack it:
        if TypeHintUtils.is_matching(object_type=DataItem, type_hint=type_hint):
            return data_item

        # Set variables to hold the manager notes and packager instructions:
        artifact_key = None
        packaging_instructions = None

        # Try to get the notes and instructions (can be found only in artifacts but data item may be a simple path/url):
        if data_item.get_artifact_type():
            # Get the artifact object in the data item:
            artifact, _ = store_manager.get_store_artifact(url=data_item.artifact_url)
            # Get the key from the artifact's metadata and instructions from the artifact's spec:
            artifact_key = artifact.metadata.key
            packaging_instructions = artifact.spec.unpackaging_instructions

        # Unpack:
        try:
            if packaging_instructions:
                # The data item is a package and the object type is equal or part of the type hint (part of is in case
                # of a `typing.Union` for example):
                return self._unpack_package(
                    data_item=data_item,
                    artifact_key=artifact_key,
                    packaging_instructions=packaging_instructions,
                    type_hint=type_hint,
                )
            # The data item is not a package or the object type is not equal or part of the type hint:
            return self._unpack_data_item(
                data_item=data_item,
                type_hint=type_hint,
            )
        except Exception as exception:
            raise MLRunPackageUnpackingError(
                f"An exception was raised during the unpacking of '{data_item.key}': {exception}"
            ) from exception

    def link_packages(
        self,
        additional_artifact_uris: dict,
        additional_results: dict,
    ):
        """
        Link packages to each other according to the provided extra data and metrics spec keys. A future link is
        marked with ellipses (...). If no link is found, None is used and a warning is printed.

        :param additional_artifact_uris:    Additional artifact URIs to link (should come from an `mlrun.MLClientCtx`).
        :param additional_results:          Additional results to link (should come from an `mlrun.MLClientCtx`).
        """
        # Join the manager's artifacts and results with the additional ones to look for a link in all of them:
        joined_results = {**additional_results, **self.results}

        # Go over the artifacts and link:
        for artifact in self.artifacts:
            # Go over the extra data keys:
            for key in artifact.spec.extra_data:
                # Future link is marked with ellipses (...):
                if artifact.spec.extra_data[key] is ...:
                    # Look for an artifact or result with this key to link it:
                    extra_data = self._look_for_extra_data(
                        key=key,
                        artifacts=self.artifacts,
                        artifact_uris=additional_artifact_uris,
                        results=joined_results,
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
        Clear the outputs of all packagers. This method should be called at the end of the run, only after logging all
        artifacts, to ensure that files that require uploading have already been uploaded.
        """
        for packager in self._get_packagers_with_default_packager():
            for path in packager.future_clearing_path_list:
                if not os.path.exists(path):
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            packager.future_clearing_path_list.clear()

    class _InstructionsNotesKey:
        """
        Library of keys for the packager instructions to be added to the packed artifact's spec.
        """

        PACKAGER_NAME = "packager_name"
        OBJECT_TYPE = "object_type"
        ARTIFACT_TYPE = "artifact_type"
        INSTRUCTIONS = "instructions"

    def _get_packagers_with_default_packager(self) -> list[Packager]:
        """
        Get the full list of packagers - the collected packagers and the default packager (located at last place in the
        list - the lowest priority).

        :return: A list of the manager's packagers with the default packager.
        """
        return [*self._packagers, self._default_packager]

    def _get_packager_by_name(self, name: str) -> Union[Packager, None]:
        """
        Look for a packager with the given name and return it.

        If a packager was not found None will be returned.

        :param name: The name of the packager to get.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager by exact name:
        for packager in self._get_packagers_with_default_packager():
            if packager.__class__.__name__ == name:
                return packager

        # No packager was found:
        logger.warn(f"The packager '{name}' was not found.")
        return None

    def _get_packager_for_packing(
        self,
        obj: Any,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> Union[Packager, None]:
        """
        Look for a packager that can pack the provided object as the provided artifact type.

        If a packager was not found None will be returned.

        :param obj:            The object to pack.
        :param artifact_type:  The artifact type the packager to get should pack / unpack as.
        :param configurations: The log hint configurations passed by the user.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager for the combination of object nad artifact type:
        for packager in self._packagers:
            if packager.is_packable(
                obj=obj, artifact_type=artifact_type, configurations=configurations
            ):
                return packager

        # No packager was found:
        return None

    def _get_packager_for_unpacking(
        self,
        data_item: Any,
        type_hint: type,
        artifact_type: Optional[str] = None,
    ) -> Union[Packager, None]:
        """
        Look for a packager that can unpack the data item of the given type hint as the provided artifact type.

        If a packager was not found None will be returned.

        :param data_item:     The data item to unpack.
        :param type_hint:     The type hint the packager to get should handle.
        :param artifact_type: The artifact type the packager to get should pack / unpack as.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager for the combination of object type nad artifact type:
        for packager in self._packagers:
            if packager.is_unpackable(
                data_item=data_item, type_hint=type_hint, artifact_type=artifact_type
            ):
                return packager

        # No packager was found:
        return None

    def _pack(self, obj: Any, log_hint: dict) -> Union[Artifact, dict, None]:
        """
        Pack an object using one of the manager's packagers.

        :param obj:      The object to pack as an artifact.
        :param log_hint: The log hint to use.

        :return: The packaged artifact or result. None is returned if there was a problem while packing the object.
        """
        # Get the artifact type (if user didn't pass any, the packager will use its configured default) and key:
        artifact_type = log_hint.pop(LogHintKey.ARTIFACT_TYPE, None)
        key = log_hint.pop(LogHintKey.KEY, None)

        # Get a packager:
        packager = self._get_packager_for_packing(
            obj=obj, artifact_type=artifact_type, configurations=log_hint
        )
        if packager is None:
            if self._default_packager.is_packable(
                obj=obj, artifact_type=artifact_type, configurations=log_hint
            ):
                logger.info(f"Using the default packager to pack the object '{key}'")
                packager = self._default_packager
            else:
                raise MLRunPackagePackingError(
                    f"No packager was found for the combination of "
                    f"'object_type={self._get_type_name(typ=type(obj))}' and 'artifact_type={artifact_type}'."
                )

        # Use the packager to pack the object:
        packed_object = packager.pack(
            obj=obj, key=key, artifact_type=artifact_type, configurations=log_hint
        )

        # If the packed object is a result, return it as is:
        if isinstance(packed_object, dict):
            # Collect the result and return:
            self._results.update(packed_object)
            return packed_object

        # It is an artifact, continue with the packaging:
        artifact, instructions = packed_object

        # Prepare the manager's unpackaging instructions:
        unpackaging_instructions = {
            self._InstructionsNotesKey.PACKAGER_NAME: packager.__class__.__name__,
            self._InstructionsNotesKey.OBJECT_TYPE: self._get_type_name(typ=type(obj)),
            self._InstructionsNotesKey.ARTIFACT_TYPE: (
                artifact_type
                if artifact_type
                else packager.get_default_packing_artifact_type(obj=obj)
            ),
            self._InstructionsNotesKey.INSTRUCTIONS: instructions,
        }

        # Set the instructions in the artifact's spec:
        artifact.spec.unpackaging_instructions = unpackaging_instructions

        # Collect the artifact and return:
        self._artifacts.append(artifact)
        return artifact

    def _unpack_package(
        self,
        data_item: DataItem,
        artifact_key: str,
        packaging_instructions: dict,
        type_hint: type,
    ) -> Any:
        """
        Unpack a data item as a package using the given notes.

        :param data_item:              The data item to unpack.
        :param artifact_key:           The artifact's key (used only to raise a meaningful error message in case of an
                                       error).
        :param packaging_instructions: The manager's noted instructions.
        :param type_hint:              The user's type hint.

        :return: The unpacked object.

        :raise MLRunPackageUnpackingError: If there is no packager with the given name.
        """
        # Extract the packaging instructions:
        packager_name = packaging_instructions[self._InstructionsNotesKey.PACKAGER_NAME]
        try:
            # For validation, we'll try to get the type of the original packaged object. The original object type might
            # not be available for 3 reasons:
            # 1. The user is trying to parse the data item to a different type than the one it was packaged - meaning it
            #    is ok to be missing, the method will call `unpack_data_item` down the road.
            # 2. The interpreter does not have the required module to unpack this object meaning it will not have the
            #    original packager as well, so it will try to use another package before raising an error.
            # 3. An edge case where the user declared the class at the MLRun function itself. Read the long warning to
            #    understand more.
            self._get_type_from_name(
                type_name=packaging_instructions[self._InstructionsNotesKey.OBJECT_TYPE]
            )
        except ModuleNotFoundError:
            logger.warn(
                f"Could not import the original type "
                f"('{packaging_instructions[self._InstructionsNotesKey.OBJECT_TYPE]}') of the input artifact "
                f"'{artifact_key}' due to a `ModuleNotFoundError`.\n"
                f"Note: If you wish to parse the input to a different type (which is not recommended) you may ignore "
                f"this warning. Otherwise, make sure the interpreter has the required module to import the type.\n"
                f"If it does, you probably implemented the class at the same file of your MLRun function, making "
                f"Python collect it twice: one from the object's own Packager class and another from the function "
                f"code. When MLRun is converting code to a MLRun function, it counts on it to be able to be imported "
                f"as a stand alone file. If other classes (like the packager who imports it) require objects declared "
                f"in this file, it is no longer stand alone. For example:\n\n"
                f""
                f"Let us look at a file '/src/my_module/my_file.py':"
                f"\tclass MyClass:\n"
                f"\t\tpass\n\n"
                f"\tclass MyClassPackager(Packager):\n"
                f"\t\tPACKABLE_OBJECT_TYPE = MyClass\n\n"
                f""
                f"The packager of this class will have the class variable `PACKABLE_OBJECT_TYPE=MyClass` where "
                f"`MyClass`'s module is `src.my_module.my_file.MyClass` because it is being collected from the repo "
                f"downloaded with the project.\n"
                f"But, if creating a MLRun function of '/src/my_module/my_file.py', then 'my_file.py' will be imported "
                f"as a stand alone module, making the same class to be imported twice: one time as `my_file.MyClass` "
                f"from the stand alone function, and another from the packager who has the correct full module path: "
                f"`src.my_module.my_file.MyClass`. This will cause both classes, although the same, to be not equal "
                f"and the first one to be not even importable outside the scope of 'my_file.py' - yielding this "
                f"warning."
            )
        artifact_type = packaging_instructions[self._InstructionsNotesKey.ARTIFACT_TYPE]
        instructions = (
            packaging_instructions[self._InstructionsNotesKey.INSTRUCTIONS] or {}
        )

        # Get the original packager by its name:
        packager = self._get_packager_by_name(name=packager_name)

        # Check if the original packager can be used (the user do not count on parsing to a different type):
        unpack_as_package = False
        if packager is None:
            # The original packager was not found, the user either did not add the custom packager or perhaps wants
            # to unpack the data item as a different type than the original one. We will warn and continue to unpack as
            # a non-package data item:
            logger.warn(
                f"{artifact_key} was originally packaged by a packager of type '{packager_name}' but it "
                f"was not found. Custom packagers should be added to the project running the function "
                f"using the `add_custom_packager` method and make sure the function was set in the project "
                f"with the attribute 'with_repo=True`.\n"
                f"MLRun will try to unpack according to the provided type hint in code."
            )
        elif type_hint is None:
            # User count on the type noted in the package, so we unpack it as is:
            unpack_as_package = True
        else:
            # A type hint is provided, check if the type hint is packable by the packager:
            type_hints = {type_hint}
            while not unpack_as_package and len(type_hints) > 0:
                # Check for each hint (one match is enough):
                for hint in type_hints:
                    if packager.is_unpackable(
                        data_item=data_item, type_hint=hint, artifact_type=artifact_type
                    ):
                        unpack_as_package = True
                        break
                if not unpack_as_package:
                    # Reduce the hints and continue:
                    type_hints = TypeHintUtils.reduce_type_hint(type_hint=type_hints)
            if not unpack_as_package:
                # They are not equal, so we can't count on the original packager noted on the package as the user
                # require different type, so we unpack as data item:
                logger.warn(
                    f"{artifact_key} was originally packaged by '{packager_name}' but the type hint given to "
                    f"unpack it as '{type_hint}' is not supported by it. MLRun will try to look for a matching "
                    f"packager to the type hint instead. Note: it is not recommended to parse an object from type to "
                    f"type using the unpacking mechanism of packagers as unknown behavior might happen."
                )

        # Unpack:
        if unpack_as_package:
            return packager.unpack(
                data_item=data_item,
                artifact_type=artifact_type,
                instructions=instructions,
            )
        return self._unpack_data_item(data_item=data_item, type_hint=type_hint)

    def _unpack_data_item(self, data_item: DataItem, type_hint: type):
        """
        Unpack a data item to the desired hinted type. In case the type hint includes multiple types (as in the case of
        `typing.Union`), the manager goes over the types, and reduces them while looking for the first packager that
        can successfully unpack the data item.

        :param data_item: The data item to unpack.
        :param type_hint: The type hint to unpack it to.

        :return: The unpacked object.

        :raise MLRunPackageUnpackingError: If there is no packager that supports the provided type hint.
        """
        # Prepare a list of a packager and exception string for all the failures in case there was no fitting packager:
        found_packagers: list[tuple[Packager, str]] = []

        # Try to unpack as one of the possible types in the type hint:
        possible_type_hints = {type_hint}
        while len(possible_type_hints) > 0:
            for hint in possible_type_hints:
                # Get the packager by the given type:
                packager = self._get_packager_for_unpacking(
                    data_item=data_item, type_hint=hint
                )
                if packager is None:
                    # No packager was found that supports this hinted type:
                    continue
                # Unpack:
                try:
                    return packager.unpack(
                        data_item=data_item,
                        instructions={},
                    )
                except Exception:
                    # Could not unpack as the reduced type hint, collect the exception and go to the next one:
                    found_packagers.append((packager, traceback.format_exc()))
            # Reduce the type hint list and continue:
            possible_type_hints = TypeHintUtils.reduce_type_hint(
                type_hint=possible_type_hints
            )

        # Check the default packager:
        logger.info(
            f"Trying to use the default packager to unpack the data item '{data_item.key}'"
        )
        try:
            return self._default_packager.unpack(
                data_item=data_item,
                artifact_type=None,
                instructions={},
            )
        except Exception:
            found_packagers.append((self._default_packager, traceback.format_exc()))

        # The method did not return until this point, raise an error:
        raise MLRunPackageUnpackingError(
            f"Could not unpack data item with the hinted type '{type_hint}'. The following packagers were tried to "
            f"be used to unpack it but raised the exceptions joined:\n\n"
            + "\n".join(
                [
                    f"Found packager: '{packager}'\nException: {exception}\n"
                    for packager, exception in found_packagers
                ]
            )
        )

    @staticmethod
    def _look_for_extra_data(
        key: str,
        artifacts: list[Artifact],
        artifact_uris: dict,
        results: dict,
    ) -> Union[Artifact, str, int, float, None]:
        """
        Look for an extra data item (artifact or result) by given key. If not found, None is returned.

        :param key:             Key to look for.
        :param artifacts:       Artifacts to look in.
        :param artifact_uris:   Artifacts URIs to look in.
        :param results:         Results to look in.

        :return: The artifact or result with the same key or None if not found.
        """
        artifact_uris = artifact_uris or {}
        for _key, uri in artifact_uris.items():
            if key == _key:
                try:
                    return get_store_resource(uri)
                except mlrun.errors.MLRunNotFoundError as exc:
                    logger.warn(
                        f"Artifact {key=} not found when looking for extra data",
                        exc=mlrun.errors.err_to_str(exc),
                    )
                    return None

        # Look in the artifacts:
        for artifact in artifacts:
            if key == artifact.key:
                return artifact

        # Look in the results:
        return results.get(key, None)

    @staticmethod
    def _split_module_path(module_path: str) -> tuple[str, str]:
        """
        Split a module path to the module name and the class name. Inner classes are not supported.

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
    def _get_type_name(typ: type) -> str:
        """
        Get an object type full name - its module path. For example, the name of a pandas data frame is "DataFrame"
        but its full name (module path) is: "pandas.core.frame.DataFrame".

        Notice: Type hints are not an object type. They are, as their name suggests, only hints. As such, typing hints
        should not be given to this function (they do not have '__name__' and '__qualname__' attributes for example).

        :param typ: The object's type to get its full name.

        :return: The object's type full name.
        """
        # Get the module name:
        module_name = typ.__module__ if hasattr(typ, "__module__") else ""

        # Get the type's (class) name
        class_name = typ.__qualname__ if hasattr(typ, "__qualname__") else typ.__name__

        return f"{module_name}.{class_name}" if module_name else class_name

    @staticmethod
    def _get_type_from_name(type_name: str) -> type:
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
