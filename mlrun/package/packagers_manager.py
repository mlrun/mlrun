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
import typing
import itertools

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem, is_store_uri, store_manager

from .packager import Packager


class PackagersManager:
    """
    A packager manager is holding the project's packagers and sending them objects to pack and data items to unpack.

    It prepares the instructions / log hint configurations and then looks for the first packager who fits the task.
    That's why when the manager collects its packagers, it first collects builtin MLRun packagers and only then the
    user's custom packagers, this way user's custom packagers will have higher priority.

    The manager should be a singleton to properly function. To get it, use `mlrun.package.get_packagers_manager()`.
    """

    def __init__(self, default_packager: Packager = None):
        """
        Initialize a packagers manager.

        :param default_packager: The default packager to use in case all packagers managed by the manager do not fit an
                                 object or data item.
        """
        # Set the default packager:
        self._default_packager = default_packager or Packager

        # Initialize the packagers list:
        self._packagers: typing.List[Packager] = []

        # Collect the packagers:
        self._collect_packagers()

    def pack(self, obj: typing.Any, log_hint: typing.Dict[str, str]) -> Artifact:
        """
        Pack an object using one of the manager's packagers.

        :param obj:      The object to pack as an artifact.
        :param log_hint: The log hint to use.

        :return: The packaged artifact.
        """
        # Get the artifact type (if user didn't pass any, the packager will use its configured default):
        artifact_type = log_hint.pop("artifact_type", None)

        # Get a packager:
        packager = self._get_packager(
            object_type=type(obj), artifact_type=artifact_type
        )

        # Use the packager to pack the object:
        artifact, instructions = packager.pack(
            obj=obj, artifact_type=artifact_type, configurations=log_hint
        )

        # Prepare the manager's labels:
        package_instructions = {
            self._InstructionsNotesKeys.PACKAGER_NAME: packager.__name__,
            self._InstructionsNotesKeys.OBJECT_TYPE: self._get_type_name(typ=type(obj)),
            self._InstructionsNotesKeys.ARTIFACT_TYPE: artifact_type,
            self._InstructionsNotesKeys.INSTRUCTIONS: instructions,
        }

        # Set the instructions in the artifact's spec:
        artifact.spec.package_instructions = package_instructions

        return artifact

    def unpack(self, data_item: DataItem, type_hint: typing.Type) -> typing.Any:
        """
        Unpack an object using one of the manager's packagers.

        :param data_item: The data item holding the package.
        :param type_hint: The type hint to parse the data item as.

        :return: The unpacked object parsed as type hinted.
        """
        # Set variables to hold the manager notes and packager instructions:
        packager_name = None
        object_type = None
        artifact_type = None
        instructions = {}

        # Try to get the notes and instructions (can be found only in artifacts but data item may be a simple path/url):
        if data_item.artifact_url and is_store_uri(url=data_item.artifact_url):
            # Get the artifact object in the data item:
            artifact, _ = store_manager.get_store_artifact(url=data_item.artifact_url)
            # Get the instructions from the artifact's spec:
            package_instructions = artifact.spec.package_instructions
            # Extract the manager notes and packager instructions found:
            if package_instructions:
                packager_name = package_instructions.pop(
                    self._InstructionsNotesKeys.PACKAGER_NAME, None
                )
                object_type = package_instructions.pop(
                    self._InstructionsNotesKeys.OBJECT_TYPE, None
                )
                if object_type is not None:
                    object_type = self._get_type_from_name(type_name=object_type)
                artifact_type = package_instructions.pop(
                    self._InstructionsNotesKeys.ARTIFACT_TYPE, None
                )
                instructions = package_instructions.pop(
                    self._InstructionsNotesKeys.INSTRUCTIONS, {}
                )

        # If both original packaged object type and user's type hint available, validate they are equal:
        if (
            object_type is not None
            and type_hint is not None
            and object_type is not type_hint
        ):
            pass  # TODO: Mismatch! raise warning, take type hint

        # Get the packager:
        packager = self._get_packager(
            object_type=type_hint or object_type,
            artifact_type=artifact_type,
            packager_name=packager_name,
        )

        # If the packager name is available (noted by manager), validate the original packager who packaged the object
        # was found:
        if packager_name is not None and packager.__name__ != packager_name:
            pass  # TODO: Mismatch! raise warning

        # Unpack:
        return packager.unpack(
            data_item=data_item,
            artifact_type=artifact_type,
            instructions=instructions,
        )

    class _InstructionsNotesKeys:
        """
        Library of keys for the packager instructions to be added to the packed artifact's spec.
        """

        PACKAGER_NAME = "packager_name"
        OBJECT_TYPE = "object_type"
        ARTIFACT_TYPE = "artifact_type"
        INSTRUCTIONS = "instructions"

    def _collect_packagers(self):
        """
        1. Collect basic MLRun requirements packagers like builtins, pathlib, numpy and pandas.
        2. Collect additional mlrun.frameworks packagers.
        3. Collect project specific packagers (custom packagers added to project).
        :return:
        """
        pass

    def _get_packager(
        self, object_type: typing.Type, artifact_type: str, packager_name: str = None
    ) -> Packager:
        """
        Get a packager by the provided arguments. If name is given, a packager with the exact name will be looked for,
        if it wasn't found or name is not provided, the packager will be searched by the object and artifact types.

        The packagers priority is set by the order they were collected (last collected -> highest priority).

        If a packagers was not found, the default packager will be returned.

        :param object_type:   The object type the packager to get should handle.
        :param artifact_type: The artifact type the packager to get should pack and log as.
        :param packager_name: The name of the packager to get.

        :return: The found packager or the default packager if none were found.
        """
        # Try to get a packager by name:
        if packager_name is not None:
            found_packager = self._get_packager_by_name(name=packager_name)
            if found_packager:
                return found_packager

        # Try to get a packager to match the given types:
        found_packager = self._get_packager_by_type(
            object_type=object_type, artifact_type=artifact_type
        )
        if found_packager:
            return found_packager

        # Return the default as no packager was found:
        return self._default_packager

    def _get_packager_by_name(self, name: str) -> typing.Union[Packager, None]:
        """
        Look for a packager with the given name and return it.

        If a packager was not found None will be returned.

        :param name: The name of the packager to get.

        :return: The found packager or None if it wasn't found.
        """
        for packager in self._packagers:
            if packager.__name__ == name:
                return packager
        return None

    def _get_packager_by_type(
        self, object_type: typing.Type, artifact_type: str
    ) -> typing.Union[Packager, None]:
        """
        Look for a packager that can handle the provided object type and can also pack it as the provided artifact type.

        If a packager was not found None will be returned.

        :param object_type:   The object type the packager to get should handle.
        :param artifact_type: The artifact type the packager to get should pack and log as.

        :return: The found packager or None if it wasn't found.
        """
        found_packager: Packager = None

        possible_type_hints = [object_type]
        while found_packager is None and len(possible_type_hints) > 0:
            for type_hint in possible_type_hints:
                for packager in self._packagers:
                    if packager.is_packable(
                        object_type=type_hint, artifact_type=artifact_type
                    ):
                        found_packager = packager
                        break
                if found_packager:
                    break
            possible_type_hints = list(
                itertools.chain(
                    *[
                        self._reduce_typing_type(type_hint=type_hint)
                        for type_hint in possible_type_hints
                    ]
                )
            )

        return found_packager

    @staticmethod
    def _get_type_name(typ: typing.Type) -> str:
        """
        Get a type full name - its module path. For example, the name of a pandas data frame will be "DataFrame" but its
        full name (module path) is: "pandas.core.frame.DataFrame".

        :param typ: The type to get its full name.

        :return: The type's full name.
        """
        # Get the module name:
        module_name = typ.__module__

        # Get the type's (class) name
        class_name = typ.__qualname__ if hasattr(typ, "__qualname__") else typ.__name__
        return f"{module_name}.{class_name}"

    @staticmethod
    def _get_type_from_name(type_name: str) -> typing.Type:
        """
        Get the type object out of the given type representation string.

        :param type_name: The type full name (module path) string.

        :return: The represented type as imported from its module.
        """
        module_name, class_name = type_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def _is_typing_type(type_hint: typing.Type) -> bool:
        """
        Check whether a given type is from the `typing` module.

        :param type_hint: The type to check.

        :return: True if the type is from `typing` and False otherwise.
        """
        return hasattr(type_hint, "___module__") and type_hint.__module__ == "typing"

    @staticmethod
    def _reduce_typing_type(type_hint: typing.Type) -> typing.List[typing.Type]:
        """
        Reduce a `typing` module to its original hinted types. For example: `typing.Union[int, float, typing.List[int]]`
        will return `[int, float, List[int]]` and `typing.List[int]` will return `[list]`.

        If a type hint cannot be reduced, an empty list will be returned.

        :param type_hint: The type hint to reduce.

        :return: The reduced type hint as list of hinted types or an empty list of the type hint could not be reduced.
        """
        # TODO: Remove when we'll no longer support Python 3.7:
        import sys

        if sys.version_info[1] < 8:
            return []

        # If it's a type var, take its constraints (e.g. A = TypeVar("A", int, str) meaning an object of type A should
        # be an integer ot a string). If it doesn't have constraints, return an empty list:
        if isinstance(type_hint, typing.TypeVar):
            if len(type_hint.__constraints__) == 0:
                return []
            return list(type_hint.__constraints__)

        # If it's a forward reference, that means the user could not import the class to type hint it (so we can't
        # either):
        if isinstance(type_hint, typing.ForwardRef):
            return []

        # Get the origin of the typing type. An origin is the subscripted typing type (origin of Union[str, int] is
        # Union). The origin can be one of Callable, Tuple, Union, Literal, Final, ClassVar, Annotated or the actual
        # type alias (e.g. origin of List[int] is list):
        origin = typing.get_origin(type_hint)

        # If the typing type has no origin (e.g. None is returned), we cannot reduce it, so we return an empty list:
        if origin is None:
            return []

        # If the origin is a type of one of builtin, contextlib or collections (for example: List's origin is list)
        # then we can be sure there is nothing to reduce as it's a regular type:
        if not PackagersManager._is_typing_type(type_hint=origin):
            return [origin]

        # Get the type's subscriptions - arguments, in order to reduce it to them (we know for sure there are arguments,
        # otherwise origin would have been None):
        args = typing.get_args(type_hint)

        # Return the reduced type as its arguments according to the origin:
        if origin is typing.Callable:
            # A callable cannot be reduced to its arguments, so we'll return the origin - Callable:
            return [typing.Callable]
        if origin is typing.Literal:
            # Literal arguments are not types, but values. So we'll take the types of the values as the reduced type:
            return [type(arg) for arg in args]
        if origin is typing.Union:
            # A union is reduced to its arguments:
            return list(args)
        if origin is typing.Annotated:
            # Annotated is used to describe (add metadata to) a type, so we take the first argument:
            return [args[0]]
        if origin is typing.Final or origin is typing.ClassVar:
            # Both Final and ClassVar takes only one argument - the type:
            return [args[0]]

        # For Generic types we return an empty list:
        return []
