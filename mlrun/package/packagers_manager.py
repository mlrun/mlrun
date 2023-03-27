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
import itertools
import os
import shutil
import typing

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem, is_store_uri, parse_store_uri, store_manager
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.package.packager import Packager
from mlrun.package.packagers.default_packager import DefaultPackager
from mlrun.utils import StorePrefix, logger


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

        :param default_packager: The default packager to use in case all packagers managed by the manager do not fit an
                                 object or data item.
        """
        # Set the default packager:
        self._default_packager = default_packager or DefaultPackager

        # Set an artifacts list and results dictionary to collect all packed objects (will be used later to write extra
        # data if noted by the user using the log hint key "extra_data")
        self._artifacts: typing.List[Artifact] = []
        self._results = {}

        # Initialize the packagers list:
        self._packagers: typing.List[Packager] = []

        # Collect the builtin standard packagers:
        self._collect_packagers()

    @property
    def artifacts(self) -> typing.List[Artifact]:
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

    def collect_packagers(self, packagers: typing.List[typing.Union[typing.Type, str]]):
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

        :raise MLRunInvalidArgumentError: In case one of the classes provided was not of type `Packager`.
        """
        for packager in packagers:
            # If it's a string, it's the module path of the class, so we import it:
            if isinstance(packager, str):
                # Import the module:
                module_name, class_name = PackagersManager._split_module_path(
                    module_path=packager
                )
                module = importlib.import_module(module_name)
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
                packager = getattr(module, class_name)
            # Validate the class given is a `Packager` type:
            if not isinstance(packager, Packager):
                raise MLRunInvalidArgumentError(
                    f"The object given to collect '{packager.__name__}' is not a `mlrun.Packager`."
                )
            # Collect the packager:
            self._packagers.append(packager)
            # For debugging, we'll print the collected packager:
            logger.debug(
                f"The packagers manager collected the packager: {str(packager)}"
            )

    def pack(
        self, obj: typing.Any, log_hint: typing.Dict[str, str]
    ) -> typing.Union[Artifact, dict, None]:
        """
        Pack an object using one of the manager's packagers.

        :param obj:      The object to pack as an artifact.
        :param log_hint: The log hint to use.

        :return: The packaged artifact or result. None is returned if there was a problem while packing the object.
        """
        # Get the artifact type (if user didn't pass any, the packager will use its configured default):
        artifact_type = log_hint.pop("artifact_type", None)

        # Get a packager:
        object_type = type(obj)
        packager = self._get_packager(
            object_type=object_type, artifact_type=artifact_type
        )

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

    def unpack(self, data_item: DataItem, type_hint: typing.Type) -> typing.Any:
        """
        Unpack an object using one of the manager's packagers.

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
                packager_name = packaging_instructions.pop(
                    self._InstructionsNotesKey.PACKAGER_NAME, None
                )
                object_type = packaging_instructions.pop(
                    self._InstructionsNotesKey.OBJECT_TYPE, None
                )
                if object_type is not None:
                    # Get the actual type from the stored string:
                    object_type = self._get_type_from_name(type_name=object_type)
                artifact_type = packaging_instructions.pop(
                    self._InstructionsNotesKey.ARTIFACT_TYPE, None
                )
                instructions = packaging_instructions.pop(
                    self._InstructionsNotesKey.INSTRUCTIONS, {}
                )

        # If both original packaged object type and user's type hint available, validate they are equal:
        if object_type is not None and type_hint is not None:
            matching_object_and_type_hint = False
            hinted_types = [type_hint]
            while len(hinted_types) > 0:
                if object_type in hinted_types:
                    matching_object_and_type_hint = True
                    break
                hinted_types = self._reduce_type_hints(type_hints=hinted_types)
            if not matching_object_and_type_hint:
                logger.warn(
                    f"{artifact_key} was originally packaged as type '{object_type}' but the type hint given to unpack "
                    f"it as is '{type_hint}'. It is recommended to not parse an object from type to type using the "
                    f"packagers as unknown behavior might happen."
                )

        # Get the packager:
        packager = self._get_packager(
            object_type=type_hint or object_type,
            artifact_type=artifact_type,
            packager_name=packager_name,
        )

        # If the packager name is available (noted by manager), validate the original packager who packaged the object
        # was found:
        if packager_name is not None and packager.__name__ != packager_name:
            logger.warn(
                f"{artifact_key} was originally packaged by a packager of type '{packager_name}' but it was not "
                f"found. Custom packagers should be added to the project running the function. MLRun will try using "
                f"the packager '{packager.__name__}' instead. It is recommended to use the same packager for packing "
                f"and unpacking to reduce unexpected behaviours."
            )

        # Unpack:
        return packager.unpack(
            data_item=data_item,
            artifact_type=artifact_type,
            instructions=instructions,
        )

    def link_packages(
        self,
        additional_artifacts: typing.List[Artifact],
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
        self, module_name: str, packagers: typing.List[str]
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
            logger.warn(
                f"Packager '{packager_name}' was not found. Looking for an alternative."
            )

        # Try to get a packager to match the given types:
        found_packager = self._get_packager_by_type(
            object_type=object_type, artifact_type=artifact_type
        )
        if found_packager:
            return found_packager

        # Return the default as no packager was found:
        logger.warn(
            f"No packager was found for the combination of 'object_type={self._get_type_name(typ=object_type)}' and "
            f"'artifact_type={artifact_type}'. Using the default packager."
        )
        return self._default_packager

    def _get_packager_by_name(self, name: str) -> typing.Union[Packager, None]:
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
            possible_type_hints = self._reduce_type_hints(
                type_hints=possible_type_hints
            )

        # No packager was found:
        return None

    @staticmethod
    def _look_for_extra_data(
        key: str,
        artifacts: typing.List[Artifact],
        results: dict,
    ) -> typing.Union[Artifact, str, int, float, None]:
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
    def _split_module_path(module_path: str) -> typing.Tuple[str, str]:
        """
        Split a module path to the module name and the class name. Notice inner classes won't be supported.

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
    def _get_type_from_name(
        type_name: str, predicate: typing.Callable[[typing.Any], bool] = None
    ) -> typing.Union[typing.Type, typing.List[typing.Type]]:
        """
        Get the type object out of the given module path. The module must be a full module path (for example:
        "pandas.DataFrame" and not "DataFrame") otherwise it assumes to be from the local run module - __main__.
        If the type name ends with a "*", all types from the module will be returned in a list.

        :param type_name: The type full name (module path) string.
        :param predicate: A filter to use on the collected members of module (only used for type name of "*").

        :return: The represented type as imported from its module.
        """
        module_name, class_name = PackagersManager._split_module_path(
            module_path=type_name
        )
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
    def _reduce_type_hint(type_hint: typing.Type) -> typing.List[typing.Type]:
        """
        Reduce a type hint. If the type hint is a `typing` module, it will be reduced to its original hinted types. For
        example: `typing.Union[int, float, typing.List[int]]` will return `[int, float, List[int]]` and
        `typing.List[int]` will return `[list]`. Regular type hints - Python object types cannot be reduced as they are
        already a core type.

        If a type hint cannot be reduced, an empty list will be returned.

        :param type_hint: The type hint to reduce.

        :return: The reduced type hint as list of hinted types or an empty list if the type hint could not be reduced.
        """
        # TODO: Remove when we'll no longer support Python 3.7:
        import sys

        if sys.version_info[1] < 8:
            return []

        # If it's not a typing type (meaning it's an actual object type) then we can't reduce it further:
        if not PackagersManager._is_typing_type(type_hint=type_hint):
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

    @staticmethod
    def _reduce_type_hints(
        type_hints: typing.Union[typing.Type, typing.List[typing.Type]],
    ) -> typing.Set[typing.Type]:
        """
        Reduce a type hint (or a list of type hints) using the `_reduce_type_hint` function.

        :param type_hints: The type hint to reduce.

        :return: The reduced type hints set or an empty set if the type hint could not be reduced.
        """
        # Wrap in a list if provided a single type hint:
        if not isinstance(type_hints, list):
            type_hints = [type_hints]

        # Iterate over the type hints and reduce each one:
        return set(
            list(
                itertools.chain(
                    *[
                        PackagersManager._reduce_type_hint(type_hint=type_hint)
                        for type_hint in type_hints
                    ]
                )
            )
        )
