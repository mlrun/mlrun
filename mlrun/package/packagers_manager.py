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

import importlib
import inspect
import os
import shutil
import traceback
from typing import Any

import mlrun.errors
from mlrun.artifacts import Artifact
from mlrun.artifacts.base import verify_target_path
from mlrun.datastore import DataItem, get_store_resource, store_manager
from mlrun.package.errors import (
    MLRunPackageBundlingError,
    MLRunPackageCollectionError,
    MLRunPackagePackingError,
    MLRunPackageUnbundlingError,
    MLRunPackageUnpackingError,
)
from mlrun.package.log_hint import LogHint
from mlrun.package.packager import Packager
from mlrun.package.packagers.default_packager import DefaultPackager
from mlrun.package.utils import TypeHintUtils
from mlrun.utils import logger


class PackagersManager:
    """
    A packager manager holds the project's packagers and sends them objects to pack, and data items to unpack.

    It prepares the instructions / log hint configurations and then looks for the first packager that fits the task.
    """

    def __init__(self, default_packager: type[Packager] | None = None):
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

        # Set an artifacts list (holding tuples of packed artifact and the `context.log_artifact` kwargs to use for it)
        # and results dictionary to collect all packed objects (will be used later to write extra data if noted by the
        # user using the log hint key "extra_data")
        self._artifacts: list[tuple[Artifact, dict]] = []
        self._results = {}

        # Temporary holder for bundle structures results to update the store paths before logging them as results:
        self._bundles = {}

    @property
    def artifacts(self) -> list[tuple[Artifact, dict]]:
        """
        Get the artifacts that were packed by the manager.

        :return: A list of tuples with the artifacts and their `context.log_artifact` method kwargs.
        """
        return self._artifacts

    @property
    def results(self) -> dict:
        """
        Get the results that were packed by the manager.

        :return: A results dictionary.
        """
        return self._results

    def get_bundles_results(self, logged_outputs: dict) -> dict:
        """
        Get the bundles results with updated store paths according to the logged outputs.

        :param logged_outputs: The logged outputs dictionary from the MLRun context.

        :return: A results dictionary with the bundles updated store paths.
        """
        updated_bundles = {}
        for key, bundle_structure in self._bundles.items():
            updated_bundles[key] = self._update_bundle_results(
                bundle_structure=bundle_structure,
                logged_outputs=logged_outputs,
            )
        return updated_bundles

    def collect_packagers(
        self, packagers: list[type[Packager] | str], default_priority: int = 5
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
        self,
        obj: Any,
        log_hint: LogHint,
    ) -> Artifact | dict | None | list[Artifact | dict | None]:
        """
        Pack an object using one of the manager's packagers.

        A `list` unpacking syntax ("*") in the log hint key unbundle the given object to pack each of its item
        separately. If a number is added before the asterisk ("X*"), it represent the level of unbundling.

        For example, if the object is a nested list `[[1, 2], [3, 4]]` and the log hint key is "1*", the object will be
        unbundled once to `[1, 2]` and `[3, 4]`, and each of these items will be packed separately. If the log hint key
        is "2*", the object will be unbundled twice to `1`, `2`, `3`, and `4`, and each of these items will be packed
        separately.

        By default, an asterisk without a number will unbundle all the levels possible.

        :param obj:             The object to pack as an artifact.
        :param log_hint:        The log hint to use.

        :return: The packaged artifact or result. None is returned if there was a problem while packing the object. If
                 unbundling is performed, a list of all the unbundled packaged objects is returned.

        :raise MLRunInvalidArgumentError:   If the key in the log hint instructs do not follow the unbundling syntax.
        :raise MLRunPackagePackingError:    If there was an error during the packing.
        :raise MLRunPackageUnbundlingError: If there was an error during the unbundling.
        """
        try:
            if log_hint.itemized:
                # Multiple objects are required to be packaged as a bundle:
                package, bundle_result = self._pack_bundle(
                    obj=obj,
                    log_hint=log_hint,
                    unbundle_level=log_hint.itemized,
                )
                # Check if the bundle result is a dict or list - meaning it was unbundled successfully so we collect
                # the bundle structure:
                if isinstance(bundle_result, dict | list):
                    self._bundles[log_hint.key] = bundle_result
            else:
                # A single object is required to be packaged:
                package = self._pack(
                    obj=obj, log_hint=log_hint.copy()
                )  # Log hint is copied to preserve key for error.
        except Exception as exception:
            raise MLRunPackagePackingError(
                f"An exception was raised during the packing of '{log_hint.key}': {exception}"
            ) from exception

        return package

    def unpack(self, data_item: DataItem | dict | list, type_hint: type) -> Any:
        """
        Unpack an object using one of the manager's packagers. The data item can be unpacked in two ways:

        * As a package: If the data item contains a package and the type hint provided is equal to the object
          type noted in the package. Or, if it's a package and a type hint was not provided.
        * As a data item: If the data item is not a package or the type hint provided is not equal to the one noted in
          the package.

        If the `data_item` received is a collection (a `dict` or `list`), each item in the collection will be unpacked
        according to the type hint provided.

        If the type hint is a `mlrun.DataItem` then it won't be unpacked.

        Notice: It is not recommended to use a different packager than the one that originally packed the object to
        unpack it. A warning displays in that case.

        :param data_item: The data item holding the package. Can be a collection of data items (the type hint must
                          match a packager that supports initializing a collection).
        :param type_hint: The type hint to parse the data item as.

        :return: The unpacked object parsed as type hinted.
        """
        # Check if a type hint was provided - if not, continue only if user set auto unpacking:
        if (
            type_hint is inspect.Parameter.empty
            and not mlrun.mlconf.packagers.auto_unpack_inputs
        ):
            return data_item

        # Check if `DataItem` is hinted - meaning the user can expect a data item and do not want to unpack it:
        if TypeHintUtils.is_matching(object_type=DataItem, type_hint=type_hint):
            return data_item

        # Check if the data item is a collection (a `dict` or `list`):
        if isinstance(data_item, dict | list):
            # Bundle it:
            try:
                return self._bundle(collection=data_item, type_hint=type_hint)
            except Exception as exception:
                raise MLRunPackageBundlingError(
                    f"An exception was raised during the bundling of '{type(data_item)}': {exception}"
                ) from exception

        # Set variables to hold the manager notes and packager instructions:
        artifact_key = None
        packaging_instructions = None

        # Try to get the notes and instructions (can be found only in artifacts but data item may be a simple path/url):
        if data_item.get_artifact_type():
            # Get the artifact object in the data item:
            artifact, _ = store_manager.get_store_artifact(url=data_item.artifact_url)
            verify_target_path(artifact)
            # Get the key from the artifact's metadata and instructions from the artifact's spec:
            artifact_key = artifact.metadata.key
            packaging_instructions = artifact.spec.unpackaging_instructions

        # Unpack:
        try:
            if packaging_instructions:
                # The data item is a package (if the object type is equal or part of the type hint (part of means in
                # case of a `typing.Union` for example) it will be unpacked as a package, otherwise as a data item):
                return self._unpack_package(
                    data_item=data_item,
                    artifact_key=artifact_key,
                    packaging_instructions=packaging_instructions,
                    type_hint=type_hint,
                )
            # The data item is not a package (will continue only if a type hint was provided):
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
    ) -> set[Artifact]:
        """
        Link packages to each other according to the provided extra data and metrics spec keys. A future link is
        marked with ellipses (...). If no link is found, None is used and a warning is printed.

        :param additional_artifact_uris: Additional artifact URIs to link (should come from an `mlrun.MLClientCtx`).
        :param additional_results:       Additional results to link (should come from an `mlrun.MLClientCtx`).

        :return: A set of the additional artifacts that require updates post linking (the packagers artifacts were not
                 logged yet).
        """
        # Join the manager's results with the additional ones to look for a link in all of them:
        all_results = {**additional_results, **self.results}

        # Convert additional artifact URIs to artifacts:
        additional_artifacts = []
        for key, uri in (additional_artifact_uris or {}).items():
            try:
                artifact = get_store_resource(uri)
                additional_artifacts.append(artifact)
            except mlrun.errors.MLRunNotFoundError as exc:
                logger.warn(
                    f"Could not get artifact {key=} from URI when linking packages",
                    exc=mlrun.errors.err_to_str(exc),
                )

        # Join all artifacts (packager artifacts + context artifacts):
        all_artifacts = [
            artifact for (artifact, _) in self.artifacts
        ] + additional_artifacts

        # Prepare a set for artifacts that require updates post linking:
        artifacts_to_update = set()

        # Go over all artifacts and link:
        for artifact in all_artifacts:
            # Go over the extra data keys:
            not_found_keys = []
            for key in artifact.spec.extra_data:
                # Future link is marked with ellipses (...):
                if artifact.spec.extra_data[key] is ...:
                    # Collect it to post update if it's a context artifact:
                    if artifact in additional_artifacts:
                        artifacts_to_update.add(artifact)
                    # Look for an artifact or result with this key to link it:
                    extra_data = self._look_for_extra_data(
                        key=key,
                        artifacts=all_artifacts,
                        results=all_results,
                    )
                    # Print a warning if a link is missing:
                    if extra_data is None:
                        logger.warn(
                            f"Could not find {key} to link as extra data for {artifact.key}."
                        )
                        not_found_keys.append(key)
                        continue
                    # Link it:
                    artifact.spec.extra_data[key] = extra_data
            # Clean the not found keys from the spec to avoid confusion:
            for key in not_found_keys:
                artifact.spec.extra_data.pop(key)
            # Go over the metrics keys if available (`ModelArtifactSpec` has a metrics property that may be waiting for
            # values from logged results):
            not_found_keys.clear()
            if hasattr(artifact.spec, "metrics"):
                for key in artifact.spec.metrics:
                    # Future link is marked with ellipses (...):
                    if artifact.spec.metrics[key] is ...:
                        # Link it (None will be used in case it was not found):
                        metric = all_results.get(key, None)
                        if metric is None:
                            logger.warn(
                                f"Could not find {key} to link as a metric for {artifact.key}."
                            )
                            not_found_keys.append(key)
                            continue
                        artifact.spec.metrics[key] = metric
                # Clean the not found keys from the spec to avoid confusion:
                for key in not_found_keys:
                    artifact.spec.metrics.pop(key)

        return artifacts_to_update

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

    def _update_bundle_results(
        self, bundle_structure: Any, logged_outputs: dict
    ) -> Any:
        """
        Update the bundle results according to the logged outputs. This method goes over the bundle structure
        recursively and look for the log hint keys to update them with the logged outputs paths.

        :param bundle_structure: The bundle structure to update.
        :param logged_outputs:   The logged outputs dictionary from the MLRun context.

        :return: The updated bundle structure.
        """
        # Dict case:
        if isinstance(bundle_structure, dict):
            return {
                bundle_key: (
                    logged_outputs[package_key]
                    if not isinstance(package_key, list | dict)
                    else self._update_bundle_results(
                        bundle_structure=bundle_structure[bundle_key],
                        logged_outputs=logged_outputs,
                    )
                )
                for bundle_key, package_key in bundle_structure.items()
            }

        # List case:
        return [
            logged_outputs[package_key]
            if not isinstance(package_key, list | dict)
            else self._update_bundle_results(
                bundle_structure=bundle_structure[index],
                logged_outputs=logged_outputs,
            )
            for index, package_key in enumerate(bundle_structure)
        ]

    def _get_packagers_with_default_packager(self) -> list[Packager]:
        """
        Get the full list of packagers - the collected packagers and the default packager (located at last place in the
        list - the lowest priority).

        :return: A list of the manager's packagers with the default packager.
        """
        return [*self._packagers, self._default_packager]

    def _get_packager_by_name(self, name: str) -> Packager | None:
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
        artifact_type: str | None = None,
        configurations: dict | None = None,
    ) -> Packager | None:
        """
        Look for a packager that can pack the provided object as the provided artifact type.

        If a packager was not found None will be returned.

        :param obj:            The object to pack.
        :param artifact_type:  The artifact type the packager to get should pack / unpack as.
        :param configurations: The log hint configurations passed by the user.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager for the combination of object and artifact type:
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
        artifact_type: str | None = None,
    ) -> Packager | None:
        """
        Look for a packager that can unpack the data item of the given type hint as the provided artifact type.

        If a packager was not found None will be returned.

        :param data_item:     The data item to unpack.
        :param type_hint:     The type hint the packager to get should handle.
        :param artifact_type: The artifact type the packager to get should pack / unpack as.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager for the combination of object type and artifact type:
        for packager in self._packagers:
            if packager.is_unpackable(
                data_item=data_item, type_hint=type_hint, artifact_type=artifact_type
            ):
                return packager

        # No packager was found:
        return None

    def _get_packager_for_bundling(
        self,
        bundle_hint: type,
        collection_type: type[dict] | type[list] | None = None,
    ) -> Packager | None:
        """
        Look for a packager that can bundle the given type hint on the provided collection type (list or dict).

        If a packager was not found None will be returned.

        :param bundle_hint:       The bundle type hint the packager to get should handle.
        :param collection_type: The collection type the packager to get should construct from.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager for the combination of type hint and collection type:
        for packager in self._packagers:
            if packager.can_bundle(
                bundle_hint=bundle_hint, collection_type=collection_type
            ):
                return packager

        # No packager was found:
        return None

    def _get_packager_for_unbundling(
        self,
        bundled_object: Any,
    ) -> Packager | None:
        """
        Look for a packager that can unbundle the given object into a collection (list or dict).

        If a packager was not found None will be returned.

        :param bundled_object:  The bundle object the packager to get should handle.

        :return: The found packager or None if it wasn't found.
        """
        # Look for a packager for the combination of type hint and collection type:
        for packager in self._packagers:
            if packager.can_unbundle(bundled_object=bundled_object):
                return packager

        # No packager was found:
        return None

    def _pack_bundle(
        self, obj: object, log_hint: LogHint, unbundle_level: bool | int
    ) -> tuple[list[Artifact | dict | None], dict | str]:
        """
        Pack a bundle of objects using one of the manager's packagers.

        Note: ``bundle_structure`` is a dict or list mirroring the unbundled object's structure with package keys as
        leaves when actual unbundling occurred. When the object could not be unbundled, it is packed as a single object
        and ``bundle_structure`` is a string (the log hint key). This string serves as a leaf value in recursive calls
        but should **not** be stored in ``_bundles`` at the top level since no actual unbundling occurred — the packed
        result / artifact is already collected normally in ``_results`` or ``_artifacts``.

        :param obj:            The objects bundle to pack as artifacts.
        :param log_hint:       The log hint to use.
        :param unbundle_level: Mention the level of unbundling to perform. If provided, the method will unbundle the
                               object only if the level is > 0, and will decrease the level by 1 for every unbundling.

        :return: A list of all packaged artifacts or results along the bundle structure as result. None is returned if
                 there was a problem while packing the objects.

        :raise MLRunPackagePackingError:    If there was an error during the packing.
        :raise MLRunPackageUnbundlingError: If there was an error during the unbundling.
        """
        # Check the object can be unbundled (we don't want to fail on non-unbundle-able object, as it is very common
        # to return a list[object] | object from a function. In this case, the user may run with a log hint that would
        # start with * but only a single object will return - we don't want to fail on this, but rather pack it as a
        # single object):
        unbundled_object = None
        if unbundle_level:
            try:
                unbundled_object = self._unbundle(bundled_object=obj)
            except MLRunPackageUnbundlingError as unbundling_error:
                if "No packager was found to unbundle the object" not in str(
                    unbundling_error
                ):
                    raise unbundling_error
                logger.debug(
                    f"Unbundle level was not reached for '{log_hint.key}', but it cannot be unbundled (there is no "
                    f"packager that can unbundle it) so we continue to pack it as a single object."
                )

        # If the object cannot be unbundled, pack it as a single object:
        if unbundled_object is None:
            return [self._pack(obj=obj, log_hint=log_hint)], log_hint.key

        # Unbundling was performed, create a log hint for each of the unbundled items:
        if isinstance(unbundled_object, dict):
            objects_to_pack = {
                f"{log_hint.key}_{dict_key}": dict_obj
                for dict_key, dict_obj in unbundled_object.items()
            }
            bundle_structure = {
                dict_key: package_key
                for dict_key, package_key in zip(
                    unbundled_object.keys(), objects_to_pack.keys()
                )
            }
        else:
            objects_to_pack = {
                f"{log_hint.key}_{i}": obj_i for i, obj_i in enumerate(unbundled_object)
            }
            bundle_structure = list(objects_to_pack.keys())

        # Go over the collected keys and objects and pack them (with decreased unbundle level):
        unbundle_level = (
            unbundle_level if isinstance(unbundle_level, bool) else unbundle_level - 1
        )
        packages = []
        for (key, per_key_obj), i in zip(
            objects_to_pack.items(),
            unbundled_object.keys()
            if isinstance(unbundled_object, dict)
            else range(len(unbundled_object)),
        ):
            # Edit the key in the log hint:
            per_key_log_hint = log_hint.copy()
            per_key_log_hint.key = key
            # Pack and collect the package:
            try:
                currently_packaged, bundle_structure[i] = self._pack_bundle(
                    obj=per_key_obj,
                    log_hint=per_key_log_hint,
                    unbundle_level=unbundle_level,
                )
                if isinstance(currently_packaged, list):
                    packages.extend(currently_packaged)
                else:
                    packages.append(currently_packaged)
            except Exception as exception:
                raise MLRunPackagePackingError(
                    f"An exception was raised during the packing of '{per_key_log_hint.key}': {exception}"
                ) from exception

        return packages, bundle_structure

    def _pack(self, obj: Any, log_hint: LogHint) -> Artifact | dict | None:
        """
        Pack an object using one of the manager's packagers.

        :param obj:      The object to pack as an artifact.
        :param log_hint: The log hint to use.

        :return: The packaged artifact or result. None is returned if there was a problem while packing the object.
        """
        # Get a packager:
        packager = self._get_packager_for_packing(
            obj=obj,
            artifact_type=log_hint.artifact_type,
            configurations=log_hint.packing_kwargs,
        )
        if packager is None:
            if self._default_packager.is_packable(
                obj=obj,
                artifact_type=log_hint.artifact_type,
                configurations=log_hint.packing_kwargs,
            ):
                logger.info(
                    f"Using the default packager to pack the object '{log_hint.key}'"
                )
                packager = self._default_packager
            else:
                raise MLRunPackagePackingError(
                    f"No packager was found for the combination of "
                    f"'object_type={self._get_type_name(typ=type(obj))}' and 'artifact_type={log_hint.artifact_type}'."
                )

        # Use the packager to pack the object:
        packed_object = packager.pack(
            obj=obj,
            key=log_hint.key,
            artifact_type=log_hint.artifact_type,
            configurations=log_hint.packing_kwargs,
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
                log_hint.artifact_type
                if log_hint.artifact_type
                else packager.get_default_packing_artifact_type(obj=obj)
            ),
            self._InstructionsNotesKey.INSTRUCTIONS: instructions,
        }

        # Set the instructions in the artifact's spec:
        artifact.spec.unpackaging_instructions = unpackaging_instructions

        # Add extra data to the artifact's spec if noted in the log hint:
        if log_hint.tag:
            artifact.tag = log_hint.tag
        if log_hint.labels:
            artifact.labels = log_hint.labels
        if log_hint.extra_data:
            artifact.extra_data = log_hint.extra_data
        if log_hint.metrics:
            if not hasattr(artifact.spec, "metrics"):
                logger.warn(
                    f"Metrics were provided in the log hint for '{log_hint.key}' but the artifact type "
                    f"'{log_hint.artifact_type}' does not support metrics, so they were ignored. Make sure to use an "
                    f"artifact type that supports metrics (for example, 'model') if you wish to log metrics. You can"
                    f"also add them as extra data if needed."
                )
            else:
                artifact.spec.metrics = log_hint.metrics

        # Add logging kwargs from the log hint:
        logging_kwargs = {}
        if log_hint.artifact_path:
            logging_kwargs["artifact_path"] = log_hint.artifact_path

        # Collect the artifact and return:
        self._artifacts.append((artifact, logging_kwargs))
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
        elif type_hint is inspect.Parameter.empty:
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

        If the type hint is empty (meaning it was not provided), a warning is printed and the data item is returned as
        is.

        :param data_item: The data item to unpack.
        :param type_hint: The type hint to unpack it to.

        :return: The unpacked object if a type hint was provided or the data item itself if type hint was empty.

        :raise MLRunPackageUnpackingError: If there is no packager that supports the provided type hint.
        """
        # Check if a type hint is available:
        if type_hint is inspect.Parameter.empty:
            logger.warn(
                f"Although 'auto_unpack_inputs' is set, the input of '{data_item.key}' could not be "
                f"unpacked as it was not originally packaged. To unpack it, please provide a type hint in the handler "
                f"code or the inputs key in the MLRun's function `run` method call."
            )
            return data_item

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

    def _bundle(self, collection: dict | list, type_hint: type) -> Any:
        """
        Bundle a collection of data items according to the type hint provided.

        :param collection: The collection of data items to unpack.
        :param type_hint:  The user's type hint.

        :return: The bundled collection.

        :raise MLRunPackageBundlingError: If there is no packager to bundle the collection type.
        :raise MLRunPackageUnpackingError: If there is no packager to initialize the collection type.
        """
        # Prepare a set to hold possible type hints to try to bundle as:
        possible_type_hints = set()

        # Check if there is no type hint (auto unpacking must be on - it was verified already in `unpack`):
        if type_hint is inspect.Parameter.empty:
            possible_type_hints.add(type(collection))
        else:
            # Reduce pure hints (like `typing.Any`, `typing.Union`, etc.) to possible real types:
            possible_type_hints_test = {type_hint}
            while possible_type_hints_test:
                for hint in possible_type_hints_test:
                    if not TypeHintUtils.is_pure_hint(type_hint=hint):
                        possible_type_hints.add(hint)
                # Remove the found types from the test set and continue reducing:
                possible_type_hints_test = (
                    possible_type_hints_test - possible_type_hints
                )
                possible_type_hints_test = TypeHintUtils.reduce_type_hint(
                    type_hint=possible_type_hints_test
                )
            if len(possible_type_hints) == 0:
                # No real type was found, set the bundle type hint to the collection type:
                possible_type_hints.add(type(collection))

        # Go over the hints and try to bundle as one of them:
        found_packagers = []
        for hint in possible_type_hints:
            # Get the origin (bundle object type) and args (bundle items type) of the type hint:
            bundle_type_hint, items_type_hint = TypeHintUtils.deconstruct_type_hint(
                type_hint=hint
            )
            if items_type_hint is not inspect.Parameter.empty:
                # TODO: We are going to take the last `Generic` variable registered. Usually this is the item type in
                #       collections like `list`, `set` (which has only one variable: list[V]) and `dict` (which has two,
                #       one for the keys and the last one is the value: `dict[_KT, _VT]`).
                #       To improve this, we can try to go over some of the popular `Generic` variable naming conventions
                #       like `T`, `V`, `VT`, etc. to identify the item type better:
                #       `[p.__name__ for p in bundle_type.__parameters__]`.
                #       Another option is to try each of them until one works in unpacking.
                items_type_hint = (
                    items_type_hint[-1]
                    if isinstance(items_type_hint, tuple)
                    else items_type_hint
                )
            # Get a packager that can bundle as the given type hint on the given collection type:
            packager = self._get_packager_for_bundling(
                bundle_hint=bundle_type_hint, collection_type=type(collection)
            )
            if packager is None:
                # No packager was found that supports this hinted type:
                continue
            # Unpack items in the collection according to the items type hint:
            try:
                if isinstance(collection, dict):
                    unpacked_collection = {
                        key: self.unpack(data_item=data_item, type_hint=items_type_hint)
                        for key, data_item in collection.items()
                    }
                else:  # It's a list.
                    unpacked_collection = [
                        self.unpack(data_item=data_item, type_hint=items_type_hint)
                        for data_item in collection
                    ]
            except Exception:
                # Could not bundle as the type hint, collect the exception and go to the next one:
                found_packagers.append((packager, traceback.format_exc()))
                continue
            # Bundle:
            try:
                return packager.bundle(collection=unpacked_collection)
            except Exception:
                # Could not bundle as the type hint, collect the exception and go to the next one:
                found_packagers.append((packager, traceback.format_exc()))
                continue

        # The method did not return until this point, raise an error:
        if found_packagers:
            raise MLRunPackageBundlingError(
                f"Could not bundle the input with the hinted type '{type_hint}'. The following packagers were tried to "
                f"be used to bundle it but raised the exceptions joined:\n\n"
                + "\n".join(
                    [
                        f"Found packager: '{packager}'\nException: {exception}\n"
                        for packager, exception in found_packagers
                    ]
                )
            )
        raise MLRunPackageBundlingError(
            f"No packager was found that can bundle a '{type(collection).__name__}' into '{type_hint}'."
        )

    def _unbundle(self, bundled_object: Any) -> dict | list:
        """
        Unbundle a bundled object into a collection of data items.

        :param bundled_object: The bundled object to unbundle.

        :return: The unbundled collection of data items.

        :raise MLRunPackageUnbundlingError: If there is no packager to unbundle the given object.
        """
        # Get a packager that can unbundle the given object:
        packager = self._get_packager_for_unbundling(bundled_object=bundled_object)
        if packager is None:
            raise MLRunPackageUnbundlingError(
                f"No packager was found to unbundle the object of type '{type(bundled_object)}'."
            )

        # Unbundle:
        try:
            return packager.unbundle(bundled_object=bundled_object)
        except Exception as exception:
            raise MLRunPackageUnbundlingError(
                f"An exception was raised during the unbundling of an object of type "
                f"'{type(bundled_object)}': {exception}"
            ) from exception

    @staticmethod
    def _look_for_extra_data(
        key: str,
        artifacts: list[Artifact],
        results: dict,
    ) -> Artifact | str | int | float | None:
        """
        Look for an extra data item (artifact or result) by given key. If not found, None is returned.

        :param key:       Key to look for.
        :param artifacts: Artifacts to look in.
        :param results:   Results to look in.

        :return: The artifact or result with the same key or None if not found.
        """
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
