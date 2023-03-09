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
import typing
import itertools

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem, is_store_uri, store_manager

from .packager import Packager


class _ManagerLabels:
    MLRUN_VERSION = "mlrun_version"
    PACKAGER = "packager"
    OBJECT_TYPE = "object_type"
    ARTIFACT_TYPE = "artifact_type"


class _InstructionsKeys:
    MANAGER_INSTRUCTIONS = "manager_instructions"
    PACKAGER_INSTRUCTIONS = "packager_instructions"


class PackagersManager:
    def __init__(self, default_packager: Packager = None):
        self._default_packager = default_packager or Packager
        self._packagers: typing.List[Packager] = None
        self._collect_packagers()

    def pack(self, obj: typing.Any, log_hint: typing.Dict[str, str]) -> Artifact:
        # Get the artifact type (if user didn't pass any, the packager will use its configured default):
        artifact_type = log_hint.pop("artifact_type", None)

        # Choose the first packager fitting to the object (packagers priority is by the order they are being added,
        # last added -> highest priority):
        packager = self._get_packager_by_type(
            object_type=type(obj), artifact_type=artifact_type
        )

        # Use the packager to pack the object:
        artifact, instructions = packager.pack(
            obj=obj, artifact_type=artifact_type, instructions=log_hint
        )

        # Prepare the manager's labels:
        instructions = {}

        # Set the instructions in the artifact's spec:
        if instructions:
            artifact.spec.package_instructions = instructions

        return artifact

    def unpack(self, data_item: DataItem, type_hint: typing.Type) -> typing.Any:
        # Check if the data item is based on an artifact (it may be a simple path/url data item):
        package_instructions: dict = None
        if data_item.artifact_url and is_store_uri(url=data_item.artifact_url):
            # Get the instructions:
            artifact, _ = store_manager.get_store_artifact(url=data_item.artifact_url)
            package_instructions = artifact.spec.package_instructions

    def _collect_packagers(self):
        pass

    def _get_packager_by_type(
        self, object_type: typing.Type, artifact_type: str
    ) -> Packager:
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
                        self._reduce_typing_type(t=type_hint)
                        for type_hint in possible_type_hints
                    ]
                )
            )

        return found_packager if found_packager is not None else self._default_packager

    def _get_packager_by_name(self, name: str) -> typing.Union[Packager, None]:
        for packager in self._packagers:
            if packager.__name__ == name:
                return packager
        return None

    @staticmethod
    def _is_typing_type(t: typing.Type):
        return hasattr(t, "___module__") and t.__module__ == "typing"

    @staticmethod
    def _reduce_typing_type(t: typing.Type) -> typing.List[typing.Type]:
        # TODO: Only works for python >= 3.8, need to re do for 3.7.
        # If it's a type var, take its constraints (e.g. A = TypeVar("A", int, str) meaning an object of type A should
        # be an integer ot a string). If it doesn't have constraints, return an empty list:
        if isinstance(t, typing.TypeVar):
            if len(t.__constraints__) == 0:
                return []
            return list(t.__constraints__)

        # If it's a forward reference, that means the user could not import the class to type hint it (so we can't
        # either):
        if isinstance(t, typing.ForwardRef):
            return []

        # Get the origin of the typing type. An origin is the subscripted typing type (origin of Union[str, int] is
        # Union). The origin can be one of Callable, Tuple, Union, Literal, Final, ClassVar, Annotated or the actual
        # type alias (e.g. origin of List[int] is list):
        origin = typing.get_origin(t)

        # If the typing type has no origin (e.g. None is returned), we cannot reduce it, so we return an empty list:
        if origin is None:
            return []

        # If the origin is a type of one of builtin, contextlib or collections (for example: List's origin is list)
        # then we can be sure there is nothing to reduce as it's a regular type:
        if not PackagersManager._is_typing_type(t=origin):
            return [origin]

        # Get the type's subscriptions - arguments, in order to reduce it to them (we know for sure there are arguments,
        # otherwise origin would have been None):
        args = typing.get_args(t)

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
