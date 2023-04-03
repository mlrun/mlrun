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
import itertools
import sys
import typing


class ArtifactType:
    """
    Possible artifact types to pack objects as and log using a `mlrun.Packager`.
    """

    DATASET = "dataset"
    DIRECTORY = "directory"
    FILE = "file"
    OBJECT = "object"
    PLOT = "plot"
    RESULT = "result"


class LogHintKey:
    """
    Known keys for a log hint to have.
    """

    KEY = "key"
    ARTIFACT_TYPE = "artifact_type"
    EXTRA_DATA = "extra_data"
    METRICS = "metrics"


class TypingUtils:
    """
    Static class for utilities functions to process typing module objects - type hints.
    """

    @staticmethod
    def is_typing_type(type_hint: typing.Type) -> bool:
        """
        Check whether a given type is from the `typing` module.

        :param type_hint: The type to check.

        :return: True if the type is from `typing` and False otherwise.
        """
        return hasattr(type_hint, "___module__") and type_hint.__module__ == "typing"

    @staticmethod
    def reduce_type_hint(
        type_hint: typing.Union[typing.Type, typing.List[typing.Type]],
    ) -> typing.Set[typing.Type]:
        """
        Reduce a type hint (or a list of type hints) using the `_reduce_type_hint` function.

        :param type_hint: The type hint to reduce.

        :return: The reduced type hints set or an empty set if the type hint could not be reduced.
        """
        # Wrap in a list if provided a single type hint:
        type_hints = [type_hint] if not isinstance(type_hint, list) else type_hint

        # Iterate over the type hints and reduce each one:
        return set(
            list(
                itertools.chain(
                    *[
                        TypingUtils._reduce_type_hint(type_hint=type_hint)
                        for type_hint in type_hints
                    ]
                )
            )
        )

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
        if sys.version_info[1] < 8:
            return []

        # If it's not a typing type (meaning it's an actual object type) then we can't reduce it further:
        if not TypingUtils.is_typing_type(type_hint=type_hint):
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
        if not TypingUtils.is_typing_type(type_hint=origin):
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
