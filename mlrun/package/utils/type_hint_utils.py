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
import builtins
import importlib
import itertools
import re
import typing

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.utils import logger


class TypeHintUtils:
    """
    Static class for utilities functions to process type hints.
    """

    @staticmethod
    def is_typing_type(type_hint: type) -> bool:
        """
        Check whether a given type is a type hint from one of the modules `typing` and `types`. The function will return
        True for generic type aliases also, meaning Python 3.9's new hinting feature that includes hinting like
        `list[int]` instead of `typing.List[int]`.

        :param type_hint: The type to check.

        :return: True if the type hint from `typing` / `types` and False otherwise.
        """
        # A type hint should be one of the based typing classes, meaning it will have "typing" as its module. Some
        # typing classes are considered a type (like `TypeVar`) so we check their type as well. The only case "types"
        # will be a module is for generic aliases like `list[int]`.
        return (type_hint.__module__ == "typing") or (
            type(type_hint).__module__ in ["typing", "types"]
        )

    @staticmethod
    def parse_type_hint(type_hint: typing.Union[type, str]) -> type:
        """
        Parse a given type hint from string to its actual hinted type class object. The string must be one of the
        following:

        * Python builtin type - for example: `tuple`, `list`, `set`, `dict` and `bytearray`.
        * Full module import path. An alias (if `import pandas as pd is used`, the type hint cannot be `pd.DataFrame`)
          is not allowed.

        The type class on its own (like `DataFrame`) cannot be used as the scope of this function is not the same as the
        handler itself, hence modules and objects that were imported in the handler's scope are not available. This is
        the same reason import aliases cannot be used as well.

        If the provided type hint is not a string, it will simply be returned as is.

        :param type_hint: The type hint to parse.

        :return: The hinted type.

        :raise MLRunInvalidArgumentError: In case the type hint is not following the 2 options mentioned above.
        """
        if not isinstance(type_hint, str):
            return type_hint

        # Validate the type hint is a valid module path:
        if not bool(
            re.fullmatch(
                r"([a-zA-Z_][a-zA-Z0-9_]*\.)*[a-zA-Z_][a-zA-Z0-9_]*", type_hint
            )
        ):
            raise MLRunInvalidArgumentError(
                f"Invalid type hint. An input type hint must be a valid python class name or its module import path. "
                f"For example: 'list', 'pandas.DataFrame', 'numpy.ndarray', 'sklearn.linear_model.LinearRegression'. "
                f"Type hint given: '{type_hint}'."
            )

        # Look for a builtin type (rest of the builtin types like `int`, `str`, `float` should be treated as results,
        # hence not given as an input to an MLRun function, but as a parameter):
        builtin_types = {
            builtin_name: builtin_type
            for builtin_name, builtin_type in builtins.__dict__.items()
            if isinstance(builtin_type, type)
        }
        if type_hint in builtin_types:
            return builtin_types[type_hint]

        # If it's not a builtin, its should have a full module path, meaning at least one '.' to separate the module and
        # the class. If it doesn't, we will try to get the class from the main module:
        if "." not in type_hint:
            logger.warn(
                f"The type hint string given '{type_hint}' is not a `builtins` python type. MLRun will try to look for "
                f"it in the `__main__` module instead."
            )
            try:
                return TypeHintUtils.parse_type_hint(type_hint=f"__main__.{type_hint}")
            except MLRunInvalidArgumentError:
                raise MLRunInvalidArgumentError(
                    f"MLRun tried to get the type hint '{type_hint}' but it can't as it is not a valid builtin Python "
                    f"type (one of `list`, `dict`, `str`, `int`, etc.) nor a locally declared type (from the "
                    f"`__main__` module). Pay attention using only the type as string is not allowed as the handler's "
                    f"scope is different than MLRun's. To properly give a type hint as string, please specify the full "
                    f"module path without aliases. For example: do not use `DataFrame` or `pd.DataFrame`, use "
                    f"`pandas.DataFrame`."
                )

        # Import the module to receive the hinted type:
        try:
            # Get the module path and the type class (If we'll wish to support inner classes, the `rsplit` won't work):
            module_path, type_hint = type_hint.rsplit(".", 1)
            # Replace alias if needed (alias assumed to be imported already, hence we look in globals):
            # For example:
            # If in handler scope there was `import A.B.C as abc` and user gave a type hint "abc.Something" then:
            # `module_path[0]` will be equal to "abc". Then, because it is an alias, it will appear in the globals, so
            # we'll replace the alias with the full module name in order to import the module.
            module_path = module_path.split(".")
            if module_path[0] in globals():
                module_path[0] = globals()[module_path[0]].__name__
            module_path = ".".join(module_path)
            # Import the module:
            module = importlib.import_module(module_path)
            # Get the class type from the module:
            type_hint = getattr(module, type_hint)
        except ModuleNotFoundError as module_not_found_error:
            # May be raised from `importlib.import_module` in case the module does not exist.
            raise MLRunInvalidArgumentError(
                f"MLRun tried to get the type hint '{type_hint}' but the module '{module_path}' cannot be imported. "
                f"Keep in mind that using alias in the module path (meaning: import module as alias) is not allowed. "
                f"If the module path is correct, please make sure the module package is installed in the python "
                f"interpreter."
            ) from module_not_found_error
        except AttributeError as attribute_error:
            # May be raised from `getattr(module, type_hint)` in case the class type cannot be imported directly from
            # the imported module.
            raise MLRunInvalidArgumentError(
                f"MLRun tried to get the type hint '{type_hint}' from the module '{module.__name__}' but it seems it "
                f"doesn't exist. Make sure the class can be imported from the module with the exact module path you "
                f"passed. Notice inner classes (a class inside of a class) are not supported."
            ) from attribute_error

        return type_hint

    @staticmethod
    def is_matching(
        object_type: type,
        type_hint: typing.Union[type, set[type]],
        include_subclasses: bool = True,
        reduce_type_hint: bool = True,
    ) -> bool:
        """
        Check if the given object type match the given hint.

        :param object_type:        The object type to match with the type hint.
        :param type_hint:          The hint to match with. Can be given as a set resulted from a reduced hint.
        :param include_subclasses: Whether to mark a subclass as valid match. Default to True.
        :param reduce_type_hint:   Whether to reduce the type hint to match with its reduced hints.

        :return: True if the object type match the type hint and False otherwise.
        """
        # Wrap in a set if provided a single type hint:
        type_hint = {type_hint} if not isinstance(type_hint, set) else type_hint

        # Try to match the object type to one of the hints:
        while len(type_hint) > 0:
            for hint in type_hint:
                # Subclass check can be made only on actual object types (not typing module types):
                if (
                    not TypeHintUtils.is_typing_type(type_hint=object_type)
                    and not TypeHintUtils.is_typing_type(type_hint=hint)
                    and include_subclasses
                    and issubclass(object_type, hint)
                ):
                    return True
                if object_type == hint:
                    return True
            # See if needed to reduce, if not end on first iteration:
            if not reduce_type_hint:
                break
            type_hint = TypeHintUtils.reduce_type_hint(type_hint=type_hint)
        return False

    @staticmethod
    def reduce_type_hint(
        type_hint: typing.Union[type, set[type]],
    ) -> set[type]:
        """
        Reduce a type hint (or a set of type hints) using the `_reduce_type_hint` function.

        :param type_hint: The type hint to reduce.

        :return: The reduced type hints set or an empty set if the type hint could not be reduced.
        """
        # Wrap in a set if provided a single type hint:
        type_hints = {type_hint} if not isinstance(type_hint, set) else type_hint

        # Iterate over the type hints and reduce each one:
        return set(
            itertools.chain(
                *[
                    TypeHintUtils._reduce_type_hint(type_hint=type_hint)
                    for type_hint in type_hints
                ]
            )
        )

    @staticmethod
    def _reduce_type_hint(type_hint: type) -> list[type]:
        """
        Reduce a type hint. If the type hint is a `typing` module, it will be reduced to its original hinted types. For
        example: `typing.Union[int, float, typing.List[int]]` will return `[int, float, List[int]]` and
        `typing.List[int]` will return `[list]`. Regular type hints - Python object types cannot be reduced as they are
        already a core type.

        If a type hint cannot be reduced, an empty list will be returned.

        :param type_hint: The type hint to reduce.

        :return: The reduced type hint as list of hinted types or an empty list if the type hint could not be reduced.
        """
        # If it's not a typing type (meaning it's an actual object type) then we can't reduce it further:
        if not TypeHintUtils.is_typing_type(type_hint=type_hint):
            return []

        # If it's a type var, take its constraints (e.g. A = TypeVar("A", int, str) meaning an object of type A should
        # be an integer or a string). If it doesn't have constraints, return an empty list:
        if isinstance(type_hint, typing.TypeVar):
            if len(type_hint.__constraints__) == 0:
                return []
            return list(type_hint.__constraints__)

        # If it's a forward reference, we will try to import the reference:
        if isinstance(type_hint, typing.ForwardRef):
            try:
                # ForwardRef is initialized with the string type it represents and optionally a module path, so we
                # construct a full module path and try to parse it:
                arg = type_hint.__forward_arg__
                if type_hint.__forward_module__:
                    arg = f"{type_hint.__forward_module__}.{arg}"
                return [TypeHintUtils.parse_type_hint(type_hint=arg)]
            except (
                MLRunInvalidArgumentError
            ):  # May be raised from `TypeHintUtils.parse_type_hint`
                logger.warn(
                    f"Could not reduce the type hint '{type_hint}' as it is a forward reference to a class without "
                    f"it's full module path. To enable importing forward references, please provide the full module "
                    f"path to them. For example: use `ForwardRef('pandas.DataFrame')` instead of "
                    f"`ForwardRef('DataFrame')`."
                )
                return []

        # Get the origin of the typing type. An origin is the subscripted typing type (origin of Union[str, int] is
        # Union). The origin can be one of Callable, Tuple, Union, Literal, Final, ClassVar, Annotated or the actual
        # type alias (e.g. origin of List[int] is list):
        origin = typing.get_origin(type_hint)

        # If the typing type has no origin (e.g. None is returned), we cannot reduce it, so we return an empty list:
        if origin is None:
            return []

        # If the origin is a type of one of `builtins`, `contextlib` or `collections` (for example: List's origin is
        # list) then we can be sure there is nothing to reduce as it's a regular type:
        if not TypeHintUtils.is_typing_type(type_hint=origin):
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
            # Annotated is used to describe (add metadata to) a type, so we take the first argument (the type the
            # metadata is being added to):
            return [args[0]]
        if origin is typing.Final or origin is typing.ClassVar:
            # Both Final and ClassVar takes only one argument - the type:
            return [args[0]]

        # For Generic types we return an empty list:
        return []
