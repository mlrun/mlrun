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

import builtins
import collections
import importlib
import inspect
import itertools
import re
import types
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
        # Handle ellipsis (...) which is used in tuple type hints like tuple[int, ...]
        if type_hint is ...:
            return False
        # A type hint should be one of the based typing classes, meaning it will have "typing" as its module. Some
        # typing classes are considered a type (like `TypeVar`) so we check their type as well. The only case "types"
        # will be a module for the type of the type hint is for `GenericAlias` like `list[int]`.
        return (type_hint.__module__ in ["typing", "types"]) or (
            type(type_hint).__module__ in ["typing", "types"]
        )

    @staticmethod
    def is_pure_hint(type_hint: type) -> bool:
        """
        Check whether a given type hint is a pure type hint, meaning it cannot be instantiated as an object (like
        `list` from `list[int]` can) or refer to one (like `List[int]` can refer to `list[int]`). For example:
        `typing.Union` and `typing.TypeVar` are pure type hints.

        :param type_hint: The type hint to check.

        :return: True if the type hint is a pure type hint and False otherwise.
        """
        # A pure type hint should be first and for all a typing type:
        if not TypeHintUtils.is_typing_type(type_hint=type_hint):
            return False

        # Check for generic aliases (like `list[int]` or `List[int]`), which are not pure type hints
        # (`typing._GenericAlias` is tested via `type(typing.List[int])`):
        if (
            type(type_hint)
            is types.GenericAlias  # Python objects as type hints: list[int], dict[str, int], etc.
            or type(type_hint)
            is type(
                typing.List[int]  # noqa: UP006
            )  # Typing module type hints: List[int], Dict[str, int], etc.
        ):
            return False
        return True

    @staticmethod
    def parse_type_hint(type_hint: type | str) -> type:
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

        # Prepare a set of builtin types for quick lookup:
        builtin_types = {
            builtin_name: builtin_type
            for builtin_name, builtin_type in builtins.__dict__.items()
            if isinstance(builtin_type, type) or builtin_name == "None"
        }

        # Prepare custom scope for eval:
        scope = dict(globals())

        # Look for inner arguments inside square brackets (e.g. `List[int]`, `Dict[str, float]`, etc.) to extract types
        # for importing them before validating the main type hint (notice we ignore literals and values, so
        # for `typing.Literal["A"]` we'll not try to import `"A"`):
        raw_matches = re.finditer(
            r"""
             (?P<type>[a-zA-Z_][\w.]*) | # Identifiers (Types)
             (?P<value>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|-?\d+(?:\.\d+)?) # Literals (Values)
             """,
            type_hint,
            re.VERBOSE,
        )
        extracted_types = set([m.group("type") for m in raw_matches if m.group("type")])

        # Iterate over the extracted types to import them first for validation (as they are needed for the main hint):
        for extracted_type in extracted_types:
            # Validate the type hint is a valid module path:
            if not bool(
                re.fullmatch(
                    r"([a-zA-Z_][a-zA-Z0-9_]*\.)*[a-zA-Z_][a-zA-Z0-9_]*", extracted_type
                )
            ):
                raise MLRunInvalidArgumentError(
                    f"Invalid type hint. An input type hint must be a valid python class name or its module import "
                    f"path. For example: 'list', 'pandas.DataFrame', 'numpy.ndarray', "
                    f"'sklearn.linear_model.LinearRegression'. Type hint given: '{extracted_type}'."
                )
            # Look for a builtin type:
            if extracted_type in builtin_types:
                continue
            # If it's not a builtin, its should have a full module path, meaning at least one '.' to separate the module
            # and the class. If it doesn't, we will try to get the class from the main module:
            if "." not in extracted_type:
                logger.warn(
                    f"The type hint string given '{extracted_type}' is not a `builtins` python type. MLRun will try to "
                    f"look for it in the `__main__` module instead."
                )
                try:
                    return TypeHintUtils.parse_type_hint(
                        type_hint=f"__main__.{extracted_type}"
                    )
                except MLRunInvalidArgumentError:
                    raise MLRunInvalidArgumentError(
                        f"MLRun tried to get the type hint '{extracted_type}' but it can't as it is not a valid "
                        f"builtin Python type (one of `list`, `dict`, `str`, `int`, etc.) nor a locally declared type "
                        f"(from the `__main__` module). Pay attention using only the type as string is not allowed as "
                        f"the handler's scope is different than MLRun's. To properly give a type hint as string, "
                        f"please specify the full module path without aliases. For example: do not use `DataFrame` or "
                        f"`pd.DataFrame`, use `pandas.DataFrame`."
                    )
            # Get the module path and the type class (If we'll wish to support inner classes, the `rsplit` won't
            # work):
            module_path, extracted_type = extracted_type.rsplit(".", 1)
            # Replace alias if needed (alias assumed to be imported already, hence we look in globals):
            # For example:
            # If in handler scope there was `import A.B.C as abc` and user gave a type hint "abc.Something" then:
            # `module_path[0]` will be equal to "abc". Then, because it is an alias, it will appear in the globals,
            # so we'll replace the alias with the full module name in order to import the module.
            module_path_with_alias = module_path.split(".")
            if module_path_with_alias[0] in globals():
                module_path_with_alias[0] = globals()[
                    module_path_with_alias[0]
                ].__name__
            module_path_with_alias = ".".join(module_path_with_alias)
            # Import the module:
            try:
                module = importlib.import_module(module_path_with_alias)
            except ModuleNotFoundError as module_not_found_error:
                # May be raised from `importlib.import_module` in case the module does not exist.
                raise MLRunInvalidArgumentError(
                    f"MLRun tried to get the type hint '{extracted_type}' but the module '{module_path}' cannot be "
                    f"imported. Keep in mind that using alias in the module path (meaning: import module as alias) is "
                    f"not allowed. If the module path is correct, please make sure the module package is installed in "
                    f"the python interpreter."
                ) from module_not_found_error
            # Check the type exists in the module:
            if not hasattr(module, extracted_type):
                # Class type cannot be imported directly from the imported module.
                raise MLRunInvalidArgumentError(
                    f"MLRun tried to get the type hint '{extracted_type}' from the module '{module.__name__}' but it "
                    f"seems it doesn't exist. Make sure the class can be imported from the module with the exact "
                    f"module path you passed. Notice inner classes (a class inside of a class) are not supported."
                )
            # Add the type to the scope for eval:
            scope[extracted_type] = getattr(module, extracted_type)
            type_hint = type_hint.replace(
                f"{module_path}.{extracted_type}", extracted_type
            )

        # Lastly, we validate the main type hint structure:
        try:
            # Evaluate the type hint string to get the actual type hint:
            type_hint = eval(type_hint, scope)
        except Exception as eval_exception:
            raise MLRunInvalidArgumentError(
                f"MLRun tried to parse the type hint string '{type_hint}' but failed evaluating it to an actual type "
                f"hint. Make sure the type hint is a valid python type hint structure. Error: {eval_exception}"
            ) from eval_exception

        return type_hint

    @staticmethod
    def is_matching(
        object_type: type,
        type_hint: type | set[type],
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
    def deconstruct_type_hint(
        type_hint: type,
    ) -> tuple[type, tuple[type, ...] | type[inspect.Parameter.empty]]:
        """
        Deconstruct a type hint to its origin and argument types. For example: `typing.List[int]` will return
        `(list, (int,))`.

        Note: Ellipsis (`...`) in type hints like `tuple[int, ...]` are filtered out from the arguments,
        since they indicate variable length rather than an actual type. In this case, `tuple[int, ...]`
        will return `(tuple, (int,))`.

        :param type_hint: The type hint to deconstruct.

        :return: A tuple of the origin type and a tuple of argument types, or `inspect.Parameter.empty`
                 if there are no arguments (or only ellipsis arguments).
        """
        # Get the origin of the type hint:
        origin = typing.get_origin(type_hint)
        if origin is None:
            # Not a typing type, return as is with no args:
            return type_hint, inspect.Parameter.empty

        # Get the type hint's subscriptions - arguments:
        args = typing.get_args(type_hint)

        # Filter out ellipsis (...) from the args, as it indicates variable length, not a type.
        # This is common in tuple type hints like `tuple[int, ...]` which means a tuple of any number of ints.
        args = tuple(arg for arg in args if arg is not ...)

        # Return inspect.Parameter.empty if no args remain after filtering:
        if len(args) == 0:
            return origin, inspect.Parameter.empty

        return origin, args

    @staticmethod
    def reduce_type_hint(
        type_hint: type | set[type],
    ) -> set[type]:
        """
        Reduce a type hint (or a set of type hints) using the `_reduce_type_hint` function.

        :param type_hint: The type hint to reduce.

        :return: The reduced type hints set or an empty set if the type hints could not be reduced.
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

        # Get the type's subscriptions - arguments, in order to reduce it to them (we know for sure there are arguments,
        # otherwise origin would have been None):
        args = typing.get_args(type_hint)

        # If the typing type has no origin (e.g. None is returned) and it has no args (meaning it is a type without
        # subscriptions), we cannot reduce it, so we return an empty list:
        if origin is None:
            return []

        # Check for a special typing type and return the reduced type accordingly:
        if origin is typing.Callable or origin is collections.abc.Callable:
            # A callable cannot be reduced to its arguments, so we'll return the origin - Callable:
            return [collections.abc.Callable]
        if origin is typing.Literal:
            # Literal arguments are not types, but values. So we'll take the types of the values as the reduced type:
            return [type(arg) for arg in args]
        if origin is typing.Union or origin is types.UnionType:
            # A union is reduced to its arguments:
            return list(args)
        if origin is typing.Annotated:
            # Annotated is used to describe (add metadata to) a type, so we take the first argument (the type the
            # metadata is being added to):
            return [args[0]]
        if origin is typing.Final or origin is typing.ClassVar:
            # Both Final and ClassVar takes only one argument - the type:
            return [args[0]]
        # For `typing.Generic` we return `[]` as it cannot be reduced further.

        # It is not a special typing type, it is most likely a `types.GenericAlias` or `typing._GenericAlias` so we
        # return the origin:
        # TODO: Technically we should reduce a generic alias from its args by one level at a time.
        #       For example:
        #       * `Dict[str, list[int]]` will be reduced to `dict[str, list]` and on another call to `dict`.
        #       * `List[int | str | dict[str, int | float]]` will be reduced to `List[int | str | dict[str, int]]` and
        #         `List[int | str | dict[str, float]]`, which then both will yield `List[int | str | dict]`, and finally
        #         `list`.
        #       The algorithm should find the deepest typing type possible (like union) and reduce it. Once there are no
        #       more typing types to reduce, the deepest args should be reduced until we reach the core origin we now
        #       return by default. If there are two or more args to reduce, all combinations (permutations) should be
        #       returned.
        #       For now, we reduce only to the origin type itself regardless of its args, as it seems to be sufficient
        #       for our users.
        return [origin]
