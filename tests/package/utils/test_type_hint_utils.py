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

import collections
import inspect
import typing

import pytest

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.package.utils.type_hint_utils import TypeHintUtils


class SomeClass:
    """
    To add a custom type for the type hinting test.
    """


class AnotherClass(SomeClass):
    """
    To add a custom inheriting class for match test.
    """


@pytest.mark.parametrize(
    "type_hint, expected_result",
    [
        (typing.Optional[int], True),
        (typing.Union[str, int], True),
        (typing.List, True),  # noqa: UP006
        (typing.Tuple[int, str], True),  # noqa: UP006
        (tuple[int, str], True),
        (typing.TypeVar("A", int, str), True),
        (typing.ForwardRef("pandas.DataFrame"), True),
        (list, False),
        (int, False),
        (SomeClass, False),
        (list[int], True),
        (tuple[int, str], True),
        (str | int, True),
    ],
)
def test_is_typing_type(type_hint: type, expected_result: bool):
    """
    Test the `TypeHintUtils.is_typing_type` function with multiple types.

    :param type_hint:       The type to check.
    :param expected_result: The expected result.
    """
    assert TypeHintUtils.is_typing_type(type_hint=type_hint) == expected_result


@pytest.mark.parametrize(
    "type_hint, expected_result",
    [
        # Pure type hints (cannot be instantiated, return True):
        (typing.Union, True),
        (typing.TypeVar("A", int, str), True),
        (typing.ForwardRef("pandas.DataFrame"), True),
        (typing.Callable, True),
        (typing.Literal, True),
        (typing.Optional, True),
        (typing.Annotated, True),
        (typing.Final, True),
        (typing.ClassVar, True),
        # Generic aliases (not pure, return False):
        (list[int], False),
        (tuple[int, str], False),
        (dict[str, int], False),
        (typing.List[int], False),  # noqa: UP006
        (typing.Dict[str, int], False),  # noqa: UP006
        (typing.Tuple[int, str], False),  # noqa: UP006
        # Regular types (not typing types, return False):
        (list, False),
        (int, False),
        (str, False),
        (SomeClass, False),
    ],
)
def test_is_pure_hint(type_hint: type, expected_result: bool):
    """
    Test the `TypeHintUtils.is_pure_hint` function with multiple types.

    :param type_hint:       The type to check.
    :param expected_result: The expected result.
    """
    assert TypeHintUtils.is_pure_hint(type_hint=type_hint) == expected_result


@pytest.mark.parametrize(
    "type_hint, expected_result",
    [
        # Generic aliases (have origin and args):
        (list[int], (list, (int,))),
        (dict[str, int], (dict, (str, int))),
        (tuple[int, str], (tuple, (int, str))),
        (typing.List[int], (list, (int,))),  # noqa: UP006
        (typing.Dict[str, int], (dict, (str, int))),  # noqa: UP006
        (typing.Tuple[int, str], (tuple, (int, str))),  # noqa: UP006
        # Typing special forms with args:
        (typing.Optional[int], (typing.Union, (int, type(None)))),
        (typing.Union[int, str], (typing.Union, (int, str))),
        # Ellipsis in tuple (should be filtered out):
        (tuple[int, ...], (tuple, (int,))),
        # Non-generic types (no origin, return empty):
        (list, (list, inspect.Parameter.empty)),
        (int, (int, inspect.Parameter.empty)),
        (str, (str, inspect.Parameter.empty)),
        (SomeClass, (SomeClass, inspect.Parameter.empty)),
    ],
)
def test_deconstruct_type_hint(type_hint: type, expected_result: tuple):
    """
    Test the `TypeHintUtils.deconstruct_type_hint` function with multiple types.

    :param type_hint:       The type hint to deconstruct.
    :param expected_result: The expected (origin, args) tuple.
    """
    assert TypeHintUtils.deconstruct_type_hint(type_hint=type_hint) == expected_result


@pytest.mark.parametrize(
    "type_string, expected_type",
    [
        ("int", int),
        ("list", list),
        ("typing.Tuple[int, str]", typing.Tuple[int, str]),  # noqa: UP006
        ("tuple[int, str]", tuple[int, str]),
        ("dict[str, int]", dict[str, int]),
        ("typing.Optional[float]", typing.Optional[float]),
        ("typing.Union[str, int]", typing.Union[str, int]),
        ("str | int", str | int),
        ("tests.package.utils.test_type_hint_utils.SomeClass", SomeClass),
        (
            'dict[str, set[tests.package.utils.test_type_hint_utils.SomeClass]] | None | typing.Literal["A", "B"]',
            dict[str, set[SomeClass]] | None | typing.Literal["A", "B"],
        ),
        (
            "fail",
            "MLRun tried to get the type hint 'fail' but it can't as it is not a valid builtin Python type (one of "
            "`list`, `dict`, `str`, `int`, etc.) nor a locally declared type (from the `__main__` module).",
        ),
        (
            "tests.package.utils.test_type_hint_utils.Fail",
            "MLRun tried to get the type hint 'Fail' from the module 'tests.package.utils.test_type_hint_utils' but it "
            "seems it doesn't exist.",
        ),
        (
            "module_not_exist.Fail",
            "MLRun tried to get the type hint 'Fail' but the module 'module_not_exist' cannot be imported.",
        ),
        ("list[int", "Make sure the type hint is a valid python type hint structure"),
        (
            "int | str |",
            "Make sure the type hint is a valid python type hint structure",
        ),
        (
            "typing.List[]",
            "Make sure the type hint is a valid python type hint structure",
        ),
    ],
)
def test_parse_type_hint(type_string: str, expected_type: str | type):
    """
    Test the `TypeHintUtils.parse_type_hint` function with multiple types.

    :param type_string:   The type to parse and
    :param expected_type: The expected parsed type. A string value indicates the parsing should fail with the provided
                          error message in the variable.
    """
    try:
        parsed_type = TypeHintUtils.parse_type_hint(type_hint=type_string)
        assert parsed_type == expected_type
    except MLRunInvalidArgumentError as error:
        if isinstance(expected_type, str):
            assert expected_type in str(error)
        else:
            raise error


@pytest.mark.parametrize(
    "object_type, type_hint, include_subclasses, reduce_type_hint, result",
    [
        (int, int, True, False, True),
        (int, str, True, True, False),
        (typing.Union[int, str], typing.Union[str, int], True, True, True),
        (typing.Union[int, str, bool], typing.Union[str, int], True, False, False),
        (int, typing.Union[int, str], True, False, False),
        (int, typing.Union[int, str], True, True, True),
        (AnotherClass, SomeClass, True, False, True),
        (AnotherClass, SomeClass, False, False, False),
        (SomeClass, AnotherClass, True, False, False),
        (AnotherClass, {SomeClass, int, str}, True, False, True),
        (AnotherClass, {SomeClass, int, str}, False, False, False),
        (SomeClass, {AnotherClass, int, str}, True, False, False),
    ],
)
def test_is_matching(
    object_type: type,
    type_hint: type,
    include_subclasses: bool,
    reduce_type_hint: bool,
    result: bool,
):
    """
    Test the `TypeHintUtils.is_matching` function with multiple types.

    :param object_type:        The type to match.
    :param type_hint:          The options to match to (the type hint of an object).
    :param include_subclasses: Whether subclasses considered a match.
    :param reduce_type_hint:   Whether to reduce the type hint to match with its reduced hints.
    :param result:             Expected test result.
    """
    assert (
        TypeHintUtils.is_matching(
            object_type=object_type,
            type_hint=type_hint,
            include_subclasses=include_subclasses,
            reduce_type_hint=reduce_type_hint,
        )
        == result
    )


@pytest.mark.parametrize(
    "type_hint, expected_result",
    [
        # `typing.TypeVar` usages:
        (typing.TypeVar("A", int, str, list[int]), {int, str, list[int]}),
        (typing.TypeVar("A"), set()),
        (typing.TypeVar, set()),
        # `typing.ForwardRef` usage:
        (typing.ForwardRef("SomeClass"), set()),
        (
            typing.ForwardRef(
                "SomeClass", module="tests.package.utils.test_type_hint_utils"
            ),
            {SomeClass},
        ),
        (
            typing.ForwardRef("tests.package.utils.test_type_hint_utils.SomeClass"),
            {SomeClass},
        ),
        (typing.ForwardRef, set()),
        # `typing.Callable` usages:
        (typing.Callable, {collections.abc.Callable}),
        (
            typing.Callable[[int, int], tuple[str, str]],
            {collections.abc.Callable},
        ),
        (collections.abc.Callable, set()),
        # `typing.Literal` usages:
        (typing.Literal["r", "w", 9], {str, int}),
        (typing.Literal, set()),
        # `typing.Union` usages:
        (typing.Union[int, float], {int, float}),
        (
            typing.Union[int, float, typing.Union[str, list]],
            {int, float, str, list},
        ),
        (
            typing.Union[int, str, list[tuple[int, str, SomeClass]]],
            {int, str, list[tuple[int, str, SomeClass]]},
        ),
        (typing.Union, set()),
        # `typing.Optional` usages:
        (typing.Optional[int], {type(None), int}),
        (typing.Optional[typing.Union[str, list]], {type(None), str, list}),
        (typing.Optional, set()),
        # `typing.Annotated` usages:
        (typing.Annotated[int, 3, 6], {int}),
        (typing.Annotated, set()),
        # `typing.Final` usages:
        (
            typing.Final[list[tuple[int, str, SomeClass]]],
            {list[tuple[int, str, SomeClass]]},
        ),
        (typing.Final, set()),
        # `typing.ClassVar` usages:
        (
            typing.ClassVar[typing.Union[int, str, list[tuple[int, str, SomeClass]]]],
            {typing.Union[int, str, list[tuple[int, str, SomeClass]]]},
        ),
        (typing.ClassVar, set()),
        # Other `typing`:
        (typing.List, {list}),  # noqa: UP006
        (list[tuple[int, str, SomeClass]], {list}),
        (tuple[int, str, SomeClass], {tuple}),
        # `collections` types:
        (typing.OrderedDict[str, int], {collections.OrderedDict}),
        (typing.OrderedDict, {collections.OrderedDict}),
        (collections.OrderedDict, set()),
        # Multiple types to reduce:
        ({int, str, list[int]}, {list}),
        (list[str], {list}),
        (str | int, {str, int}),
    ],
)
def test_reduce_type_hint(type_hint: type, expected_result: set[type]):
    """
    Test the `TypeHintUtils.reduce_type_hint` function with multiple type hints.

    :param type_hint:       The type hint to reduce.
    :param expected_result: The expected result.
    """
    assert TypeHintUtils.reduce_type_hint(type_hint=type_hint) == expected_result
