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
import collections
import typing

import pytest

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.package.utils.type_hint_utils import TypeHintUtils


class SomeClass:
    """
    To add a custom type for the type hinting test.
    """

    pass


class AnotherClass(SomeClass):
    """
    To add a custom inheriting class for match test.
    """

    pass


@pytest.mark.parametrize(
    "type_hint, expected_result",
    [
        (typing.Optional[int], True),
        (typing.Union[str, int], True),
        (typing.List, True),
        (typing.Tuple[int, str], True),
        (typing.TypeVar("A", int, str), True),
        (typing.ForwardRef("pandas.DataFrame"), True),
        (list, False),
        (int, False),
        (SomeClass, False),
        # TODO: Uncomment once we support Python >= 3.9:
        # (list[int], True),
        # (tuple[int, str], True),
        # TODO: Uncomment once we support Python >= 3.10:
        # (str | int, True),
    ],
)
def test_is_typing_type(type_hint: typing.Type, expected_result: bool):
    """
    Test the `TypeHintUtils.is_typing_type` function with multiple types.

    :param type_hint:       The type to check.
    :param expected_result: The expected result.
    """
    assert TypeHintUtils.is_typing_type(type_hint=type_hint) == expected_result


@pytest.mark.parametrize(
    "type_string, expected_type",
    [
        ("int", int),
        ("list", list),
        ("tests.package.utils.test_type_hint_utils.SomeClass", SomeClass),
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
    ],
)
def test_parse_type_hint(type_string: str, expected_type: typing.Union[str, type]):
    """
    Test the `TypeHintUtils.parse_type_hint` function with multiple types.

    :param type_string:   The type to parse and
    :param expected_type: The expected parsed type. A string value indicates the parsing should fail with the provided
                          error message in the variable.
    """
    try:
        parsed_type = TypeHintUtils.parse_type_hint(type_hint=type_string)
        assert parsed_type is expected_type
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
        (typing.TypeVar("A", int, str, typing.List[int]), {int, str, typing.List[int]}),
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
            typing.Callable[[int, int], typing.Tuple[str, str]],
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
            typing.Union[int, str, typing.List[typing.Tuple[int, str, SomeClass]]],
            {int, str, typing.List[typing.Tuple[int, str, SomeClass]]},
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
            typing.Final[typing.List[typing.Tuple[int, str, SomeClass]]],
            {typing.List[typing.Tuple[int, str, SomeClass]]},
        ),
        (typing.Final, set()),
        # `typing.ClassVar` usages:
        (
            typing.ClassVar[
                typing.Union[int, str, typing.List[typing.Tuple[int, str, SomeClass]]]
            ],
            {typing.Union[int, str, typing.List[typing.Tuple[int, str, SomeClass]]]},
        ),
        (typing.ClassVar, set()),
        # Other `typing`:
        (typing.List, {list}),
        (typing.List[typing.Tuple[int, str, SomeClass]], {list}),
        (typing.Tuple[int, str, SomeClass], {tuple}),
        # `collections` types:
        (typing.OrderedDict[str, int], {collections.OrderedDict}),
        (typing.OrderedDict, {collections.OrderedDict}),
        (collections.OrderedDict, set()),
        # Multiple types to reduce:
        ({int, str, typing.List[int]}, {list}),
        # TODO: Uncomment once we support Python >= 3.9:
        # (list[str], {list}),
        # TODO: Uncomment once we support Python >= 3.10:
        # (str | int, {str, int}),
    ],
)
def test_reduce_type_hint(
    type_hint: typing.Type, expected_result: typing.Set[typing.Type]
):
    """
    Test the `TypeHintUtils.reduce_type_hint` function with multiple type hints.

    :param type_hint:       The type hint to reduce.
    :param expected_result: The expected result.
    """
    assert TypeHintUtils.reduce_type_hint(type_hint=type_hint) == expected_result
