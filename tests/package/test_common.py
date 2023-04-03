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
import collections
import typing

import pytest

from mlrun.package.common import TypingUtils


class SomeClass:
    pass


@pytest.mark.parametrize(
    "typing_type_test",
    [
        (typing.Optional[int], True),
        (typing.Union[str, int], True),
        (typing.List, True),
        (typing.Tuple[int, str], True),
        (typing.TypeVar("A", int, str), True),
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
def test_is_typing_type(typing_type_test: typing.Tuple[typing.Type, bool]):
    """
    Test the `TypingUtils.is_typing_type` function with multiple types.

    :param typing_type_test: A tuple of the type to check and the expected result.
    """
    assert (
        TypingUtils.is_typing_type(type_hint=typing_type_test[0]) == typing_type_test[1]
    )


@pytest.mark.parametrize(
    "type_hint_test",
    [
        # `typing.TypeVar` usages:
        (typing.TypeVar("A", int, str, typing.List[int]), {int, str, typing.List[int]}),
        (typing.TypeVar("A"), set()),
        (typing.TypeVar, set()),
        # `typing.ForwardRef` usage:
        (typing.ForwardRef("SomeClass"), set()),
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
    type_hint_test: typing.Tuple[typing.Type, typing.Set[typing.Type]]
):
    """
    Test the `TypingUtils.reduce_type_hint` function with multiple type hints.

    :param type_hint_test: A tuple of the type hint to reduce and the expected result.
    """
    assert (
        TypingUtils.reduce_type_hint(type_hint=type_hint_test[0]) == type_hint_test[1]
    )
