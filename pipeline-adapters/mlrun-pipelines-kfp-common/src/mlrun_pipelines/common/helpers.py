# Copyright 2024 Iguazio
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

import inspect
from collections.abc import Iterator, MutableMapping
from typing import Any, NoReturn

PROJECT_ANNOTATION = "mlrun/project"
RUN_ANNOTATION = "mlrun/pipeline-step-type"
FUNCTION_ANNOTATION = "mlrun/function-uri"


class FlexibleMapper(MutableMapping):
    """
    Custom mapper implementation that provides flexibility in handling the mapping
    of data between class attributes and a dictionary.

    This implementation includes compatibility with dictionary-like objects through
    get, set, and delete methods. This allows the class to be handled like a native dict,
    making it highly flexible.

    Inheritors of this class encapsulate KFP data models and abstract away from MLRun their
    differences across different versions

    Attributes:
        _external_data: The external data source in dict form.
    """

    _external_data: dict

    def __init__(self, external_data: Any):
        """
        Constructs a FlexibleMapper from the given external_data source.

        Args:
            external_data: the initial data source. Can be a dict or any object
            with a 'to_dict' method.
        """
        if isinstance(external_data, dict):
            self._external_data = external_data
        elif hasattr(external_data, "to_dict"):
            self._external_data = external_data.to_dict()

    def __getitem__(self, key: str) -> Any:
        """
        Gets the value for the given key. If the key is not a class attribute,
        it looks for it in the _external_data dict.

        Args:
            key: the key to look up.

        Returns:
            the value associated with the key.

        Raises:
            KeyError: if the key is not found
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return self._external_data[key]

    def __setitem__(self, key, value) -> NoReturn:
        """
        Sets the value for the given key. If the key isn't a class attribute,
        it sets it in the _external_data dict.

        Args:
            key: the key to set.
            value: the value to set for the key.
        """
        try:
            setattr(self, key, value)
        except AttributeError:
            self._external_data[key] = value

    def __delitem__(self, key) -> NoReturn:
        """
        Deletes the item associated with the given key. If the key isn't a class attribute,
        it deletes it in the _external_data dict.

        Args:
            key: the key to delete.

        Raises:
            KeyError: if the key is not found.
        """
        try:
            delattr(self, key)
        except AttributeError:
            del self._external_data[key]

    def __len__(self) -> int:
        """
        Returns the sum of the number of class attributes and items in the _external_data dict.

        Returns:
            the length of the mapping.
        """
        return len(self._external_data) + len(vars(self)) - 1

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the keys of the mapping. It yields keys only from the class
        attributes and not the _external_data dict.

        Returns
            an iterator over the keys.
        """
        yield from [
            m[0]
            for m in inspect.getmembers(self)
            if not callable(m[1]) and not m[0].startswith("_")
        ]

    def __bool__(self) -> bool:
        """
        Determines the boolean value of the mapping. The mapping is True if the _external_data dict is non-empty.

        Returns:
            True if the mapping is non-empty; False otherwise.
        """
        return bool(self._external_data)

    def to_dict(self) -> dict:
        """
        Converts the mapping to a dict. This method follows the attribute rules defined on __iter__

        Returns:
            a dict representation of the mapping.
        """
        return {a: getattr(self, a, None) for a in self}
