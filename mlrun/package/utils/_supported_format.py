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
from abc import ABC
from typing import Generic, TypeVar, Union

# A generic type for a supported format handler class type:
FileHandlerType = TypeVar("FileHandlerType")


class SupportedFormat(ABC, Generic[FileHandlerType]):
    """
    Library of supported formats by some builtin MLRun packagers.
    """

    # Add here all the supported formats in ALL CAPS and their value as a string:
    ...

    # The map to use in the method `get_format_handler`. A dictionary of string key to a class type to handle that
    # format. New supported formats and handlers should be added to it:
    _FORMAT_HANDLERS_MAP: dict[str, type[FileHandlerType]] = {}

    @classmethod
    def get_all_formats(cls) -> list[str]:
        """
        Get all supported formats.

        :return: A list of all the supported formats.
        """
        return [
            value
            for key, value in cls.__dict__.items()
            if isinstance(value, str) and not key.startswith("_")
        ]

    @classmethod
    def get_format_handler(cls, fmt: str) -> type[FileHandlerType]:
        """
        Get the format handler to the provided format (file extension):

        :param fmt: The file extension to get the corresponding handler.

        :return: The handler class.
        """
        return cls._FORMAT_HANDLERS_MAP[fmt]

    @classmethod
    def match_format(cls, path: str) -> Union[str, None]:
        """
        Try to match one of the available formats this class holds to a given path.

        :param path: The path to match the format to.

        :return: The matched format if found and None otherwise.
        """
        formats = cls.get_all_formats()
        for fmt in formats:
            if path.endswith(f".{fmt}"):
                return fmt
        return None
