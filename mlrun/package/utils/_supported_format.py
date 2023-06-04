from abc import ABC
from typing import Dict, Generic, List, Type, TypeVar, Union

# A generic type for a supported format handler class type:
FileHandlerType = TypeVar("FileHandlerType")


class SupportedFormat(ABC, Generic[FileHandlerType]):
    """
    Library of supported formats by some builtin MLRun packagers.
    """

    # Add here the all the supported formats in ALL CAPS and their value as a string:
    ...

    # The map to use in the method `get_format_handler`. A dictionary of string key to a class type to handle that
    # format. New supported formats and handlers should be added to it:
    _FORMAT_HANDLERS_MAP: Dict[str, Type[FileHandlerType]] = {}

    @classmethod
    def get_all_formats(cls) -> List[str]:
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
    def get_format_handler(cls, fmt: str) -> Type[FileHandlerType]:
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
