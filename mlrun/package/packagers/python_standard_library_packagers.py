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
import os
import pathlib
import tempfile
from typing import Optional, Union

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError

from ..utils import (
    DEFAULT_ARCHIVE_FORMAT,
    DEFAULT_STRUCT_FILE_FORMAT,
    ArchiveSupportedFormat,
    ArtifactType,
    StructFileSupportedFormat,
)
from .default_packager import DefaultPackager

# ----------------------------------------------------------------------------------------------------------------------
# builtins packagers:
# ----------------------------------------------------------------------------------------------------------------------


class NonePackager(DefaultPackager):
    """
    ``None`` packager.
    """

    # TODO: From python 3.10 the `PACKABLE_OBJECT_TYPE` should be changed to `types.NoneType`
    PACKABLE_OBJECT_TYPE = type(None)
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.RESULT

    # TODO: `None` as pickle will be available from Python 3.10, so this method can be removed once we move to 3.10.
    def get_supported_artifact_types(self) -> list[str]:
        """
        Get all the supported artifact types on this packager. It will be the same as `DefaultPackager` but without the
        'object' artifact type support (None cannot be pickled, only from Python 3.10, and it should not be pickled
        anyway as it is simply None - a result will do).

        :return: A list of all the supported artifact types.
        """
        supported_artifacts = super().get_supported_artifact_types()
        supported_artifacts.remove("object")
        return supported_artifacts


class IntPackager(DefaultPackager):
    """
    ``builtins.int`` packager.
    """

    PACKABLE_OBJECT_TYPE = int
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.RESULT


class FloatPackager(DefaultPackager):
    """
    ``builtins.float`` packager.
    """

    PACKABLE_OBJECT_TYPE = float
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.RESULT


class BoolPackager(DefaultPackager):
    """
    ``builtins.bool`` packager.
    """

    PACKABLE_OBJECT_TYPE = bool
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.RESULT


class StrPackager(DefaultPackager):
    """
    ``builtins.str`` packager.
    """

    PACKABLE_OBJECT_TYPE = str
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.RESULT
    DEFAULT_UNPACKING_ARTIFACT_TYPE = ArtifactType.PATH

    def pack_path(
        self, obj: str, key: str, archive_format: str = DEFAULT_ARCHIVE_FORMAT
    ) -> tuple[Artifact, dict]:
        """
        Pack a path string value content (pack the file or directory in that path).

        :param obj:            The string path value to pack.
        :param key:            The key to use for the artifact.
        :param archive_format: The archive format to use in case the path is of a directory. Default is zip.

        :return: The packed artifact and instructions.
        """
        # TODO: Add a configuration like `archive_file: bool = False` to enable archiving a single file to shrink it in
        #       size. In that case the `is_directory` instruction will make it so when an archive is received, if its
        #       a directory, when exporting it a directory path should be returned. And, if its a file, a path to the
        #       single file exported should be returned.
        # Verify the path is of an existing file:
        if not os.path.exists(obj):
            raise MLRunInvalidArgumentError(f"The given path do not exist: '{obj}'")

        # Proceed by path type (file or directory):
        if os.path.isfile(obj):
            # Create the artifact:
            artifact = Artifact(key=key, src_path=os.path.abspath(obj))
            instructions = {"is_directory": False}
        elif os.path.isdir(obj):
            # Archive the directory:
            output_path = tempfile.mkdtemp()
            archiver = ArchiveSupportedFormat.get_format_handler(fmt=archive_format)
            archive_path = archiver.create_archive(
                directory_path=obj, output_path=output_path
            )
            # Create the artifact:
            artifact = Artifact(key=key, src_path=archive_path)
            instructions = {"archive_format": archive_format, "is_directory": True}
        else:
            raise MLRunInvalidArgumentError(
                f"The given path is not a file nor a directory: '{obj}'"
            )

        return artifact, instructions

    def unpack_path(
        self,
        data_item: DataItem,
        is_directory: bool = False,
        archive_format: Optional[str] = None,
    ) -> str:
        """
        Unpack a data item representing a path string. If the path is of a file, the file is downloaded to a local
        temporary directory and its path is returned. If the path is of a directory, the archive is extracted and the
        directory path extracted is returned.

        :param data_item:      The data item to unpack.
        :param is_directory:   Whether the path should be treated as a file or a directory. Files (even archives like
                               zip) won't be extracted.
        :param archive_format: The archive format to use in case the path is of a directory. Default is None - will be
                               read by the archive file extension.

        :return: The unpacked string.
        """
        # Get the file:
        path = self.get_data_item_local_path(data_item=data_item)

        # If it's not a directory, return the file path. Otherwise, it should be extracted according to the archive
        # format:
        if not is_directory:
            return path

        # Get the archive format by the file extension:
        if archive_format is None:
            archive_format = ArchiveSupportedFormat.match_format(path=path)
        if archive_format is None:
            raise MLRunInvalidArgumentError(
                f"Archive format of {data_item.key} ('{''.join(pathlib.Path(path).suffixes)}') is not supported. "
                f"Supported formats are: {' '.join(ArchiveSupportedFormat.get_all_formats())}"
            )

        # Extract the archive:
        archiver = ArchiveSupportedFormat.get_format_handler(fmt=archive_format)
        directory_path = archiver.extract_archive(
            archive_path=path, output_path=os.path.dirname(path)
        )

        # Mark the extracted content for future clear:
        self.add_future_clearing_path(path=directory_path)

        # Return the extracted directory path:
        return directory_path


class _BuiltinCollectionPackager(DefaultPackager):
    """
    A base packager for builtin python dictionaries and lists as they share common artifact and file types.
    """

    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.RESULT
    DEFAULT_UNPACKING_ARTIFACT_TYPE = ArtifactType.FILE

    def pack_file(
        self,
        obj: Union[dict, list],
        key: str,
        file_format: str = DEFAULT_STRUCT_FILE_FORMAT,
    ) -> tuple[Artifact, dict]:
        """
        Pack a builtin collection as a file by the given format.

        :param obj:         The builtin collection to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is json.

        :return: The packed artifact and instructions.
        """
        # Write to file:
        formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
        temp_directory = pathlib.Path(tempfile.mkdtemp())
        self.add_future_clearing_path(path=temp_directory)
        file_path = temp_directory / f"{key}.{file_format}"
        formatter.write(obj=obj, file_path=str(file_path))

        # Create the artifact and instructions:
        artifact = Artifact(key=key, src_path=os.path.abspath(file_path))
        instructions = {"file_format": file_format}

        return artifact, instructions

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> Union[dict, list]:
        """
        Unpack a builtin collection from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the builtin collection. Default is None - will be read by
                            the file extension.

        :return: The unpacked builtin collection.
        """
        # Get the file:
        file_path = self.get_data_item_local_path(data_item=data_item)

        # Get the archive format by the file extension if needed:
        if file_format is None:
            file_format = StructFileSupportedFormat.match_format(path=file_path)
        if file_format is None:
            raise MLRunInvalidArgumentError(
                f"File format of {data_item.key} ('{''.join(pathlib.Path(file_path).suffixes)}') is not supported. "
                f"Supported formats are: {' '.join(StructFileSupportedFormat.get_all_formats())}"
            )

        # Read the object:
        formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
        obj = formatter.read(file_path=file_path)

        return obj


class DictPackager(_BuiltinCollectionPackager):
    """
    ``builtins.dict`` packager.
    """

    PACKABLE_OBJECT_TYPE = dict

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> dict:
        """
        Unpack a dictionary from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the dictionary. Default is None - will be read by the
                            file extension.

        :return: The unpacked dictionary.
        """
        # Unpack the object:
        obj = super().unpack_file(data_item=data_item, file_format=file_format)

        # Check if needed to cast from list:
        if isinstance(obj, list):
            return {index: element for index, element in enumerate(obj)}
        return obj


class ListPackager(_BuiltinCollectionPackager):
    """
    ``builtins.list`` packager.
    """

    PACKABLE_OBJECT_TYPE = list

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> list:
        """
        Unpack a list from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the list. Default is None - will be read by the file
                            extension.

        :return: The unpacked list.
        """
        # Unpack the object:
        obj = super().unpack_file(data_item=data_item, file_format=file_format)

        # Check if needed to cast from dict:
        if isinstance(obj, dict):
            return list(obj.values())
        return obj


class TuplePackager(ListPackager):
    """
    ``builtins.tuple`` packager.

    Notice: a ``tuple`` returned from a function is usually treated as multiple returned objects, and so MLRun will try
    to pack each of them separately and not as a single tuple. For example::

        def example_func_1():
            return 10, [1, 2, 3], "Hello MLRun"

    Will be returned as a ``tuple`` of 3 items: `(10, [1, 2, 3], "Hello MLRun")` but the items will be packaged
    separately one by one and not as a single ``tuple``.

    In order to pack tuples (not recommended), use the configuration::

        mlrun.mlconf.packagers.pack_tuple = True

    Or more correctly, cast your returned tuple to a ``list`` like so::

        def example_func_2():
            my_tuple = (2, 4)
            return list(my_tuple)
    """

    PACKABLE_OBJECT_TYPE = tuple

    def pack_result(self, obj: tuple, key: str) -> dict:
        """
        Pack a tuple as a result.

        :param obj: The tuple to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return super().pack_result(obj=list(obj), key=key)

    def pack_file(
        self, obj: tuple, key: str, file_format: str = DEFAULT_STRUCT_FILE_FORMAT
    ) -> tuple[Artifact, dict]:
        """
        Pack a tuple as a file by the given format.

        :param obj:         The tuple to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is json.

        :return: The packed artifact and instructions.
        """
        return super().pack_file(obj=list(obj), key=key, file_format=file_format)

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> tuple:
        """
        Unpack a tuple from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the tuple. Default is None - will be read by the file
                            extension.

        :return: The unpacked tuple.
        """
        return tuple(super().unpack_file(data_item=data_item, file_format=file_format))


class SetPackager(ListPackager):
    """
    ``builtins.set`` packager.
    """

    PACKABLE_OBJECT_TYPE = set

    def pack_result(self, obj: set, key: str) -> dict:
        """
        Pack a set as a result.

        :param obj: The set to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return super().pack_result(obj=list(obj), key=key)

    def pack_file(
        self, obj: set, key: str, file_format: str = DEFAULT_STRUCT_FILE_FORMAT
    ) -> tuple[Artifact, dict]:
        """
        Pack a set as a file by the given format.

        :param obj:         The set to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is json.

        :return: The packed artifact and instructions.
        """
        return super().pack_file(obj=list(obj), key=key, file_format=file_format)

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> set:
        """
        Unpack a set from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the set. Default is None - will be read by the file
                            extension.

        :return: The unpacked set.
        """
        return set(super().unpack_file(data_item=data_item, file_format=file_format))


class FrozensetPackager(SetPackager):
    """
    ``builtins.frozenset`` packager.
    """

    PACKABLE_OBJECT_TYPE = frozenset

    def pack_file(
        self, obj: frozenset, key: str, file_format: str = DEFAULT_STRUCT_FILE_FORMAT
    ) -> tuple[Artifact, dict]:
        """
        Pack a frozenset as a file by the given format.

        :param obj:         The frozenset to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is json.

        :return: The packed artifact and instructions.
        """
        return super().pack_file(obj=set(obj), key=key, file_format=file_format)

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> frozenset:
        """
        Unpack a frozenset from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the frozenset. Default is None - will be read by the file
                            extension.

        :return: The unpacked frozenset.
        """
        return frozenset(
            super().unpack_file(data_item=data_item, file_format=file_format)
        )


class BytesPackager(ListPackager):
    """
    ``builtins.bytes`` packager.
    """

    PACKABLE_OBJECT_TYPE = bytes

    def pack_result(self, obj: bytes, key: str) -> dict:
        """
        Pack bytes as a result.

        :param obj: The bytearray to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return {key: obj}

    def pack_file(
        self, obj: bytes, key: str, file_format: str = DEFAULT_STRUCT_FILE_FORMAT
    ) -> tuple[Artifact, dict]:
        """
        Pack a bytes as a file by the given format.

        :param obj:         The bytes to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is json.

        :return: The packed artifact and instructions.
        """
        return super().pack_file(obj=list(obj), key=key, file_format=file_format)

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> bytes:
        """
        Unpack a bytes from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the bytes. Default is None - will be read by the file
                            extension.

        :return: The unpacked bytes.
        """
        return bytes(super().unpack_file(data_item=data_item, file_format=file_format))


class BytearrayPackager(BytesPackager):
    """
    ``builtins.bytearray`` packager.
    """

    PACKABLE_OBJECT_TYPE = bytearray

    def pack_result(self, obj: bytearray, key: str) -> dict:
        """
        Pack a bytearray as a result.

        :param obj: The bytearray to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return {key: bytes(obj)}

    def pack_file(
        self, obj: bytearray, key: str, file_format: str = DEFAULT_STRUCT_FILE_FORMAT
    ) -> tuple[Artifact, dict]:
        """
        Pack a bytearray as a file by the given format.

        :param obj:         The bytearray to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is json.

        :return: The packed artifact and instructions.
        """
        return super().pack_file(obj=bytes(obj), key=key, file_format=file_format)

    def unpack_file(
        self, data_item: DataItem, file_format: Optional[str] = None
    ) -> bytearray:
        """
        Unpack a bytearray from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the bytearray. Default is None - will be read by the file
                            extension.

        :return: The unpacked bytearray.
        """
        return bytearray(
            super().unpack_file(data_item=data_item, file_format=file_format)
        )


# ----------------------------------------------------------------------------------------------------------------------
# pathlib packagers:
# ----------------------------------------------------------------------------------------------------------------------


class PathPackager(StrPackager):
    """
    ``pathlib.Path`` packager. It is also used for all `Path` inheriting pathlib objects (`PosixPath` and
    `WindowsPath`).
    """

    PACKABLE_OBJECT_TYPE = pathlib.Path
    PACK_SUBCLASSES = True
    DEFAULT_PACKING_ARTIFACT_TYPE = "path"

    def pack_result(self, obj: pathlib.Path, key: str) -> dict:
        """
        Pack the `Path` as a string result.

        :param obj: The `Path` to pack.
        :param key: The key to use in the results dictionary.

        :return: The packed result.
        """
        return super().pack_result(obj=str(obj), key=key)

    def pack_path(
        self, obj: pathlib.Path, key: str, archive_format: str = DEFAULT_ARCHIVE_FORMAT
    ) -> tuple[Artifact, dict]:
        """
        Pack a `Path` value (pack the file or directory in that path).

        :param obj:            The `Path` to pack.
        :param key:            The key to use for the artifact.
        :param archive_format: The archive format to use in case the path is of a directory. Default is zip.

        :return: The packed artifact and instructions.
        """
        return super().pack_path(obj=str(obj), key=key, archive_format=archive_format)

    def unpack_path(
        self,
        data_item: DataItem,
        is_directory: bool = False,
        archive_format: Optional[str] = None,
    ) -> pathlib.Path:
        """
        Unpack a data item representing a `Path`. If the path is of a file, the file is downloaded to a local
        temporary directory and its path is returned. If the path is of a directory, the archive is extracted and the
        directory path extracted is returned.

        :param data_item:      The data item to unpack.
        :param is_directory:   Whether the path should be treated as a file or a directory. Files (even archives like
                               zip) won't be extracted.
        :param archive_format: The archive format to use in case the path is of a directory. Default is None - will be
                               read by the archive file extension.

        :return: The unpacked `Path`.
        """
        return pathlib.Path(
            super().unpack_path(
                data_item=data_item,
                is_directory=is_directory,
                archive_format=archive_format,
            )
        )


# ----------------------------------------------------------------------------------------------------------------------
# TODO: collection packagers:
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# TODO: datetime packagers:
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# TODO: enum packagers:
# ----------------------------------------------------------------------------------------------------------------------
