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
import os
import pathlib

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError

from ..utils import ArchiveFormat, ArtifactType
from .default_packager import DefaultPackager


# builtins packagers:
class IntPackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = int
    DEFAULT_ARTIFACT_TYPE = ArtifactType.RESULT


class FloatPackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = float
    DEFAULT_ARTIFACT_TYPE = ArtifactType.RESULT


class BytesPackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = bytes
    DEFAULT_ARTIFACT_TYPE = ArtifactType.RESULT


class BoolPackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = bool
    DEFAULT_ARTIFACT_TYPE = ArtifactType.RESULT


class StrPackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = str
    DEFAULT_ARTIFACT_TYPE = ArtifactType.RESULT

    @classmethod
    def pack_file(cls, obj: str, key: str):
        # Verify the path is of an existing file:
        if not os.path.exists(obj):
            raise MLRunInvalidArgumentError(f"The given path do not exist: '{obj}'")
        if not os.path.isfile(obj):
            raise MLRunInvalidArgumentError(f"The given path is not a file: '{obj}'")

        # Create the artifact:
        artifact = Artifact(key=key, src_path=os.path.abspath(obj))

        return artifact, None

    @classmethod
    def pack_directory(cls, obj: str, key: str, archive_format: str = "zip"):
        pass

    @classmethod
    def unpack_file(
        cls,
        data_item: DataItem,
    ) -> str:
        file_path = data_item.local()

        cls.future_clear(path=file_path)

        return file_path

    @classmethod
    def unpack_directory(cls, data_item: DataItem, archive_format: str = None):
        archive_file = data_item.local()

        if archive_format is None:
            archive_formats = ArchiveFormat.get_formats()
            for archive_format in archive_formats:
                if archive_file.endswith(archive_format):
                    archive_format = archive_format
                    break
        if archive_format is None:
            raise MLRunInvalidArgumentError(
                f"Could not get the archive format of {data_item.key}."
            )

        archiver = ArchiveFormat.get_archiver(archive_format=archive_format)
        directory_path = archiver.extract_archive(
            archive_path=archive_file, output_path=os.path.dirname(archive_file)
        )

        cls.future_clear(path=archive_file)
        cls.future_clear(path=directory_path)

        return directory_path


# pathlib packagers:
class PathPackager(StrPackager):
    PACKABLE_OBJECT_TYPE = pathlib.Path

    @classmethod
    def pack_result(cls, obj: pathlib.Path, key: str) -> dict:
        return super().pack_result(obj=str(obj), key=key)

    @classmethod
    def pack_file(cls, obj: pathlib.Path, key: str):
        return super().pack_file(obj=str(obj), key=key)

    @classmethod
    def pack_directory(cls, obj: pathlib.Path, key: str, archive_format: str = "zip"):
        return super().pack_directory(
            obj=str(obj), key=key, archive_format=archive_format
        )
