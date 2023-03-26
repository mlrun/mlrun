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
import datetime
import os
import pathlib
import tarfile
import zipfile
from enum import Enum

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError

from ..constants import ArtifactTypes
from ..packager import Packager


# builtins packagers:
class IntPackager(Packager):
    TYPE = int
    DEFAULT_ARTIFACT_TYPE = ArtifactTypes.RESULT


class FloatPackager(Packager):
    TYPE = float
    DEFAULT_ARTIFACT_TYPE = ArtifactTypes.RESULT


class BytesPackager(Packager):
    TYPE = bytes
    DEFAULT_ARTIFACT_TYPE = ArtifactTypes.RESULT


class StrPackager(Packager):
    TYPE = str
    DEFAULT_ARTIFACT_TYPE = ArtifactTypes.RESULT

    class ArchiveFormats(Enum):
        ZIP = "zip"
        TAR_GZ = "tar.gz"

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
    def pack_directory(cls, obj: str, key: str, fmt: str = "zip"):
        pass

    @classmethod
    def unpack_file(
        cls,
        data_item: DataItem,
    ) -> str:
        return data_item.local()

    @classmethod
    def unpack_directory(cls, data_item: DataItem, fmt: str = None):
        archive_file = data_item.local()
        if fmt is None:
            archive_formats = [
                archive_format.value
                for archive_format in cls.ArchiveFormats.__members__.values()
            ]
            for archive_format in archive_formats:
                if archive_file.endswith(archive_format):
                    fmt = archive_format
                    break
        if fmt is None:
            raise MLRunInvalidArgumentError(
                f"Could not get the archive format of {data_item.key}."
            )


# pathlib packagers:
class PathPackager(StrPackager):
    TYPE = pathlib.Path
    DEFAULT_ARTIFACT_TYPE = ArtifactTypes.RESULT

    @classmethod
    def pack_result(cls, obj: pathlib.Path, key: str) -> dict:
        return super().pack_result(obj=str(obj), key=key)

    @classmethod
    def pack_file(cls, obj: pathlib.Path, key: str):
        return super().pack_file(obj=str(obj), key=key)

    @classmethod
    def pack_directory(cls, obj: pathlib.Path, key: str, fmt: str = "zip"):
        return super().pack_directory(obj=str(obj), key=key, fmt=fmt)
