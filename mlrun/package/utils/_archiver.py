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
import tarfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import mlrun.utils

from ._supported_format import SupportedFormat


class _Archiver(ABC):
    """
    An abstract base class for an archiver - a class to manage archives of multiple files.
    """

    @classmethod
    @abstractmethod
    def create_archive(cls, directory_path: str, output_path: str) -> str:
        """
        Create an archive of all the contents in the given directory and save it to an archive file named as the
        directory in the provided output path.

        :param directory_path: The directory with the files to archive.
        :param output_path:    The output path to store the created archive file.

        :return: The created archive path.
        """
        pass

    @classmethod
    @abstractmethod
    def extract_archive(cls, archive_path: str, output_path: str) -> str:
        """
        Extract the given archive to a directory named as the archive file (without the extension) located in the
        provided output path.

        :param archive_path: The archive file to extract its contents.
        :param output_path:  The output path to extract the directory of the archive to.

        :return: The extracted contents directory path.
        """
        pass


class _ZipArchiver(_Archiver):
    """
    A static class for managing zip archives.
    """

    @classmethod
    def create_archive(cls, directory_path: str, output_path: str) -> str:
        """
        Create an archive of all the contents in the given directory and save it to an archive file named as the
        directory in the provided output path.

        :param directory_path: The directory with the files to archive.
        :param output_path:    The output path to store the created archive file.

        :return: The created archive path.
        """
        # Convert to `pathlib.Path` objects:
        directory_path = Path(directory_path)
        output_path = Path(output_path)

        # Construct the archive file path:
        archive_path = output_path / f"{directory_path.stem}.zip"

        # Archive:
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            for path in directory_path.rglob("*"):
                zip_file.write(filename=path, arcname=path.relative_to(directory_path))

        return str(archive_path)

    @classmethod
    def extract_archive(cls, archive_path: str, output_path: str) -> str:
        """
        Extract the given archive to a directory named as the archive file (without the extension) located in the
        provided output path.

        :param archive_path: The archive file to extract its contents.
        :param output_path:  The output path to extract the directory of the archive to.

        :return: The extracted contents directory path.
        """
        # Convert to `pathlib.Path` objects:
        archive_path = Path(archive_path)
        output_path = Path(output_path)

        # Create the directory path, add timestamp to avoid collisions:
        directory_path = output_path / archive_path.stem
        directory_path = directory_path.with_name(
            f"{directory_path.name}-{mlrun.utils.now_date().isoformat()}"
        )
        os.makedirs(directory_path)

        # Extract:
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(directory_path)

        return str(directory_path)


class _TarArchiver(_Archiver):
    """
    A static class for managing tar archives.
    """

    # Inner class variable to note how to open a `TarFile` object for reading and writing:
    _MODE_STRING = ""

    @classmethod
    def create_archive(cls, directory_path: str, output_path: str) -> str:
        """
        Create an archive of all the contents in the given directory and save it to an archive file named as the
        directory in the provided output path.

        :param directory_path: The directory with the files to archive.
        :param output_path:    The output path to store the created archive file.

        :return: The created archive path.
        """
        # Convert to `pathlib.Path` objects:
        directory_path = Path(directory_path)
        output_path = Path(output_path)

        # Construct the archive file path:
        archive_file_extension = (
            "tar" if cls._MODE_STRING == "" else f"tar.{cls._MODE_STRING}"
        )
        archive_path = output_path / f"{directory_path.stem}.{archive_file_extension}"

        # Archive:
        with tarfile.open(archive_path, f"w:{cls._MODE_STRING}") as tar_file:
            for path in directory_path.rglob("*"):
                tar_file.add(name=path, arcname=path.relative_to(directory_path))

        return str(archive_path)

    @classmethod
    def extract_archive(cls, archive_path: str, output_path: str) -> str:
        """
        Extract the given archive to a directory named as the archive file (without the extension) located in the
        provided output path.

        :param archive_path: The archive file to extract its contents.
        :param output_path:  The output path to extract the directory of the archive to.

        :return: The extracted contents directory path.
        """
        # Convert to `pathlib.Path` objects:
        archive_path = Path(archive_path)
        output_path = Path(output_path)

        # Get the archive file name (can be constructed of multiple extensions like tar.gz so `Path.stem` won't work):
        archive_file_name = archive_path
        while archive_file_name.with_suffix(suffix="") != archive_file_name:
            archive_file_name = archive_file_name.with_suffix(suffix="")
        archive_file_name = archive_file_name.stem

        # Create the directory path:
        directory_path = output_path / archive_file_name
        os.makedirs(directory_path)

        # Extract:
        with tarfile.open(archive_path, f"r:{cls._MODE_STRING}") as tar_file:
            # use 'data' to ensure no security risks are imposed by the archive files
            # see: https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
            tar_file.extractall(directory_path, filter="data")

        return str(directory_path)


class _TarGZArchiver(_TarArchiver):
    """
    A static class for managing tar.gz archives.
    """

    # Inner class variable to note how to open a `TarFile` object for reading and writing:
    _MODE_STRING = "gz"


class _TarBZ2Archiver(_TarArchiver):
    """
    A static class for managing tar.bz2 archives.
    """

    # Inner class variable to note how to open a `TarFile` object for reading and writing:
    _MODE_STRING = "bz2"


class _TarXZArchiver(_TarArchiver):
    """
    A static class for managing tar.gz archives.
    """

    # Inner class variable to note how to open a `TarFile` object for reading and writing:
    _MODE_STRING = "xz"


class ArchiveSupportedFormat(SupportedFormat[_Archiver]):
    """
    Library of archive formats (file extensions) supported by some builtin MLRun packagers.
    """

    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    TAR_XZ = "tar.xz"

    _FORMAT_HANDLERS_MAP = {
        ZIP: _ZipArchiver,
        TAR: _TarArchiver,
        TAR_GZ: _TarGZArchiver,
        TAR_BZ2: _TarBZ2Archiver,
        TAR_XZ: _TarXZArchiver,
    }
