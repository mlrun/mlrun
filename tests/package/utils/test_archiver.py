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
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from mlrun.package.utils import ArchiveSupportedFormat


@pytest.mark.parametrize(
    "archive_format",
    ArchiveSupportedFormat.get_all_formats(),
)
@pytest.mark.parametrize(
    "directory_layout",
    [
        ["my_file.bin"],
        ["empty_dir"],
        ["a.bin", "b.bin"],
        ["inner_dir", os.path.join("inner_dir", "my_file.bin")],
        [
            "a.bin",
            "b.bin",
            "inner_dir",
            os.path.join("inner_dir", "my_file.bin"),
            os.path.join("inner_dir", "empty_dir"),
            "empty_dir",
        ],
    ],
)
def test_archiver(archive_format: str, directory_layout: List[str]):
    """
    Test the archivers for creating archives of multiple layouts and extracting them while keeping their original
    layout, names and data.

    :param archive_format:   The archive format to use.
    :param directory_layout: The layout to archive.
    """
    # Create a temporary directory for the test outputs:
    test_directory = tempfile.TemporaryDirectory()

    # Generate random array for the content of the files:
    files_content: np.ndarray = np.random.random(size=100)

    # Set up the main directory to archive and the output path for the archive file:
    directory_name = "my_dir"
    directory_path = Path(test_directory.name) / directory_name
    output_path = Path(test_directory.name) / "output_path"
    os.makedirs(directory_path)
    os.makedirs(output_path)

    # Create the files according to the layout provided:
    for path in directory_layout:
        full_path = directory_path / path
        if "." in path:
            files_content.tofile(full_path)
            assert full_path.is_file()
        else:
            os.makedirs(full_path)
            assert full_path.is_dir()
        assert full_path.exists()
    assert len(list(directory_path.rglob("*"))) == len(directory_layout)

    # Archive the files:
    archiver = ArchiveSupportedFormat.get_format_handler(fmt=archive_format)
    archive_path = Path(
        archiver.create_archive(
            directory_path=str(directory_path), output_path=str(output_path)
        )
    )
    assert archive_path.exists()
    assert archive_path == output_path / f"{directory_name}.{archive_format}"

    # Extract the files:
    extracted_dir_path = Path(
        archiver.extract_archive(
            archive_path=str(archive_path), output_path=str(output_path)
        )
    )
    assert extracted_dir_path.exists()
    assert extracted_dir_path == output_path / directory_name

    # Validate all files were extracted as they originally were:
    for path in directory_layout:
        full_path = extracted_dir_path / path
        assert full_path.exists()
        if "." in path:
            assert full_path.is_file()
            np.testing.assert_equal(np.fromfile(file=full_path), files_content)
        else:
            assert full_path.is_dir()
    assert len(list(extracted_dir_path.rglob("*"))) == len(directory_layout)

    # Clean the test outputs:
    test_directory.cleanup()
