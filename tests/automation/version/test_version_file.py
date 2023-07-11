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
import json
import os
import subprocess

import packaging.version
import pytest

from automation.version.version_file import (
    create_or_update_version_file,
    get_current_version,
    is_stable_version,
    resolve_next_version,
)


@pytest.fixture
def git_repo(tmpdir, request):
    # change working directory to tmpdir
    os.chdir(tmpdir)

    # set up git repository
    subprocess.run(["git", "init"])
    if hasattr(request, "param"):
        subprocess.run(["git", "checkout", "-b", request.param["branch"]])

    # add commits
    for i in range(5):
        with open(f"file{i}.txt", "w") as f:
            f.write(f"test {i}\n")
        subprocess.run(["git", "add", f"file{i}.txt"])
        subprocess.run(["git", "commit", "-m", f"test commit {i}"])

    return tmpdir


# tags structure:
# list of tuples, where each tuple is (commit order (0-index, where 0 is latest), tag name)
@pytest.mark.parametrize(
    "base_version,tags,expected_current_version",
    [
        # no tags were made, default to base_version
        ("1.5.0", [], "1.5.0"),
        # tags were made, but none of them are similar to base_version, use latest greatest (< base version)
        (
            "1.5.0",
            [
                (0, "1.4.0"),
                (1, "1.3.0"),
            ],
            "1.4.0",
        ),
        # tags were made, but none of them are similar to base_version, use latest greatest (> base version)
        (
            "1.5.0",
            [
                (1, "1.6.0"),
                (2, "1.4.0"),
            ],
            "1.6.0",
        ),
        # tags were made, similar to base_version, use latest greatest
        (
            "1.5.0",
            [
                (1, "1.5.0"),
            ],
            "1.5.0",
        ),
        # tags were made, similar to base_version, use latest greatest
        (
            "1.5.0",
            [
                (1, "1.5.1"),
            ],
            "1.5.1",
        ),
    ],
)
def test_current_version(git_repo, base_version, tags, expected_current_version):
    for tag in tags:
        subprocess.run(
            [
                "git",
                "tag",
                "-a",
                "-m",
                f"test tag {tag[1]}",
                f"v{tag[1]}",
                f"HEAD~{tag[0]}",
            ]
        )
    current_version = get_current_version(base_version=packaging.version.parse("1.5.0"))
    assert current_version == expected_current_version


@pytest.mark.parametrize(
    "bump_type,current_version,base_version,feature_name,expected_next_version",
    [
        # current version is olden than current base version,
        # the next expected version is derived from the base version
        ("rc", "1.0.0", "1.1.0", None, "1.1.0-rc1"),
        ("rc-grad", "1.0.0", "1.1.0", None, "1.1.0"),
        ("patch", "1.0.0", "1.1.0", None, "1.1.1"),
        ("minor", "1.0.0", "1.1.0", None, "1.2.0"),
        ("major", "1.0.0", "1.1.0", None, "2.0.0"),
        # current+base tagged
        ("rc", "1.0.0", "1.0.0", None, "1.0.1-rc1"),
        ("rc", "1.0.0", "1.0.0", "ft-test", "1.0.1-rc1+ft-test"),
        ("rc", "1.0.0-rc1", "1.0.0", None, "1.0.0-rc2"),
        ("rc", "1.0.0-rc1", "1.0.0", "ft-test", "1.0.0-rc2+ft-test"),
        ("rc-grad", "1.0.0-rc1", "1.0.0", None, "1.0.0"),
        ("rc-grad", "1.0.0-rc1", "1.0.0", "ft-test", "1.0.0-rc1+ft-test"),
        ("patch", "1.0.0", "1.0.0", None, "1.0.1"),
        ("patch", "1.0.0", "1.0.0", "ft-test", "1.0.1-rc1+ft-test"),
        ("patch", "1.0.0-rc1", "1.0.0", None, "1.0.1"),
        ("patch", "1.0.0-rc1", "1.0.0", "ft-test", "1.0.1-rc1+ft-test"),
        ("minor", "1.0.0", "1.0.0", None, "1.1.0"),
        ("minor", "1.0.0", "1.0.0", "ft-test", "1.1.0-rc1+ft-test"),
        ("minor", "1.0.0-rc1", "1.0.0", None, "1.1.0"),
        ("minor", "1.0.0-rc1", "1.0.0", "ft-test", "1.1.0-rc1+ft-test"),
        ("major", "1.0.0", "1.0.0", None, "2.0.0"),
        ("major", "1.0.0", "1.0.0", "ft-test", "2.0.0-rc1+ft-test"),
        ("major", "1.0.0-rc1", "1.0.0", None, "2.0.0"),
        ("major", "1.0.0-rc1", "1.0.0", "ft-test", "2.0.0-rc1+ft-test"),
    ],
)
def test_next_version(
    bump_type, current_version, base_version, feature_name, expected_next_version
):
    next_version = resolve_next_version(
        bump_type,
        packaging.version.parse(current_version),
        packaging.version.parse(base_version),
        feature_name,
    )
    assert (
        next_version == expected_next_version
    ), f"expected {expected_next_version}, got {next_version}"


@pytest.mark.parametrize(
    "git_repo,base_version,expected_version",
    [
        (
            {"branch": "development"},
            "1.5.0",
            "1.5.0",
        ),
        (
            # fills feature from branch
            {"branch": "feature/something"},
            "1.5.0",
            "1.5.0+something",
        ),
    ],
    indirect=["git_repo"],
)
def test_create_or_update_version_file(git_repo, base_version, expected_version):
    latest_commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE
    )
    create_or_update_version_file(base_version, git_repo / "version.json")
    with open(git_repo / "version.json") as f:
        version = json.loads(f.read())
    assert version == {
        "version": expected_version,
        "git_commit": latest_commit_hash.stdout.strip().decode(),
    }


@pytest.mark.parametrize(
    "version,expected_is_stable",
    [
        ("1.0.0", True),
        ("1.0.0-rc1", False),
        ("1.0.0+unstable", False),
        ("1.0.0-rc1+ft-test", False),
    ],
)
def test_is_stable_version(version: str, expected_is_stable: bool):
    assert is_stable_version(version) is expected_is_stable
