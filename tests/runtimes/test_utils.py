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
import tempfile

import git
import pytest

import mlrun.runtimes.utils


@pytest.fixture
def repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = git.Repo.init(tmpdir)
        repo.create_remote("origin", "git@github.com:somewhere/else.git")

        # first commit
        tempfilename = "tempfile"
        open(f"{repo.working_dir}/{tempfilename}", "wb").close()
        repo.index.add([tempfilename])
        repo.index.commit("initialcommit")

        yield repo


def test_add_code_metadata_sanity(repo):
    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert (
        repo.remote("origin").url in code_metadata
    ), "code metadata should contain git info"
    assert (
        repo.head.commit.hexsha in code_metadata
    ), "commit hash should be in code metadata"


def test_add_code_metadata_stale_remote(repo):
    # simulating a malformed / stale remote that has no url attribute
    with open(f"{repo.git_dir}/config", "a") as f:
        f.write('[remote "stale"]\n')

    # origin is still there and valid, use that
    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert (
        repo.remote("origin").url in code_metadata
    ), "code metadata should contain git info"
    assert (
        repo.head.commit.hexsha in code_metadata
    ), "commit hash should be in code metadata"

    repo.delete_remote(repo.remote("origin"))

    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert code_metadata is None, "code metadata should be None as there is no remote"
