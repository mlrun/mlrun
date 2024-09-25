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
import unittest.mock

import deepdiff
import git
import pytest

import mlrun.common.constants as mlrun_constants
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


@pytest.mark.parametrize(
    "labels, labels_to_enrich, expected_labels, env_vars_to_mock",
    [
        (
            {},
            None,
            {
                mlrun_constants.MLRunInternalLabels.owner: mlrun_constants.MLRunInternalLabels.v3io_user,
                mlrun_constants.MLRunInternalLabels.v3io_user: mlrun_constants.MLRunInternalLabels.v3io_user,
            },
            None,
        ),
        (
            {},
            {},
            {mlrun_constants.MLRunInternalLabels.owner: "test_user"},
            {"LOGNAME": "test_user", "V3IO_USERNAME": ""},
        ),
        (
            {mlrun_constants.MLRunInternalLabels.owner: "Mahatma"},
            {},
            {
                mlrun_constants.MLRunInternalLabels.owner: "Mahatma",
                mlrun_constants.MLRunInternalLabels.v3io_user: mlrun_constants.MLRunInternalLabels.v3io_user,
            },
            None,
        ),
        (
            {
                mlrun_constants.MLRunInternalLabels.owner: "Mahatma",
                mlrun_constants.MLRunInternalLabels.v3io_user: "Gandhi",
            },
            {},
            {
                mlrun_constants.MLRunInternalLabels.owner: "Mahatma",
                mlrun_constants.MLRunInternalLabels.v3io_user: "Gandhi",
            },
            None,
        ),
        (
            {"a": "A", "b": "B"},
            {mlrun.common.runtimes.constants.RunLabels.owner},
            {
                "a": "A",
                "b": "B",
                mlrun_constants.MLRunInternalLabels.owner: mlrun_constants.MLRunInternalLabels.v3io_user,
            },
            None,
        ),
    ],
)
def test_enrich_run_labels(labels, labels_to_enrich, expected_labels, env_vars_to_mock):
    env_vars_to_mock = env_vars_to_mock or {
        "V3IO_USERNAME": mlrun_constants.MLRunInternalLabels.v3io_user,
    }
    with unittest.mock.patch.dict(
        os.environ,
        env_vars_to_mock,
    ):
        enriched_labels = mlrun.runtimes.utils.enrich_run_labels(
            labels, labels_to_enrich
        )
        assert (
            deepdiff.DeepDiff(
                enriched_labels,
                expected_labels,
                ignore_order=True,
            )
            == {}
        )
