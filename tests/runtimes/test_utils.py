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

import os
import tempfile
import unittest.mock

import deepdiff
import git
import pytest

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants
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
    "labels, labels_to_enrich, expected_labels, env_vars_to_mock, owner_to_enrich",
    [
        (
            {},
            None,
            {
                mlrun_constants.MLRunInternalLabels.owner: mlrun_constants.MLRunInternalLabels.v3io_user,
            },
            None,
            None,
        ),
        (
            {},
            None,
            {mlrun_constants.MLRunInternalLabels.owner: "test_user"},
            {"LOGNAME": "test_user", "V3IO_USERNAME": ""},
            None,
        ),
        (
            {},
            {},
            {},
            {"LOGNAME": "test_user", "V3IO_USERNAME": ""},
            None,
        ),
        (
            {mlrun_constants.MLRunInternalLabels.owner: "Mahatma"},
            None,
            {
                mlrun_constants.MLRunInternalLabels.owner: "Mahatma",
            },
            None,
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
            None,
        ),
        (
            {"a": "A", "b": "B"},
            {mlrun_constants.MLRunInternalLabels.owner},
            {
                "a": "A",
                "b": "B",
                mlrun_constants.MLRunInternalLabels.owner: mlrun_constants.MLRunInternalLabels.v3io_user,
            },
            None,
            None,
        ),
        (
            {"job-type": "workflow-runner"},
            None,
            {
                "job-type": "workflow-runner",
                mlrun_constants.MLRunInternalLabels.owner: "owner_user",
            },
            None,
            "owner_user",
        ),
        (
            {"job-type": "rerun-workflow-runner"},
            None,
            {
                "job-type": "rerun-workflow-runner",
                mlrun_constants.MLRunInternalLabels.owner: "owner_user",
            },
            None,
            "owner_user",
        ),
    ],
)
def test_enrich_run_labels(
    labels, labels_to_enrich, expected_labels, env_vars_to_mock, owner_to_enrich
):
    env_vars_to_mock = env_vars_to_mock or {
        "V3IO_USERNAME": mlrun_constants.MLRunInternalLabels.v3io_user,
    }
    with unittest.mock.patch.dict(
        os.environ,
        env_vars_to_mock,
    ):
        enriched_labels = mlrun.runtimes.utils.enrich_run_labels(
            labels, labels_to_enrich, owner_to_enrich=owner_to_enrich
        )
        assert (
            deepdiff.DeepDiff(
                enriched_labels,
                expected_labels,
                ignore_order=True,
            )
            == {}
        )


@pytest.mark.parametrize(
    "labels, env_vars, owner_to_enrich, expected_owner",
    [
        # No job-type, no owner_to_enrich, should use V3IO_USERNAME
        (
            {},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            None,
            "v3io_user",
        ),
        # No job-type, V3IO_USERNAME empty, fallback to getpass.getuser()
        (
            {},
            {"V3IO_USERNAME": "", "LOGNAME": "fallback_user"},
            None,
            "fallback_user",
        ),
        # job-type is workflow-runner, should use owner_to_enrich
        (
            {"job-type": mlrun_constants.JOB_TYPE_WORKFLOW_RUNNER},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            "owner_user",
            "owner_user",
        ),
        # job-type is rerun-workflow-runner, should use owner_to_enrich
        (
            {"job-type": mlrun_constants.JOB_TYPE_RERUN_WORKFLOW_RUNNER},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            "owner_user",
            "owner_user",
        ),
        # job-type is workflow-runner, but no owner_to_enrich, fallback to env
        (
            {"job-type": mlrun_constants.JOB_TYPE_WORKFLOW_RUNNER},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            None,
            "v3io_user",
        ),
    ],
)
def test_resolve_owner(labels, env_vars, owner_to_enrich, expected_owner):
    with unittest.mock.patch.dict(os.environ, env_vars, clear=True):
        with unittest.mock.patch("getpass.getuser", return_value=env_vars["LOGNAME"]):
            owner = mlrun.runtimes.utils.resolve_owner(labels, owner_to_enrich)
            assert owner == expected_owner


def test_results_to_iter_status_resolution(rundb_mock):
    """
    Test that results_to_iter correctly updates the execution state based on the results provided.
    Results objects contains result of each iteration, including their parameters and status.

    The test first simulates a scenario where one of the iteration fails and is pending a retry,
    then it simulates all iterations being successful.
    """
    results = [
        {
            "spec": {"parameters": {"p1": 2, "p2": 0}},
            "status": {
                "state": "pendingRetry",
                "error": "division by zero",
                "retry_count": None,
            },
        },
        {
            "spec": {"parameters": {"p1": 2, "p2": 1}},
            "status": {"state": "completed", "results": {"multiplier": 2.0}},
        },
        {
            "spec": {"parameters": {"p1": 2, "p2": 2}},
            "status": {"state": "completed", "results": {"multiplier": 1.0}},
        },
    ]
    run = {
        "kind": "run",
        "spec": {
            "log_level": "info",
            "parameters": {"p1": 2, "p2": 0},
            "handler": "my_function",
            "outputs": [],
            "output_path": "artifacts",
            "inputs": {},
            "notifications": [],
            "retry": {"count": 2, "backoff": {"base_delay": "30 sec"}},
            "data_stores": [],
        },
    }
    run = mlrun.run.RunObject.from_dict(run)

    execution = mlrun.execution.MLClientCtx.from_dict(
        run.to_dict(),
        rundb_mock,
        autocommit=False,
        is_api=True,
        store_run=False,
    )
    # Replace execution.commit with a no-op to avoid persisting changes during test
    execution.commit = lambda: None

    mlrun.runtimes.utils.results_to_iter(results, run, execution)
    assert execution.state == mlrun.common.runtimes.constants.RunStates.pending_retry

    # delete the failed result to simulate all iterations being successful
    results = results[1:]
    mlrun.runtimes.utils.results_to_iter(results, run, execution)
    assert execution.state == mlrun.common.runtimes.constants.RunStates.completed
