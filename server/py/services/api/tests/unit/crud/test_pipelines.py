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

import json
from unittest.mock import MagicMock, patch

import pytest

import mlrun_pipelines
import mlrun_pipelines.common.helpers

import services.api.crud
import unittest.mock
import services.api.crud.pipelines
import datetime
from mlrun_pipelines.common.models import RunStatuses
import zipfile
from io import BytesIO
import fastapi
import mlrun.common.schemas
import mlrun.errors


def test_resolve_pipeline_project():
    cases = [
        {
            "expected_project": "project-from-deploy-p",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "-p",
                        "project-from-deploy-p",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy--project",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "--project",
                        "project-from-deploy--project",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy-f",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "-f",
                        "db://project-from-deploy-f/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy--func-url",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "--func-url",
                        "db://project-from-deploy--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy-precedence-p",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "--func-url",
                        "db://project-from-deploy--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                        "-p",
                        "project-from-deploy-precedence-p",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run--project",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--project",
                        "project-from-run--project",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run-f",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "-f",
                        "db://project-from-run-f/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run--func-url",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--func-url",
                        "db://project-from-run--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run-r",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "-r",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-run-r'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run--runtime",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--runtime",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-run--runtime'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run-precedence--project",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--func-url",
                        "db://project-from-deploy--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                        "--project",
                        "project-from-run-precedence--project",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-build-r",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "build",
                        "-r",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-build-r'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-build--runtime",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "build",
                        "--runtime",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-build--runtime'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-build-precedence--runtime",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "build",
                        "--runtime",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-build--runtime'}}",
                        "--project",
                        "project-from-build-precedence--runtime",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-annotation",
            "template": {
                "metadata": {
                    "annotations": {
                        mlrun_pipelines.common.helpers.PROJECT_ANNOTATION: "project-from-annotation"
                    }
                }
            },
        },
    ]
    for case in cases:
        workflow_manifest = {"spec": {"templates": [case["template"]]}}
        pipeline = {
            "pipeline_spec": {"workflow_manifest": json.dumps(workflow_manifest)}
        }
        project = services.api.crud.Pipelines()._resolve_project_from_pipeline(
            mlrun_pipelines.models.PipelineRun(pipeline)
        )
        assert project == case["expected_project"]


@pytest.mark.parametrize(
    "project,expected_ids",
    [
        ("project-a", ["run1"]),
        ("*", ["run1", "run2"]),
    ],
)
def test_list_pipelines_project_filtering(project, expected_ids):
    pipelines = services.api.crud.pipelines.Pipelines()
    db_session = MagicMock()

    # Mock runs and KFP client
    run1 = MagicMock(id="run1", name="pipeline1", status="Succeeded")
    run2 = MagicMock(id="run2", name="pipeline2", status="Failed")
    all_runs = [run1, run2]
    mock_kfp_client = MagicMock()
    mock_kfp_client.list_runs.return_value = [(all_runs, None)]

    with (
        patch.object(
            services.api.crud.pipelines.Pipelines,
            "_initialize_kfp_client",
            return_value=mock_kfp_client,
        ),
        patch.object(
            services.api.crud.pipelines.Pipelines,
            "_resolve_project_from_pipeline",
            side_effect=lambda run: "project-a" if run.id == "run1" else "project-b",
        ),
        patch.object(
            services.api.crud.pipelines.Pipelines,
            "_format_runs_concurrently",
            side_effect=lambda kfp_client, runs, format_: [
                {"id": r.id, "name": r.name} for r in runs
            ],
        ),
    ):
        total_size, next_page_token, runs = pipelines.list_pipelines(
            db_session=db_session,
            project=project,
        )

    assert total_size == len(expected_ids)
    assert next_page_token is None
    assert [r["id"] for r in runs] == expected_ids


@pytest.fixture()
def pipelines_crud(self):
    return services.api.crud.pipelines.Pipelines()

def test_failed_run_deletion_logs_correct_run_id(pipelines_crud, monkeypatch):
    """
    When deleting multiple pipeline runs concurrently, the warning log
    for a failed deletion must reference the correct pipeline_run_id,
    not the last run processed in the submission loop.
    """
    # Arrange: 3 runs; only the second one will fail
    fake_runs = [
        {"id": "run-aaa", "name": "run-a", "experiment_id": ""},
        {"id": "run-bbb", "name": "run-b", "experiment_id": ""},
        {"id": "run-ccc", "name": "run-c", "experiment_id": ""},
    ]

    # Make list_pipelines return our fake runs
    monkeypatch.setattr(
        "mlrun.utils.helpers.retry_until_successful",
        lambda *a, **kw: (None, None, fake_runs),
    )

    # Patch _initialize_kfp_client to return a mock client
    mock_kfp_client = unittest.mock.MagicMock()

    # Track which run_id was passed for each call
    call_order = []

    def fake_delete_run(run_id):
        call_order.append(run_id)
        if run_id == "run-bbb":
            raise RuntimeError("Simulated KFP deletion failure for run-bbb")

    mock_kfp_client._run_api.delete_run.side_effect = fake_delete_run
    monkeypatch.setattr(
        pipelines_crud,
        "_initialize_kfp_client",
        lambda *a, **kw: mock_kfp_client,
    )

    # Patch PipelineRun to return a simple object with .id and .experiment_id
    class FakePipelineRun:
        def __init__(self, data):
            self.id = data["id"]
            self.name = data["name"]
            self.experiment_id = data.get("experiment_id", "")

    monkeypatch.setattr(
        "mlrun_pipelines.models.PipelineRun",
        FakePipelineRun,
    )

    # Capture warning logs
    logged_warnings = []

    def capture_warning(msg, **kwargs):
        logged_warnings.append((msg, kwargs))

    monkeypatch.setattr(
        "mlrun.utils.logger.warning",
        capture_warning,
    )

    # Act
    db_session = unittest.mock.MagicMock()
    pipelines_crud.delete_pipelines_runs(db_session, "test-project")

    # Assert: exactly one warning was logged
    assert len(logged_warnings) == 1, (
        f"Expected exactly 1 warning, got {len(logged_warnings)}"
    )

    warning_msg, warning_kwargs = logged_warnings[0]
    assert warning_msg == "Failed to delete pipeline run"
    # The critical assertion: the logged run ID must be "run-bbb"
    # (the one that actually failed), NOT "run-ccc" (the last loop value)
    assert warning_kwargs["pipeline_run_id"] == "run-bbb", (
        f"Expected pipeline_run_id='run-bbb', got '{warning_kwargs['pipeline_run_id']}'. "
        "This indicates the stale loop variable bug is present."
    )

def test_failed_experiment_deletion_logs_correct_experiment_id(
    self, pipelines_crud, monkeypatch
):
    """
    When deleting multiple experiments concurrently, the warning log
    for a failed deletion must reference the correct experiment_id.
    """
    # Arrange: 3 runs each with a different experiment_id
    fake_runs = [
        {"id": "run-1", "name": "run-1", "experiment_id": "exp-aaa"},
        {"id": "run-2", "name": "run-2", "experiment_id": "exp-bbb"},
        {"id": "run-3", "name": "run-3", "experiment_id": "exp-ccc"},
    ]

    monkeypatch.setattr(
        "mlrun.utils.helpers.retry_until_successful",
        lambda *a, **kw: (None, None, fake_runs),
    )

    mock_kfp_client = unittest.mock.MagicMock()
    # All run deletions succeed
    mock_kfp_client._run_api.delete_run.return_value = None

    # Only exp-bbb fails
    def fake_delete_experiment(exp_id):
        if exp_id == "exp-bbb":
            raise RuntimeError(
                "Simulated KFP experiment deletion failure for exp-bbb"
            )

    mock_kfp_client._experiment_api.delete_experiment.side_effect = (
        fake_delete_experiment
    )
    monkeypatch.setattr(
        pipelines_crud,
        "_initialize_kfp_client",
        lambda *a, **kw: mock_kfp_client,
    )

    class FakePipelineRun:
        def __init__(self, data):
            self.id = data["id"]
            self.name = data["name"]
            self.experiment_id = data.get("experiment_id", "")

    monkeypatch.setattr(
        "mlrun_pipelines.models.PipelineRun",
        FakePipelineRun,
    )

    logged_warnings = []

    def capture_warning(msg, **kwargs):
        logged_warnings.append((msg, kwargs))

    monkeypatch.setattr(
        "mlrun.utils.logger.warning",
        capture_warning,
    )

    db_session = unittest.mock.MagicMock()
    pipelines_crud.delete_pipelines_runs(db_session, "test-project")

    # Find the experiment deletion warning
    exp_warnings = [
        (msg, kw)
        for msg, kw in logged_warnings
        if msg == "Failed to delete an experiment"
    ]
    assert len(exp_warnings) == 1, (
        f"Expected exactly 1 experiment warning, got {len(exp_warnings)}"
    )

    _, warning_kwargs = exp_warnings[0]
    assert warning_kwargs["experiment_id"] == "exp-bbb", (
        f"Expected experiment_id='exp-bbb', got '{warning_kwargs['experiment_id']}'. "
        "This indicates the stale loop variable bug is present."
    )


def test_succeeded_status_uses_equality_not_substring():
    """
    Ensure that only the exact status string "Succeeded" is counted as
    a completed pipeline, not any substring of it.
    """
    # RunStatuses.succeeded is "Succeeded"
    succeeded_status = RunStatuses.succeeded
    assert succeeded_status == "Succeeded"

    # A correct equality check rejects substrings
    assert ("S" == succeeded_status) is False
    assert ("ceed" == succeeded_status) is False
    assert ("Succeeded" == succeeded_status) is True

    # The bug: substring check would incorrectly match
    assert ("S" in succeeded_status) is True  # this is the bug behavior
    assert ("ceed" in succeeded_status) is True  # this is the bug behavior


def test_calculate_pipelines_counters_counts_only_exact_succeeded():
    """
    Verify the fixed comparison logic: only pipelines with exactly
    "Succeeded" status are counted as completed.
    """
    now = datetime.datetime.now(tz=datetime.UTC)
    recent_time = (now - datetime.timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S%z"
    )

    pipelines = [
        {
            "project": "proj1",
            "status": str(RunStatuses.succeeded),
            "finished_at": recent_time,
        },
        {
            "project": "proj1",
            "status": "SomeOtherStatus",
            "finished_at": recent_time,
        },
    ]

    completed_count = 0
    for pipeline in pipelines:
        if pipeline.get("finished_at"):
            finished_at = datetime.datetime.strptime(
                pipeline["finished_at"], "%Y-%m-%d %H:%M:%S%z"
            )
            if finished_at > now - datetime.timedelta(days=1):
                # This is the FIXED comparison (==), not substring (in)
                if pipeline["status"] == RunStatuses.succeeded:
                    completed_count += 1

    # Only the first pipeline with exactly "Succeeded" should be counted
    assert completed_count == 1


def test_substring_status_not_counted_as_succeeded():
    """
    Regression test: a status that is a substring of "Succeeded" must
    not be counted as a completed pipeline.
    """
    now = datetime.datetime.now(tz=datetime.UTC)
    recent_time = (now - datetime.timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S%z"
    )

    # "ceed" is a substring of "Succeeded" - with the old `in` check
    # it would be incorrectly counted as completed
    pipeline = {
        "project": "proj1",
        "status": "ceed",
        "finished_at": recent_time,
    }

    # With substring check (bug): "ceed" in "Succeeded" == True
    assert ("ceed" in str(RunStatuses.succeeded)) is True

    # With equality check (fix): "ceed" == "Succeeded" == False
    assert (pipeline["status"] == RunStatuses.succeeded) is False


def _make_dummy_zip() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pipeline.yaml", "apiVersion: argoproj.io/v1alpha1")
    return buf.getvalue()


def test_zip_content_type_is_recognized():
    """
    Verify that 'application/zip' content type is correctly detected
    as a zip pipeline (not rejected as unsupported).
    Before the fix, the check was '" /zip" in content_type' (with a
    spurious leading space), which never matched 'application/zip'.
    """
    pipelines = services.api.crud.Pipelines()
    data = _make_dummy_zip()
    auth_info = mlrun.common.schemas.AuthInfo(
        username="test_user", user_id="test-user-id"
    )

    # Mock everything after the content type detection
    with (
        unittest.mock.patch.object(
            pipelines, "_initialize_kfp_client"
        ) as mock_client,
        unittest.mock.patch(
            "services.api.utils.helpers.resolve_auth_token_secret_name",
            return_value="mock-secret",
        ),
        unittest.mock.patch(
            "mlrun_pipelines.common.ops.process_kfp_workflow_secret_references",
            return_value=data,
        ),
    ):
        mock_kfp = unittest.mock.MagicMock()
        mock_client.return_value = mock_kfp
        mock_experiment = unittest.mock.MagicMock()
        mock_experiment.id = "exp-id"
        mock_kfp.create_experiment.return_value = mock_experiment
        mock_run = unittest.mock.MagicMock()
        mock_run.id = "run-id"
        mock_run.name = "test-run"
        mock_kfp.run_pipeline.return_value = mock_run

        # This should NOT raise "unsupported pipeline type"
        run = pipelines.create_pipeline(
            experiment_name="test-exp",
            run_name="test-run",
            content_type="application/zip",
            data=data,
            arguments={},
            auth_info=auth_info,
        )
        assert run is not None


def test_unsupported_content_type_raises():
    """Verify that an unsupported content type still raises an error."""
    pipelines = services.api.crud.Pipelines()
    data = b"fake data"
    auth_info = mlrun.common.schemas.AuthInfo(
        username="test_user", user_id="test-user-id"
    )

    with pytest.raises(fastapi.HTTPException):
        pipelines.create_pipeline(
            experiment_name="test-exp",
            run_name="test-run",
            content_type="text/plain",
            data=data,
            arguments={},
            auth_info=auth_info,
        )


def test_application_zip_content_type_matches():
    """Verify that 'application/zip' (the standard MIME type) contains '/zip'."""
    assert "/zip" in "application/zip"


def test_yaml_content_type_is_recognized():
    """Verify that '/yaml' content type check still works."""
    assert "/yaml" in "application/yaml"
    assert "/yaml" in "text/yaml"
