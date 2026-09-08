# Copyright 2026 Iguazio
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
import uuid

import pytest

import mlrun.common.schemas
import mlrun.errors
import mlrun.projects.project
import mlrun.utils.orca_projects as orca_projects


def _project(owner=None, description=None, labels=None, annotations=None):
    return mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(
            name="p1", labels=labels, annotations=annotations
        ),
        spec=mlrun.common.schemas.ProjectSpec(owner=owner, description=description),
    )


def test_create_project_wire_minimal():
    assert orca_projects.create_project_wire(_project()) == {"name": "p1"}


def test_create_project_wire_does_not_leak_mlrun_only_spec_fields():
    # a real MlrunProject carries ~20 MLRun-specific spec fields (functions, artifacts,
    # workflows, build, ...) that aren't part of Orca's contract (the common set only:
    # name/labels/annotations/owner/description). create_project_wire/update_project_wire only
    # ever read the common-set attributes by name, so passing the richer MlrunProject object
    # directly (as mlrun/db/orca.py does) must never leak those extra fields onto the wire.
    project = mlrun.projects.project.MlrunProject(
        metadata=mlrun.projects.project.ProjectMetadata(name="p1"),
        spec=mlrun.projects.project.ProjectSpec(
            owner="jsmith",
            goals="win",
            params={"a": "b"},
            source="git://example.com/repo.git",
        ),
    )
    wire = orca_projects.create_project_wire(project)
    assert wire == {"name": "p1", "owner": "jsmith"}


def test_create_project_wire_full():
    project = _project(
        owner="jsmith",
        description="desc",
        labels={"a": "b"},
        annotations={"c": "d"},
    )
    assert orca_projects.create_project_wire(project) == {
        "name": "p1",
        "owner": "jsmith",
        "description": "desc",
        "labels": {"a": "b"},
        "annotations": {"c": "d"},
    }


@pytest.mark.parametrize("prev_op_id", [uuid.uuid4(), None])
def test_update_project_wire_prev_op_id(prev_op_id):
    project = _project(owner="jsmith")
    wire = orca_projects.update_project_wire(project, prev_op_id)
    assert wire["prevOpId"] == (str(prev_op_id) if prev_op_id else None)
    assert wire["owner"] == "jsmith"


def test_project_from_wire_round_trip():
    op_id = uuid.uuid4()
    body = {
        "metadata": {"name": "p1", "labels": {"a": "b"}, "annotations": {"c": "d"}},
        "spec": {"owner": "jsmith", "description": "desc"},
        "status": {"state": "online", "opId": str(op_id), "updatedAt": None},
    }
    project = orca_projects.project_from_wire(body)
    assert project.metadata.name == "p1"
    assert project.metadata.labels == {"a": "b"}
    assert project.metadata.annotations == {"c": "d"}
    assert project.spec.owner == "jsmith"
    assert project.spec.description == "desc"
    assert project.status.state == "online"
    assert project.status.op_id == op_id


def test_action_execution_query_params():
    op_id = uuid.uuid4()
    assert orca_projects.action_execution_query_params(op_id) == {
        "correlationId": str(op_id),
        "actionType": "sync-project",
        "subdomain": "projects",
        "limit": 1,
    }


def test_verify_action_execution_terminal_succeeded():
    # no exception raised
    orca_projects.verify_action_execution_terminal(
        {"items": [{"status": {"state": "succeeded"}}]}, "p1", "op-id"
    )


def test_verify_action_execution_terminal_failed_raises_fatal():
    with pytest.raises(orca_projects.OrcaActionFailedError):
        orca_projects.verify_action_execution_terminal(
            {"items": [{"status": {"state": "failed"}}]}, "p1", "op-id"
        )


@pytest.mark.parametrize(
    "body",
    [
        {"items": [{"status": {"state": "running"}}]},
        {"items": []},
    ],
)
def test_verify_action_execution_terminal_not_yet_terminal_raises_retryable(body):
    # not-observed-yet and still-in-progress are both retryable (not fatal like "failed"),
    # so callers driving a poll loop must be able to distinguish them from OrcaActionFailedError.
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        orca_projects.verify_action_execution_terminal(body, "p1", "op-id")
