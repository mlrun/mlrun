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
"""Orca projects wire protocol, shared between MLRun's server-side Orca leader-proxy
(server/py/framework/utils/clients/iguazio/v4.py) and MLRun's SDK-direct-to-Orca project client
(mlrun/db/orca.py). Kept here rather than in either caller so neither implementation can drift
from Orca's actual contract.
"""

import typing
import uuid

import mlrun.common.schemas
import mlrun.errors

# Orca's project endpoints - reached via the same iguazio_api_url used for auth/token operations. The
# path is doubled ("projects/projects") because Orca's route registry combines the "projects"
# subdomain with that subdomain's own "/projects" resource path.
PROJECTS_ENDPOINT = "v1/projects/projects"
PROJECT_ENDPOINT_TEMPLATE = "v1/projects/projects/{name}"
ACTION_EXECUTIONS_ENDPOINT = "v1/trackable-actions/executions"

# The project-sync driver publishes a "sync-project" trackable action, keyed by op_id as its
# correlation_id, on the "projects" subdomain's ActionRunner - see orca SDK PR #1059.
PROJECT_SYNC_ACTION_TYPE = "sync-project"
PROJECT_SYNC_SUBDOMAIN = "projects"


class OrcaActionFailedError(Exception):
    """Raised when the sync-project trackable action for an op_id reaches a terminal 'failed' state."""


class ProjectMetadataLike(typing.Protocol):
    """Structural shape of a project's ``metadata`` this module's wire functions need."""

    name: str | None
    labels: dict | None
    annotations: dict | None


class ProjectSpecLike(typing.Protocol):
    """Structural shape of a project's ``spec`` this module's wire functions need."""

    owner: str | None
    description: str | None


class ProjectLike(typing.Protocol):
    """Structural shape ``create_project_wire``/``update_project_wire`` need. Satisfied by
    :class:`~mlrun.projects.MlrunProject`, :class:`mlrun.common.schemas.Project`, and the
    ``types.SimpleNamespace`` :func:`mlrun.db.orca._as_project_like` builds from a dict - the
    three shapes MLRun's own project CUD API already accepts interchangeably.
    """

    metadata: ProjectMetadataLike
    spec: ProjectSpecLike


def create_project_wire(project: ProjectLike) -> dict:
    """Build Orca's flat CreateProjectOptions body: name is required, everything else - owner
    included, Orca derives it from the authenticated caller when omitted - is optional.

    :param project: The project to create.
    :return: The JSON-serializable request body for Orca's ``POST /projects`` endpoint.
    """
    wire = {"name": project.metadata.name}
    if project.spec.owner:
        wire["owner"] = project.spec.owner
    if project.spec.description:
        wire["description"] = project.spec.description
    if project.metadata.labels:
        wire["labels"] = project.metadata.labels
    if project.metadata.annotations:
        wire["annotations"] = project.metadata.annotations
    return wire


def update_project_wire(project: ProjectLike, prev_op_id: uuid.UUID | None) -> dict:
    """Build Orca's flat UpdateProjectOptions body: prevOpId/owner are required by Orca's contract,
    so a missing value is sent through as-is and surfaces as a real validation error from Orca.

    :param project: The project's desired state to update to.
    :param prev_op_id: The CAS witness - the ``op_id`` last observed by the caller, or ``None``
        for a PUT that upserts a project that doesn't exist yet.
    :return: The JSON-serializable request body for Orca's ``PUT``/``PATCH /projects/{name}``.
    """
    wire = {
        "prevOpId": str(prev_op_id) if prev_op_id else None,
        "owner": project.spec.owner,
    }
    if project.spec.description:
        wire["description"] = project.spec.description
    if project.metadata.labels:
        wire["labels"] = project.metadata.labels
    if project.metadata.annotations:
        wire["annotations"] = project.metadata.annotations
    return wire


def project_from_wire(body: dict) -> mlrun.common.schemas.Project:
    """Parse an Orca project response body. Orca's wire format is camelCase (the SDK schemas
    camelize every field), so this can't just pydantic-validate the body directly - op_id/updated_at
    need explicit remapping.

    :param body: The parsed JSON body of an Orca project response.
    :return: The equivalent :class:`mlrun.common.schemas.Project`.
    """
    metadata = body.get("metadata", {})
    spec = body.get("spec", {})
    status = body.get("status", {})
    return mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(
            name=metadata["name"],
            labels=metadata.get("labels") or {},
            annotations=metadata.get("annotations") or {},
        ),
        spec=mlrun.common.schemas.ProjectSpec(
            owner=spec.get("owner"),
            description=spec.get("description"),
        ),
        status=mlrun.common.schemas.ProjectStatus(
            state=status.get("state"),
            op_id=status.get("opId"),
            updated_at=status.get("updatedAt"),
        ),
    )


def action_execution_query_params(op_id: uuid.UUID | str) -> dict:
    """Query params for ``GET .../trackable-actions/executions`` to find the sync-project action for
    ``op_id``.

    :param op_id: The operation id to filter the trackable-action execution by.
    :return: The query params for the request.
    """
    return {
        "correlationId": str(op_id),
        "actionType": PROJECT_SYNC_ACTION_TYPE,
        "subdomain": PROJECT_SYNC_SUBDOMAIN,
        "limit": 1,
    }


def verify_action_execution_terminal(
    body: dict, name: str, op_id: uuid.UUID | str
) -> None:
    """Interpret a trackable-action executions response body for one sync-project op_id.

    Raises :class:`OrcaActionFailedError` if the action failed. Raises
    :class:`mlrun.errors.MLRunRuntimeError` if it hasn't reached a terminal state yet (including "not
    observed yet") - callers drive the retry/poll loop and treat that as still-in-progress, not
    failure.

    :param body: The parsed JSON body of a ``GET .../trackable-actions/executions`` response.
    :param name: The project name, for error messages only.
    :param op_id: The operation id being awaited, for error messages only.
    """
    items = body.get("items", [])
    if not items:
        raise mlrun.errors.MLRunRuntimeError(
            f"No Orca sync-project action observed yet for project {name} (op_id={op_id})"
        )
    state = items[0].get("status", {}).get("state")
    if state == "failed":
        raise OrcaActionFailedError(
            f"Orca sync-project action for project {name} (op_id={op_id}) failed"
        )
    if state != "succeeded":
        raise mlrun.errors.MLRunRuntimeError(
            f"Orca sync-project action for project {name} (op_id={op_id}) is still in "
            f"progress (state={state})"
        )
