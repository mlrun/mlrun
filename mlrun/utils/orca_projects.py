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
import mlrun.common.types
import mlrun.errors

# Orca's project endpoints - reached via the same iguazio_api_url used for auth/token operations. The
# path is doubled ("projects/projects") because Orca's route registry combines the "projects"
# subdomain with that subdomain's own "/projects" resource path.
PROJECTS_ENDPOINT = "v1/projects/projects"
PROJECT_ENDPOINT_TEMPLATE = "v1/projects/projects/{name}"
ACTION_EXECUTIONS_ENDPOINT = "v1/trackable-actions/executions"

# The project-sync driver publishes a "sync-project" trackable action, keyed by op_id as its
# correlation_id, on the "projects" subdomain's ActionRunner.
PROJECT_SYNC_ACTION_TYPE = "sync-project"
PROJECT_SYNC_SUBDOMAIN = "projects"


class ActionExecutionState(mlrun.common.types.StrEnum):
    """Lifecycle state of one Orca Trackable Action execution - mirrors Orca's own
    ``ActionExecutionState`` (``backend/subdomains/trackableactions/types/actionexecution.go``).
    """

    created = "created"
    dispatched = "dispatched"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class OrcaActionFailedError(Exception):
    """Raised when the sync-project trackable action for an op_id reaches a terminal 'failed' state."""


class ProjectMetadataLike(typing.Protocol):
    """Structural shape of a project's ``metadata`` :func:`resolve_project_body` needs."""

    name: str | None
    labels: dict | None
    annotations: dict | None


class ProjectSpecLike(typing.Protocol):
    """Structural shape of a project's ``spec`` :func:`resolve_project_body` needs."""

    owner: str | None
    description: str | None


class ProjectStatusLike(typing.Protocol):
    """Structural shape of a project's ``status`` that
    :meth:`~mlrun.utils.orca_client.ProjectsOrchestrator.resolve_prev_op_id` needs.
    """

    op_id: uuid.UUID | str | None


class ProjectLike(typing.Protocol):
    """Structural shape this package's Orca project-sync code needs. Satisfied by
    :class:`~mlrun.projects.MlrunProject`, :class:`mlrun.common.schemas.Project`, and the
    ``types.SimpleNamespace`` :func:`mlrun.db.orca._as_project_like` builds from a dict - the
    three shapes MLRun's own project CUD API already accepts interchangeably.

    Deliberately duck-typed rather than always coercing to a real
    :class:`mlrun.common.schemas.Project` (e.g. via pydantic's validation-skipping
    ``Model.construct()``): this package binds the pydantic-v1 face of that schema in the SDK
    but the API server binds native pydantic-v2 (see ``mlrun.common.schemas._dispatch``) - CI
    guarantees the two faces have the same fields, but not that every pydantic-version-specific
    API behaves identically across them, so this avoids leaning on one at all.

    ``status`` is optional - :func:`resolve_project_body` never reads it (a create has no CAS
    witness yet); only :meth:`~mlrun.utils.orca_client.ProjectsOrchestrator.resolve_prev_op_id`
    does, for a caller that already observed one.
    """

    metadata: ProjectMetadataLike
    spec: ProjectSpecLike
    status: ProjectStatusLike | None


# Sentinel distinguishing "no prev_op_id argument given" (the create shape) from an explicit
# prev_op_id=None (the update/patch shape's valid CAS witness for a project that doesn't exist
# yet) - resolve_project_body() can't use None for both without losing that distinction.
_NO_PREV_OP_ID = object()


def resolve_project_body(
    project: ProjectLike, prev_op_id: uuid.UUID | str | None = _NO_PREV_OP_ID
) -> dict:
    """Build the flat request body Orca's create/update/patch endpoints expect.

    Create (``POST``) has no CAS concept: only ``name`` is required, everything else - owner
    included, Orca derives it from the authenticated caller when omitted - is optional. Update/
    patch (``PUT``/``PATCH``) drop ``name`` (it's in the URL) but require ``prevOpId``/``owner``,
    so a missing value is sent through as-is and surfaces as a real validation error from Orca -
    ``None`` is still a valid witness there, for a PUT that upserts a project that doesn't exist
    yet.

    Pass ``prev_op_id`` (even ``None``) to get the update/patch shape; omit it entirely to get the
    create shape. For example, ``resolve_project_body(project)`` returns
    ``{"name": "p1", "owner": "u1"}``, while ``resolve_project_body(project, op_id)`` returns
    ``{"prevOpId": "...", "owner": "u1"}``.

    :param project: The project's desired state.
    :param prev_op_id: The CAS witness for an update/patch - the ``op_id`` last observed by the
        caller. Omit for a create.
    :return: The JSON-serializable request body.
    """
    if prev_op_id is _NO_PREV_OP_ID:
        body = {"name": project.metadata.name}
        if project.spec.owner:
            body["owner"] = project.spec.owner
    else:
        body = {
            "prevOpId": str(prev_op_id) if prev_op_id else None,
            "owner": project.spec.owner,
        }
    if project.spec.description:
        body["description"] = project.spec.description
    if project.metadata.labels:
        body["labels"] = project.metadata.labels
    if project.metadata.annotations:
        body["annotations"] = project.metadata.annotations
    return body


def extract_op_id(body: dict) -> uuid.UUID | str:
    """Pull the ``op_id`` a create/update/patch/delete response minted, from its parsed body.

    :param body: The parsed JSON body of an Orca create/update/patch/delete response.
    :return: The op_id.
    """
    return body["status"]["opId"]


def extract_error_details(body: dict) -> tuple[str | None, str | None]:
    """Pull ``errorMessage``/``ctx`` out of an Orca error response body's ``status`` envelope -
    shared by every caller that talks to Orca (not just the projects ones), since it's Orca's
    own generic error envelope (``BaseStatus`` in ``v1/common/message.proto``), not something
    specific to the projects wire protocol.

    :param body: The parsed JSON body of an error response.
    :return: ``(error_message, ctx)``, either ``None`` if not present.
    """
    status = body.get("status", {})
    return status.get("errorMessage"), status.get("ctx")


def to_mlproject(body: dict) -> mlrun.common.schemas.Project:
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
    if state == ActionExecutionState.failed:
        raise OrcaActionFailedError(
            f"Orca sync-project action for project {name} (op_id={op_id}) failed"
        )
    if state != ActionExecutionState.succeeded:
        raise mlrun.errors.MLRunRuntimeError(
            f"Orca sync-project action for project {name} (op_id={op_id}) is still in "
            f"progress (state={state})"
        )
