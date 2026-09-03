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
"""Shared orchestration for Orca's project-sync operations.

The sequence of requests create/update/patch/delete/get make, and how each polls to a terminal
state - shared between MLRun's server-side Orca leader-proxy
(``server/py/framework/utils/clients/iguazio/v4.py``) and MLRun's SDK-direct-to-Orca client
(``mlrun/db/orca.py``), so neither maintains its own copy of "what request to make when". The
two callers only differ in how they actually send an authenticated request - the SDK uses its
own held credentials, the leader-proxy relays the acting user's identity - so that's the one
thing each supplies via :class:`RequestSender`. See :mod:`mlrun.utils.orca_projects` for the
wire protocol these requests carry.
"""

import types
import typing
import uuid

import mergedeep
import requests

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.orca_projects as orca_projects


class RequestSender(typing.Protocol):
    """Sends one authenticated request to an Orca endpoint and returns the response, having
    already applied the caller's own status-code-to-exception mapping. Each caller of
    :class:`OrcaProjectsOrchestrator` supplies its own - see
    ``mlrun.db.orca.OrcaProjectsClient._send_request`` (SDK, own credentials) and
    ``server/py/framework/utils/clients/iguazio/v4.py``'s ``Client._send_project_request``
    (server, relays the acting user's identity).
    """

    def __call__(
        self, method: str, path: str, error_message: str, **kwargs
    ) -> requests.Response: ...


class OrcaProjectsOrchestrator:
    """The sequence of Orca requests behind create/update/patch/delete/get, and how each polls
    to completion. Returns raw pieces (responses, op_ids, schema objects) rather than any one
    caller's own public return contract - callers adapt those into whatever shape their own
    interface promises (e.g. the SDK returns ``mlrun.projects.MlrunProject``/``op_id``; the
    server-side leader-proxy returns ``bool``/``None`` per ``project_leader.Member``).
    """

    def __init__(
        self,
        send_request: RequestSender,
        logger,
        poll_interval_seconds: float,
        poll_timeout_seconds: float,
    ):
        self._send_request = send_request
        self._logger = logger
        self._poll_interval_seconds = poll_interval_seconds
        self._poll_timeout_seconds = poll_timeout_seconds

    def create(
        self, project: orca_projects.ProjectLike
    ) -> tuple[requests.Response, uuid.UUID | str]:
        """``POST`` a new project - always async per the HLD (no synchronous-create case).

        :return: The raw response, and the minted ``op_id``.
        """
        name = project.metadata.name
        self._logger.debug("Creating project in Orca", project=name)
        response = self._send_request(
            "POST",
            orca_projects.PROJECTS_ENDPOINT,
            f"Failed creating project {name} in Orca",
            json=orca_projects.create_project_wire(project),
        )
        return response, response.json()["status"]["opId"]

    def update(
        self, name: str, project: orca_projects.ProjectLike
    ) -> tuple[requests.Response, uuid.UUID | str]:
        """``PUT`` a project's desired state. Resolves the CAS witness (``prev_op_id``) first
        if the caller didn't supply one. May settle synchronously (200) or asynchronously (202)
        per the HLD.

        :return: The raw response, and the ``op_id`` this update minted.
        """
        prev_op_id = self.resolve_prev_op_id(name, project)
        self._logger.debug("Updating project in Orca", name=name, prev_op_id=prev_op_id)
        response = self._send_request(
            "PUT",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed updating project {name} in Orca",
            json=orca_projects.update_project_wire(project, prev_op_id),
        )
        return response, response.json()["status"]["opId"]

    def patch(
        self,
        name: str,
        project: orca_projects.ProjectLike,
        patch_mode: mlrun.common.schemas.PatchMode,
    ) -> tuple[requests.Response, uuid.UUID | str]:
        """``PATCH`` a project. Orca's ``PATCH`` is full-replace, not merge (see
        :mod:`mlrun.utils.orca_projects`), so this reads the project's current state first and
        merges ``project``'s changes into it (:mod:`mergedeep`, keyed by ``patch_mode``) before
        sending the merged result as a full object.

        :return: The raw response, and the ``op_id`` this patch minted.
        """
        current = self.get(name)
        merged = _merge_for_patch(current, project, patch_mode)
        response = self._send_request(
            "PATCH",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed patching project {name} in Orca",
            json=orca_projects.update_project_wire(merged, current.status.op_id),
        )
        return response, response.json()["status"]["opId"]

    def delete(self, name: str) -> requests.Response:
        """``DELETE`` a project. Callers handle any deletion-strategy short-circuit themselves
        before calling this - Orca's ``DeleteProjectOptions`` has no strategy concept yet.
        """
        self._logger.debug("Deleting project in Orca", name=name)
        return self._send_request(
            "DELETE",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed deleting project {name} in Orca",
        )

    def get(self, name: str) -> mlrun.common.schemas.Project:
        """``GET`` a project."""
        response = self._send_request(
            "GET",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed getting project {name} from Orca",
        )
        return orca_projects.project_from_wire(response.json())

    def settle(
        self,
        name: str,
        response: requests.Response,
        op_id: uuid.UUID | str,
        wait_for_completion: bool,
    ) -> mlrun.common.schemas.Project | None:
        """Decide whether a create/update/patch response needs polling, and return the final
        project once it's known - or ``None`` if the caller didn't ask to wait.

        ``wait_for_completion=False`` always returns ``None`` immediately, regardless of how the
        response actually settled - callers that asked not to wait get nothing further to look
        at here; ``op_id`` (already returned by :meth:`create`/:meth:`update`/:meth:`patch`) is
        their handle to check later. A synchronous (non-202) response is already terminal, so
        this fetches nothing further and parses the project out of it directly; a 202 is polled
        to a terminal state via :meth:`wait_for_op`, then the final project is fetched fresh.
        """
        if not wait_for_completion:
            return None
        if response.status_code != requests.codes.accepted:
            return orca_projects.project_from_wire(response.json())
        self.wait_for_op(name, op_id)
        return self.get(name)

    def resolve_prev_op_id(
        self, name: str, project: orca_projects.ProjectLike
    ) -> uuid.UUID | str | None:
        # The CAS witness Orca requires for an update is the last op_id the caller observed; if
        # the caller didn't supply one, read the current state from Orca first (matches the
        # HLD's "client reads the project, then PUT/PATCH with prev_op_id" contract). A missing
        # project (an upsert-create case: PUT on a project that doesn't exist yet) has no prior
        # op_id to CAS against - fall through with None.
        prev_op_id = getattr(getattr(project, "status", None), "op_id", None)
        if prev_op_id:
            return prev_op_id
        try:
            return self.get(name).status.op_id
        except mlrun.errors.MLRunNotFoundError:
            return None

    def wait_for_op(self, name: str, op_id: uuid.UUID | str) -> None:
        """Poll the sync-project trackable action for ``op_id`` to a terminal state.

        :raises mlrun.errors.MLRunRuntimeError: if the action fails or the poll times out.
        """
        self._logger.debug(
            "Waiting for Orca sync-project action to reach a terminal state",
            name=name,
            op_id=op_id,
        )
        try:
            mlrun.utils.helpers.retry_until_successful(
                self._poll_interval_seconds,
                self._poll_timeout_seconds,
                self._logger,
                False,
                self._verify_op_terminal,
                name,
                op_id,
                fatal_exceptions=(orca_projects.OrcaActionFailedError,),
            )
        except orca_projects.OrcaActionFailedError as exc:
            raise mlrun.errors.MLRunRuntimeError(str(exc)) from exc

    def _verify_op_terminal(self, name: str, op_id: uuid.UUID | str) -> None:
        response = self._send_request(
            "GET",
            orca_projects.ACTION_EXECUTIONS_ENDPOINT,
            "Failed getting Orca sync-project action execution",
            params=orca_projects.action_execution_query_params(op_id),
        )
        orca_projects.verify_action_execution_terminal(response.json(), name, op_id)


def _merge_for_patch(
    current: mlrun.common.schemas.Project,
    project: orca_projects.ProjectLike,
    patch_mode: mlrun.common.schemas.PatchMode,
) -> orca_projects.ProjectLike:
    merged_common = {
        "labels": dict(current.metadata.labels or {}),
        "annotations": dict(current.metadata.annotations or {}),
        "owner": current.spec.owner,
        "description": current.spec.description,
    }
    patch_common = {
        "labels": project.metadata.labels,
        "annotations": project.metadata.annotations,
        "owner": project.spec.owner,
        "description": project.spec.description,
    }
    patch_common = {k: v for k, v in patch_common.items() if v is not None}
    mergedeep.merge(
        merged_common, patch_common, strategy=patch_mode.to_mergedeep_strategy()
    )
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            name=current.metadata.name,
            labels=merged_common["labels"],
            annotations=merged_common["annotations"],
        ),
        spec=types.SimpleNamespace(
            owner=merged_common["owner"],
            description=merged_common["description"],
        ),
    )
