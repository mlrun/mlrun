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
"""SDK-direct client for Orca's user-facing project endpoints.

In enterprise (IG4/Orca-led) deployments, this bypasses the MLRun API entirely for project
create/update/patch/delete: it calls Orca with the SDK's own user credentials (Orca authorizes
the user itself, via OPA), then polls to a terminal state the same way MLRun's own backward-
compatibility proxy does server-side (``server/py/framework/utils/clients/iguazio/v4.py``,
mlrun#10043) - see :mod:`mlrun.utils.orca_projects` for the wire protocol shared between the two.
"""

import types
import typing
import uuid

import humanfriendly
import mergedeep
import requests

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.orca_projects as orca_projects
from mlrun.utils import logger

if typing.TYPE_CHECKING:
    import mlrun.db.httpdb


def _as_project_like(project):
    """Normalize ``project`` into something with ``.metadata``/``.spec``/``.status`` attribute
    access - the shape :mod:`mlrun.utils.orca_projects`'s wire functions expect.

    Accepts a dict, :class:`~mlrun.projects.MlrunProject`, or
    :class:`mlrun.common.schemas.Project` - all three are already used interchangeably by
    MLRun's own project CUD API (see ``HTTPRunDB.create_project``/``store_project``/
    ``patch_project``). A dict may be partial (``patch_project``'s body only carries the changed
    fields, and never ``metadata.name`` - that comes from the separate ``name`` argument), so
    missing fields become ``None`` rather than raising - matching the wire functions' own
    ``if project.spec.owner:``-style optional-field handling.
    """
    if not isinstance(project, dict):
        return project
    metadata = project.get("metadata") or {}
    spec = project.get("spec") or {}
    status = project.get("status") or {}
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            name=metadata.get("name"),
            labels=metadata.get("labels"),
            annotations=metadata.get("annotations"),
        ),
        spec=types.SimpleNamespace(
            owner=spec.get("owner"),
            description=spec.get("description"),
        ),
        status=types.SimpleNamespace(op_id=status.get("op_id") or status.get("opId")),
    )


class OrcaProjectsClient:
    """Talks to Orca's user-facing project endpoints directly, using the credentials of the
    :class:`~mlrun.db.httpdb.HTTPRunDB` instance it is attached to - the same credentials that
    instance uses to talk to the MLRun API, since Orca and the MLRun API trust the same IG4
    session/token.
    """

    def __init__(self, run_db: "mlrun.db.httpdb.HTTPRunDB"):
        self._run_db = run_db
        self._session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_exception=(
                mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
                == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value
            ),
            verbose=True,
        )
        self._poll_interval_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.iguazio_project_states_poll_interval
        )
        self._poll_timeout_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.iguazio_project_states_poll_timeout
        )

    def create_project(
        self, project, wait_for_completion: bool = True
    ) -> typing.Union["mlrun.projects.MlrunProject", uuid.UUID]:
        """Create a project directly in Orca.

        :param project: The project to create - a :class:`~mlrun.projects.MlrunProject`,
            :class:`mlrun.common.schemas.Project`, or an equivalent dict.
        :param wait_for_completion: Block until Orca's create operation reaches a terminal state
            and return the resulting project. If ``False``, return the operation's ``op_id``
            immediately instead.
        :return: The created project, or the operation's ``op_id`` if ``wait_for_completion`` is
            ``False``.
        """
        project = _as_project_like(project)
        name = project.metadata.name
        response = self._send_request(
            "POST",
            orca_projects.PROJECTS_ENDPOINT,
            f"Failed creating project {name} in Orca",
            json=orca_projects.create_project_wire(project),
        )
        return self._settle(name, response, wait_for_completion)

    def update_project(
        self, name: str, project, wait_for_completion: bool = True
    ) -> typing.Union["mlrun.projects.MlrunProject", uuid.UUID]:
        """Update (``PUT``) a project directly in Orca. See :meth:`create_project`."""
        project = _as_project_like(project)
        prev_op_id = self._resolve_prev_op_id(name, project)
        response = self._send_request(
            "PUT",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed updating project {name} in Orca",
            json=orca_projects.update_project_wire(project, prev_op_id),
        )
        return self._settle(name, response, wait_for_completion)

    def patch_project(
        self,
        name: str,
        project,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        wait_for_completion: bool = True,
    ) -> typing.Union["mlrun.projects.MlrunProject", uuid.UUID]:
        """Patch (partial update) a project directly in Orca.

        Orca's ``PATCH`` is full-replace, not merge (same required fields, same semantics as its
        ``PUT`` - see orca SDK PR #1059), unlike MLRun's own ``patch_project``, which merges only
        the given fields into the existing stored project. To preserve that partial-patch
        contract, this reads the project's current common-set fields from Orca first, merges
        ``project``'s changes into them the same way MLRun's own server does
        (:mod:`mergedeep`, keyed by ``patch_mode``), and sends the merged result to Orca's
        ``PATCH`` as a full object.
        """
        project = _as_project_like(project)
        current = self.get_project(name)
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
        merged_project = types.SimpleNamespace(
            metadata=types.SimpleNamespace(
                name=name,
                labels=merged_common["labels"],
                annotations=merged_common["annotations"],
            ),
            spec=types.SimpleNamespace(
                owner=merged_common["owner"],
                description=merged_common["description"],
            ),
        )
        response = self._send_request(
            "PATCH",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed patching project {name} in Orca",
            json=orca_projects.update_project_wire(
                merged_project, current.status.op_id
            ),
        )
        return self._settle(name, response, wait_for_completion)

    def delete_project(
        self, name: str, wait_for_completion: bool = True
    ) -> uuid.UUID | None:
        """Delete a project directly in Orca.

        :param name: Name of the project to delete.
        :param wait_for_completion: Block until Orca's delete operation reaches a terminal state.
        :return: The operation's ``op_id`` if the delete is still converging and
            ``wait_for_completion`` is ``False``, otherwise ``None``.
        """
        response = self._send_request(
            "DELETE",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed deleting project {name} in Orca",
        )
        if response.status_code != requests.codes.accepted:
            return None
        op_id = response.json()["status"]["opId"]
        if not wait_for_completion:
            return op_id
        self._wait_for_op(name, op_id)
        return None

    def get_project(self, name: str) -> "mlrun.common.schemas.Project":
        """Get a project directly from Orca."""
        response = self._send_request(
            "GET",
            orca_projects.PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            f"Failed getting project {name} from Orca",
        )
        return orca_projects.project_from_wire(response.json())

    def _settle(
        self, name: str, response: requests.Response, wait_for_completion: bool
    ) -> typing.Union["mlrun.common.schemas.Project", uuid.UUID]:
        # update/patch may settle synchronously (200 - "all ack -> clear phase -> 200" per the
        # HLD) rather than go through the async 202 + poll path create always takes. A 200 has
        # already reached its terminal state, and - unlike 202 - has no trackable-action record
        # to poll for, so it must be handled before any polling is attempted.
        if response.status_code != requests.codes.accepted:
            return orca_projects.project_from_wire(response.json())
        op_id = response.json()["status"]["opId"]
        if not wait_for_completion:
            return op_id
        self._wait_for_op(name, op_id)
        return self.get_project(name)

    def _resolve_prev_op_id(self, name: str, project) -> uuid.UUID | None:
        # The CAS witness Orca requires for an update is the last op_id the caller observed; if
        # the caller didn't supply one, read the current state from Orca first (matches the
        # HLD's "client reads the project, then PUT/PATCH with prev_op_id" contract). A missing
        # project (store_project's upsert-create case: PUT on a project that doesn't exist yet)
        # has no prior op_id to CAS against - fall through with None.
        prev_op_id = getattr(getattr(project, "status", None), "op_id", None)
        if prev_op_id:
            return prev_op_id
        try:
            return self.get_project(name).status.op_id
        except mlrun.errors.MLRunNotFoundError:
            return None

    def _wait_for_op(self, name: str, op_id: uuid.UUID | str) -> None:
        logger.debug(
            "Waiting for Orca sync-project action to reach a terminal state",
            name=name,
            op_id=op_id,
        )
        try:
            mlrun.utils.helpers.retry_until_successful(
                self._poll_interval_seconds,
                self._poll_timeout_seconds,
                logger,
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

    def _send_request(
        self, method: str, path: str, error_message: str, **kwargs
    ) -> requests.Response:
        if not mlrun.mlconf.iguazio_api_url:
            raise mlrun.errors.MLRunRuntimeError(
                "Cannot talk to Orca directly: iguazio_api_url is not configured"
            )
        url = f"{mlrun.mlconf.iguazio_api_url}/api/{path}"
        kwargs.update(self._run_db._auth_request_kwargs(kwargs.pop("headers", None)))
        try:
            response = self._session.request(
                method,
                url,
                timeout=20,
                verify=mlrun.mlconf.iguazio_api_ssl_verify,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"{mlrun.errors.err_to_str(exc)}: {error_message}"
            ) from exc
        # Orca's error response body shape is unverified (same caveat mlrun#10043 flagged for the
        # server-side proxy), so this doesn't try to extract error details from it - just the
        # status-code-to-exception mapping raise_for_status already gives for free.
        mlrun.errors.raise_for_status(response, error_message)
        return response
