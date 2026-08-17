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

import datetime

import humanfriendly
import requests

import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
from mlrun.utils import logger

import framework.utils.clients.helpers
import framework.utils.projects.remotes.leader as project_leader

PROJECTS_ENDPOINT = "projects"
PROJECT_ENDPOINT_TEMPLATE = "projects/{name}"
PROJECT_STATE_ENDPOINT_TEMPLATE = "project-states/{name}"


class Client(project_leader.Member):
    """
    Leader-client used when ``httpdb.projects.leader == "orca"``.

    This is the "role 2" backward-compatibility proxy from the Project Sync Mechanism HLD: MLRun's user
    project endpoints never apply these operations locally in this mode. Every call is forwarded to Orca,
    relaying the acting user's own auth headers so Orca's OPA authorizes the request (MLRun never runs
    authorization itself here); ``wait_for_completion`` is then honored by polling Orca's single-project
    state endpoint until the operation reaches a terminal state, since Orca's CUD endpoints are always
    async (202 + op_id).
    """

    def __init__(self) -> None:
        self._logger = logger.get_child("orca-client")
        self._session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_exception=(
                mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
                == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value
            ),
            verbose=True,
        )
        self._poll_interval_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.orca_project_states_poll_interval
        )
        self._poll_timeout_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.orca_project_states_poll_timeout
        )

    @property
    def _api_url(self) -> str:
        return mlrun.mlconf.orca_api_url

    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        """
        :return: whether the operation is still running in the background (Orca is always async, so this is
            ``True`` unless the caller asked to wait and the operation already reached a terminal state).
        """
        self._logger.debug("Creating project in Orca", project=project.metadata.name)
        response = self._send_request_to_orca(
            mlrun.common.types.HTTPMethod.POST,
            PROJECTS_ENDPOINT,
            "Failed creating project in Orca",
            auth_info,
            json=self._project_to_wire(project),
        )
        op_id = response.json()["status"]["op_id"]
        return self._wait_for_op_if_requested(
            project.metadata.name, op_id, wait_for_completion, auth_info
        )

    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        # the CAS witness Orca requires for an update is the last op_id the caller observed; if the caller
        # didn't supply one, read the current state from Orca first (matches the HLD's "client reads the
        # project, then PUT/PATCH with prev_op_id" contract)
        current_op_id = (
            project.status.op_id
            or self.get_project(session, name, auth_info).status.op_id
        )
        self._logger.debug(
            "Updating project in Orca", name=name, current_op_id=current_op_id
        )
        body = self._project_to_wire(project)
        body["current_op_id"] = str(current_op_id) if current_op_id else None
        response = self._send_request_to_orca(
            mlrun.common.types.HTTPMethod.PUT,
            PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            "Failed updating project in Orca",
            auth_info,
            json=body,
        )
        op_id = response.json()["status"]["op_id"]
        # the shared leader-client interface has no wait_for_completion for update, so always settle here to
        # match the legacy synchronous contract the caller expects
        self._wait_for_op_if_requested(name, op_id, True, auth_info)

    def delete_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        self._logger.debug(
            "Deleting project in Orca", name=name, deletion_strategy=deletion_strategy
        )
        # the HLD documents the deletion-strategy gate as leader-side logic but doesn't pin how a caller
        # communicates the strategy to Orca; sending it in the body (rather than reusing an Iguazio-specific
        # header) keeps this consistent with every other write on this client until Orca's real contract lands
        response = self._send_request_to_orca(
            mlrun.common.types.HTTPMethod.DELETE,
            PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            "Failed deleting project in Orca",
            auth_info,
            json={"deletion_strategy": deletion_strategy.value},
        )
        if response.status_code in (
            requests.codes.ok,
            requests.codes.no_content,
        ):
            # deleted (or was already absent) synchronously - nothing to poll
            return False
        op_id = response.json()["status"]["op_id"]
        # a delete's terminal signal is the project disappearing from project-states, not a status value
        return self._wait_for_op_if_requested(
            name, op_id, wait_for_completion, auth_info, absence_is_terminal=True
        )

    def get_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.Project:
        response = self._send_request_to_orca(
            mlrun.common.types.HTTPMethod.GET,
            PROJECT_STATE_ENDPOINT_TEMPLATE.format(name=name),
            "Failed getting project state from Orca",
            auth_info,
        )
        return self._wire_to_project(response.json())

    def list_projects(
        self,
        session: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        updated_after: datetime.datetime | None = None,
    ) -> tuple[list[mlrun.common.schemas.Project], datetime.datetime | None]:
        # Orca-mode reconciliation is leader-driven push plus the MLRun follower's own one-time startup sync
        # (the dedicated /follower/projects/* surface); this role-2 proxy never runs mlrun's legacy
        # periodic/full project sync against Orca, so this must stay unreachable (project_sync feature gate
        # must stay disabled when leader=orca).
        raise NotImplementedError(
            "Periodic project sync is not supported when the leader is Orca"
        )

    def get_project_owner(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.ProjectOwner:
        raise NotImplementedError(
            "get_project_owner is not supported when the leader is Orca"
        )

    def format_as_leader_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.IguazioProject:
        raise NotImplementedError(
            "format_as_leader_project is not supported when the leader is Orca"
        )

    def _wait_for_op_if_requested(
        self,
        name: str,
        op_id: str,
        wait_for_completion: bool,
        auth_info: mlrun.common.schemas.AuthInfo,
        absence_is_terminal: bool = False,
    ) -> bool:
        if not wait_for_completion:
            return True
        self._logger.debug(
            "Waiting for Orca operation to reach a terminal state",
            name=name,
            op_id=op_id,
        )
        mlrun.utils.helpers.retry_until_successful(
            self._poll_interval_seconds,
            self._poll_timeout_seconds,
            self._logger,
            False,
            self._verify_op_terminal,
            name,
            op_id,
            auth_info,
            absence_is_terminal,
        )
        return False

    def _verify_op_terminal(
        self,
        name: str,
        op_id: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        absence_is_terminal: bool,
    ):
        try:
            project = self.get_project("", name, auth_info)
        except mlrun.errors.MLRunNotFoundError:
            if absence_is_terminal:
                return
            raise
        if str(project.status.op_id) != str(op_id):
            # a newer operation superseded ours (e.g. a concurrent update) - nothing more to wait for
            return
        if (
            project.status.state
            not in mlrun.common.schemas.ProjectState.terminal_states()
        ):
            raise mlrun.errors.MLRunRuntimeError(
                f"Orca operation {op_id} for project {name} is still in progress "
                f"(state={project.status.state})"
            )

    @staticmethod
    def _project_to_wire(project: mlrun.common.schemas.Project) -> dict:
        return {
            "metadata": {
                "name": project.metadata.name,
                "labels": project.metadata.labels or {},
                "annotations": project.metadata.annotations or {},
            },
            "spec": {
                "owner": project.spec.owner,
                "description": project.spec.description,
            },
        }

    @staticmethod
    def _wire_to_project(body: dict) -> mlrun.common.schemas.Project:
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
                op_id=status.get("op_id"),
                updated_at=status.get("updated_at"),
            ),
        )

    def _send_request_to_orca(
        self,
        method: str,
        path: str,
        error_message: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        **kwargs,
    ) -> requests.Response:
        # Orca's Enterprise auth is JWT-bearer, not the Iguazio-v3 session cookie: relay the acting user's
        # original request headers so Orca's OPA authorizes on the user's own identity, matching the pattern
        # already used for is_iguazio_v4_mode() calls to Nuclio (framework.utils.clients.async_nuclio.Client).
        # Deliberately not using enrich_headers(): it injects x-projects-role: mlrun for any "projects" path,
        # which would assert mlrun's own leader-role identity - the opposite of a pure user-identity relay.
        headers = dict(auth_info.request_headers or {})
        headers.pop("content-length", None)
        headers.pop(mlrun.common.schemas.HeaderNames.projects_role, None)
        headers.update(kwargs.pop("headers", None) or {})
        framework.utils.clients.helpers.inject_context_id_header(headers)
        kwargs["headers"] = headers
        kwargs.setdefault("timeout", 20)
        url = f"{self._api_url}/api/v1/{path}"
        response = self._session.request(
            method, url, verify=mlrun.mlconf.orca_api_ssl_verify, **kwargs
        )
        if not response.ok:
            if response.status_code == requests.codes.not_found:
                raise mlrun.errors.MLRunNotFoundError(error_message)
            mlrun.errors.raise_for_status(response, error_message)
        return response
