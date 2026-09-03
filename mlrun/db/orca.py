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
mlrun#10043). The actual request sequencing (what to call, in what order, how to poll) lives in
:mod:`mlrun.utils.orca_client`, shared with that server-side proxy - this module only supplies
the SDK-specific pieces: how to send an authenticated request, and how to translate results into
the SDK's own public return types.
"""

import types
import typing
import uuid

import humanfriendly
import requests

import mlrun.common.schemas
import mlrun.errors
import mlrun.projects
import mlrun.utils
import mlrun.utils.orca_client as orca_client
import mlrun.utils.orca_projects as orca_projects

if typing.TYPE_CHECKING:
    import mlrun.db.httpdb

# The three shapes MLRun's own project CUD API already accepts interchangeably (see
# HTTPRunDB.create_project/store_project/patch_project) - before normalization to
# orca_projects.ProjectLike via _as_project_like().
ProjectInput = typing.Union[
    dict, "mlrun.projects.MlrunProject", "mlrun.common.schemas.Project"
]


def _as_project_like(project: ProjectInput) -> orca_projects.ProjectLike:
    """Normalize ``project`` into something with ``.metadata``/``.spec`` attribute access - the
    shape :mod:`mlrun.utils.orca_projects`'s wire functions expect.

    A dict input may be partial (``patch_project``'s body only carries the changed fields, and
    never ``metadata.name`` - that comes from the separate ``name`` argument), so missing fields
    become ``None`` rather than raising - matching the wire functions' own
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
        self._orchestrator = orca_client.OrcaProjectsOrchestrator(
            self._send_request,
            mlrun.utils.logger,
            poll_interval_seconds=humanfriendly.parse_timespan(
                mlrun.mlconf.httpdb.projects.iguazio_project_states_poll_interval
            ),
            poll_timeout_seconds=humanfriendly.parse_timespan(
                mlrun.mlconf.httpdb.projects.iguazio_project_states_poll_timeout
            ),
        )

    @typing.overload
    def create_project(
        self, project: ProjectInput, wait_for_completion: typing.Literal[True] = True
    ) -> "mlrun.projects.MlrunProject": ...
    @typing.overload
    def create_project(
        self, project: ProjectInput, wait_for_completion: typing.Literal[False]
    ) -> uuid.UUID: ...
    def create_project(
        self, project: ProjectInput, wait_for_completion: bool = True
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
        project_like = _as_project_like(project)
        response, op_id = self._orchestrator.create(project_like)
        if not wait_for_completion:
            return op_id
        return self._to_mlrun_project(
            self._orchestrator.settle(
                project_like.metadata.name, response, op_id, wait_for_completion=True
            )
        )

    @typing.overload
    def update_project(
        self,
        name: str,
        project: ProjectInput,
        wait_for_completion: typing.Literal[True] = True,
    ) -> "mlrun.projects.MlrunProject": ...
    @typing.overload
    def update_project(
        self,
        name: str,
        project: ProjectInput,
        wait_for_completion: typing.Literal[False],
    ) -> uuid.UUID: ...
    def update_project(
        self, name: str, project: ProjectInput, wait_for_completion: bool = True
    ) -> typing.Union["mlrun.projects.MlrunProject", uuid.UUID]:
        """Update (``PUT``) a project directly in Orca.

        :param name: Name of the project to update.
        :param project: The project's desired state - see :meth:`create_project`.
        :param wait_for_completion: See :meth:`create_project`.
        :return: The updated project, or the operation's ``op_id`` if ``wait_for_completion`` is
            ``False``.
        """
        project_like = _as_project_like(project)
        response, op_id = self._orchestrator.update(name, project_like)
        if not wait_for_completion:
            return op_id
        return self._to_mlrun_project(
            self._orchestrator.settle(name, response, op_id, wait_for_completion=True)
        )

    @typing.overload
    def patch_project(
        self,
        name: str,
        project: ProjectInput,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        wait_for_completion: typing.Literal[True] = True,
    ) -> "mlrun.projects.MlrunProject": ...
    @typing.overload
    def patch_project(
        self,
        name: str,
        project: ProjectInput,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        *,
        wait_for_completion: typing.Literal[False],
    ) -> uuid.UUID: ...
    def patch_project(
        self,
        name: str,
        project: ProjectInput,
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

        :param name: Name of the project to patch.
        :param project: The changes to apply - only the fields present are merged in.
        :param patch_mode: The strategy for merging the changes with the existing object. Can be
            either ``replace`` or ``additive``.
        :param wait_for_completion: See :meth:`create_project`.
        :return: The patched project, or the operation's ``op_id`` if ``wait_for_completion`` is
            ``False``.
        """
        project_like = _as_project_like(project)
        response, op_id = self._orchestrator.patch(name, project_like, patch_mode)
        if not wait_for_completion:
            return op_id
        return self._to_mlrun_project(
            self._orchestrator.settle(name, response, op_id, wait_for_completion=True)
        )

    @typing.overload
    def delete_project(
        self, name: str, wait_for_completion: typing.Literal[True] = True
    ) -> None: ...
    @typing.overload
    def delete_project(
        self, name: str, wait_for_completion: typing.Literal[False]
    ) -> uuid.UUID | None: ...
    def delete_project(
        self, name: str, wait_for_completion: bool = True
    ) -> uuid.UUID | None:
        """Delete a project directly in Orca.

        :param name: Name of the project to delete.
        :param wait_for_completion: Block until Orca's delete operation reaches a terminal state.
        :return: The operation's ``op_id`` if the delete is still converging and
            ``wait_for_completion`` is ``False``, otherwise ``None``.
        """
        response = self._orchestrator.delete(name)
        if response.status_code != requests.codes.accepted:
            return None
        op_id = response.json()["status"]["opId"]
        if not wait_for_completion:
            return op_id
        self._orchestrator.wait_for_op(name, op_id)
        return None

    def get_project(self, name: str) -> "mlrun.projects.MlrunProject":
        """Get a project directly from Orca.

        :param name: Name of the project to get.
        :return: The project.
        """
        return self._to_mlrun_project(self._orchestrator.get(name))

    @staticmethod
    def _to_mlrun_project(
        project: "mlrun.common.schemas.Project",
    ) -> "mlrun.projects.MlrunProject":
        return mlrun.projects.MlrunProject.from_dict(project.dict())

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
