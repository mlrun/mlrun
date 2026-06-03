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
import typing

import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.errors
from mlrun.utils import logger

import services.api.crud.functions
import services.api.crud.projects


def json_loads_if_not_none(field: typing.Any) -> typing.Any:
    return (
        json.loads(field) if field and field != "null" and field is not None else None
    )


def get_access_key(auth_info: mlrun.common.schemas.AuthInfo):
    """
    Get access key from the current data session. This method is usually used to verify that the session
    is valid and contains an access key.

    param auth_info: The auth info of the request.

    :return: Access key as a string.
    """
    access_key = auth_info.data_session
    if not access_key:
        raise mlrun.errors.MLRunBadRequestError("Data session is missing")
    return access_key


async def get_stream_url(
    db_session: sqlalchemy.orm.Session,
    project: str,
) -> str | None:
    """
    Return the internal cluster HTTP URL of the model monitoring stream pod for *project*.

    The URL is resolved from the nuclio function's internal_invocation_urls, which are
    only reachable from within the Kubernetes cluster (e.g. from other nuclio functions
    or pods running in the same cluster). It is NOT an externally accessible URL.

    :param db_session: A session that manages the current dialog with the database.
    :param project:    Project name.
    :return: Internal cluster HTTP URL of the stream pod, or None when no HTTP trigger is configured.
        A non-ready stream pod (e.g. mid-deploy, scaling) still returns its URL with a warning —
        the URL may not be reachable until the pod becomes ready. A stream pod in terminal
        error state raises (the deploy cannot rely on a broken stream).
    :raises MLRunNotFoundError: if the stream function is not deployed.
    :raises MLRunPreconditionFailedError: if the stream function is in terminal error state.
    """
    try:
        func = await run_in_threadpool(
            services.api.crud.functions.Functions().get_function,
            db_session,
            mm_constants.MonitoringFunctionNames.STREAM,
            project,
        )
    except mlrun.errors.MLRunNotFoundError as exc:
        raise mlrun.errors.MLRunNotFoundError(
            f"Model monitoring stream function not found for project {project!r}. "
            f"Run `project.enable_model_monitoring()` first."
        ) from exc

    status = func.get("status", {})
    state = status.get("state", "")
    if state in mlrun.common.schemas.FunctionState.failed_states():
        raise mlrun.errors.MLRunPreconditionFailedError(
            f"Model monitoring stream function is in terminal failure state {state!r} "
            f"for project {project!r}. Re-run `project.enable_model_monitoring()` to recover."
        )
    if state != mlrun.common.schemas.FunctionState.ready:
        logger.warning(
            "Model monitoring stream function is not in ready state — "
            "MODEL_MONITORING_URL may not be reachable until the stream pod becomes ready.",
            project=project,
            state=state,
        )

    internal_urls = status.get("internal_invocation_urls") or []
    if internal_urls and internal_urls[0]:
        url = internal_urls[0]
        return url if url.startswith("http") else f"http://{url}"

    return None


def get_monitoring_parquet_path(
    db_session: sqlalchemy.orm.Session,
    project: str,
    kind: str = mlrun.common.schemas.model_monitoring.FileTargetKind.PARQUET,
) -> str:
    """Get model monitoring parquet target for the current project. The parquet target path is based on the
    project artifact path. If project artifact path is not defined, the parquet target path will be based on MLRun
    artifact path.

    :param db_session: A session that manages the current dialog with the database. Will be used in this function
                       to get the project record from DB.
    :param project:    Project name.
    :param kind:       indicate the kind of the parquet path, can be either stream_parquet or stream_controller_parquet

    :return:           Monitoring parquet target path.
    """

    # Get the artifact path from the project record that was stored in the DB
    project_obj = services.api.crud.projects.Projects().get_project(
        session=db_session, name=project
    )
    artifact_path = project_obj.spec.artifact_path
    # Generate monitoring parquet path value
    parquet_path = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=kind,
        target="offline",
        artifact_path=artifact_path,
    )
    return parquet_path
