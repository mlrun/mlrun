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

import traceback
from distutils.util import strtobool
from http import HTTPStatus
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    Query,
    Request,
    Response,
)
from fastapi.concurrency import run_in_threadpool
from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.common.model_monitoring
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
import server.api.api.utils
import server.api.crud.model_monitoring.deployment
import server.api.crud.runtimes.nuclio.function
import server.api.db.session
import server.api.launcher
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.chief
import server.api.utils.functions
import server.api.utils.pagination
import server.api.utils.singletons.k8s
import server.api.utils.singletons.project_member
from mlrun.common.helpers import parse_versioned_object_uri
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.run import new_function
from mlrun.runtimes import RuntimeKinds
from mlrun.utils import get_in, logger, update_in
from server.api.api import deps
from server.api.api.endpoints.nuclio import (
    _handle_nuclio_deploy_status,
)
from server.api.utils.singletons.scheduler import get_scheduler

router = APIRouter()


@router.post("/projects/{project}/functions/{name}")
async def store_function(
    request: Request,
    project: str,
    name: str,
    tag: str = "",
    versioned: bool = False,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    data = None
    try:
        data = await request.json()
    except ValueError:
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value, reason="bad JSON body"
        )

    logger.debug("Storing function", project=project, name=name, tag=tag)
    hash_key = await run_in_threadpool(
        server.api.crud.Functions().store_function,
        db_session,
        data,
        name,
        project,
        tag=tag,
        versioned=versioned,
        auth_info=auth_info,
    )
    return {
        "hash_key": hash_key,
    }


@router.get("/projects/{project}/functions/{name}")
async def get_function(
    project: str,
    name: str,
    tag: str = "",
    hash_key="",
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    func = await run_in_threadpool(
        server.api.crud.Functions().get_function,
        db_session,
        name,
        project,
        tag,
        hash_key,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return {
        "func": func,
    }


@router.delete(
    "/projects/{project}/functions/{name}", status_code=HTTPStatus.NO_CONTENT.value
)
async def delete_function(
    request: Request,
    project: str,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    #  If the requested function has a schedule, we must delete it before deleting the function
    try:
        function_schedule = await run_in_threadpool(
            get_scheduler().get_schedule,
            db_session,
            project,
            name,
        )
    except mlrun.errors.MLRunNotFoundError:
        function_schedule = None

    if function_schedule:
        # when deleting a function, we should also delete its schedules if exists
        # schedules are only supposed to be run by the chief, therefore, if the function has a schedule,
        # and we are running in worker, we send the request to the chief client
        if (
            mlrun.mlconf.httpdb.clusterization.role
            != mlrun.common.schemas.ClusterizationRole.chief
        ):
            logger.info(
                "Function has a schedule, deleting",
                function=name,
                project=project,
            )
            chief_client = server.api.utils.clients.chief.Client()
            await chief_client.delete_schedule(
                project=project, name=name, request=request
            )
        else:
            await run_in_threadpool(
                get_scheduler().delete_schedule, db_session, project, name
            )
    await run_in_threadpool(
        server.api.crud.Functions().delete_function, db_session, project, name
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/projects/{project}/functions")
async def list_functions(
    project: str = None,
    name: str = None,
    tag: str = None,
    labels: list[str] = Query([], alias="label"),
    hash_key: str = None,
    page: int = Query(None, gt=0),
    page_size: int = Query(None, alias="page-size", gt=0),
    page_token: str = Query(None, alias="page-token"),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if project is None:
        project = config.default_project
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    paginator = server.api.utils.pagination.Paginator()

    async def _filter_functions_by_permissions(_functions):
        return await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.function,
            _functions,
            lambda function: (
                function.get("metadata", {}).get(
                    "project", mlrun.mlconf.default_project
                ),
                function["metadata"]["name"],
            ),
            auth_info,
        )

    functions, page_info = await paginator.paginate_permission_filtered_request(
        db_session,
        server.api.crud.Functions().list_functions,
        _filter_functions_by_permissions,
        auth_info,
        token=page_token,
        page=page,
        page_size=page_size,
        project=project,
        name=name,
        tag=tag,
        labels=labels,
        hash_key=hash_key,
    )

    return {
        "funcs": functions,
        "pagination": page_info,
    }


@router.post("/build/function")
@router.post("/build/function/")
async def build_function(
    request: Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    client_version: Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
    client_python_version: Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.python_version
    ),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value, reason="bad JSON body"
        )

    logger.info("Building function", data=data)
    function = data.get("function")
    project = function.get("metadata", {}).get("project", mlrun.mlconf.default_project)
    function_name = function.get("metadata", {}).get("name")
    await run_in_threadpool(
        server.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        project,
        function_name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )

    # schedules are meant to be run solely by the chief then if serving function and track_models is enabled,
    # it means that schedules will be created as part of building the function, and if not chief then redirect to chief.
    # to reduce redundant load on the chief, we re-route the request only if the user has permissions
    if function.get("kind", "") == mlrun.runtimes.RuntimeKinds.serving and function.get(
        "spec", {}
    ).get("track_models", False):
        if (
            mlrun.mlconf.httpdb.clusterization.role
            != mlrun.common.schemas.ClusterizationRole.chief
        ):
            logger.info(
                "Requesting to deploy serving function with track models, re-routing to chief",
                function_name=function_name,
                project=project,
                function=function,
            )
            chief_client = server.api.utils.clients.chief.Client()
            return await chief_client.build_function(request=request, json=data)

    if isinstance(data.get("with_mlrun"), bool):
        with_mlrun = data.get("with_mlrun")
    else:
        with_mlrun = strtobool(data.get("with_mlrun", "on"))
    skip_deployed = data.get("skip_deployed", False)
    force_build = data.get("force_build", False)
    mlrun_version_specifier = data.get("mlrun_version_specifier")
    fn, ready = await run_in_threadpool(
        server.api.utils.functions.build_function,
        db_session,
        auth_info,
        function,
        with_mlrun,
        skip_deployed,
        mlrun_version_specifier,
        data.get("builder_env"),
        client_version,
        client_python_version,
        force_build,
    )

    # clone_target_dir is deprecated but needs to remain for backward compatibility
    func_dict = fn.to_dict()
    func_dict["spec"]["clone_target_dir"] = get_in(
        func_dict, "spec.build.source_code_target_dir"
    )

    return {
        "data": func_dict,
        "ready": ready,
    }


@router.post("/start/function", response_model=mlrun.common.schemas.BackgroundTask)
@router.post("/start/function/", response_model=mlrun.common.schemas.BackgroundTask)
async def start_function(
    request: Request,
    background_tasks: BackgroundTasks,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    client_version: Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
    client_python_version: Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.python_version
    ),
):
    # TODO: ensure project here !!! for background task
    data = None
    try:
        data = await request.json()
    except ValueError:
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value, reason="bad JSON body"
        )

    logger.info("Got request to start function", body=data)

    function = await run_in_threadpool(_parse_start_function_body, db_session, data)
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        function.metadata.project,
        function.metadata.name,
        mlrun.common.schemas.AuthorizationAction.update,
        auth_info,
    )
    background_timeout = mlrun.mlconf.background_tasks.default_timeouts.runtimes.dask

    background_task = await run_in_threadpool(
        server.api.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task,
        db_session,
        function.metadata.project,
        background_tasks,
        _start_function_wrapper,
        background_timeout,
        None,
        # args for _start_function
        function,
        auth_info,
        client_version,
        client_python_version,
    )

    return background_task


@router.post("/status/function")
@router.post("/status/function/")
async def function_status(
    request: Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value, reason="bad JSON body"
        )

    resp = await _get_function_status(data, auth_info)
    return {
        "data": resp,
    }


@router.get("/build/status")
@router.get("/build/status/")
async def build_status(
    name: str = "",
    project: str = "",
    tag: str = "",
    offset: int = 0,
    logs: bool = True,
    last_log_timestamp: float = 0.0,
    verbose: bool = False,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        project or mlrun.mlconf.default_project,
        name,
        # store since with the current mechanism we update the status (and store the function) in the DB when a client
        # query for the status
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    fn = await run_in_threadpool(
        server.api.crud.Functions().get_function, db_session, name, project, tag
    )
    if not fn:
        server.api.api.utils.log_and_raise(
            HTTPStatus.NOT_FOUND.value, name=name, project=project, tag=tag
        )

    # nuclio deploy status
    if fn.get("kind") in RuntimeKinds.pure_nuclio_deployed_runtimes():
        return await run_in_threadpool(
            _handle_nuclio_deploy_status,
            db_session,
            auth_info,
            fn,
            name,
            project,
            tag,
            last_log_timestamp,
            verbose,
        )

    return await run_in_threadpool(
        _handle_job_deploy_status,
        db_session,
        fn,
        name,
        project,
        tag,
        offset,
        logs,
    )


def _handle_job_deploy_status(
    db_session: Session,
    fn: dict,
    name: str,
    project: str,
    tag: str,
    offset: int,
    logs: bool,
):
    # job deploy status
    function_state = get_in(fn, "status.state", "")
    pod = get_in(fn, "status.build_pod", "")
    image = get_in(fn, "spec.build.image", "")
    out = b""
    if not pod:
        if function_state == mlrun.common.schemas.FunctionState.ready:
            # when the function has been built we set the created image into the `spec.image` for reference see at the
            # end of the function where we resolve if the status is ready and then set the spec.build.image to
            # spec.image
            # TODO: spec shouldn't hold backend enriched attributes, but rather in the status block
            #   therefore need set it as a new attribute in status.image which will ease our resolution
            #   of whether it is a user defined image or MLRun enriched one.
            image = image or get_in(fn, "spec.image")
        return Response(
            content=out,
            media_type="text/plain",
            headers={
                "function_status": function_state,
                "function_image": image,
                "builder_pod": pod,
            },
        )

    # read from log file
    log_file = server.api.api.utils.log_path(
        project, f"build_{name}__{tag or 'latest'}"
    )
    if (
        function_state in mlrun.common.schemas.FunctionState.terminal_states()
        and log_file.exists()
    ):
        if function_state == mlrun.common.schemas.FunctionState.ready:
            # when the function has been built we set the created image into the `spec.image` for reference see at the
            # end of the function where we resolve if the status is ready and then set the spec.build.image to
            # spec.image
            # TODO: spec shouldn't hold backend enriched attributes, but rather in the status block
            #   therefore need set it as a new attribute in status.image which will ease our resolution
            #   of whether it is a user defined image or MLRun enriched one.
            image = image or get_in(fn, "spec.image")

        with log_file.open("rb") as fp:
            fp.seek(offset)
            out = fp.read()
        return Response(
            content=out,
            media_type="text/plain",
            headers={
                "x-mlrun-function-status": function_state,
                "function_status": function_state,
                "function_image": image,
                "builder_pod": pod,
            },
        )

    build_pod_state = server.api.utils.singletons.k8s.get_k8s_helper(
        silent=False
    ).get_pod_status(pod)
    logger.debug(
        "Resolved pod status",
        function_name=name,
        pod_status=build_pod_state,
        pod_name=pod,
    )

    normalized_pod_function_state = (
        mlrun.common.schemas.FunctionState.get_function_state_from_pod_state(
            build_pod_state
        )
    )
    if normalized_pod_function_state == mlrun.common.schemas.FunctionState.ready:
        logger.info(
            "Build completed successfully",
            function_name=name,
            pod=pod,
            pod_state=build_pod_state,
        )
    elif normalized_pod_function_state == mlrun.common.schemas.FunctionState.error:
        logger.error(
            "Build failed", function_name=name, pod_name=pod, pod_status=build_pod_state
        )

    if (
        (
            logs
            and normalized_pod_function_state
            != mlrun.common.schemas.FunctionState.pending
        )
        or normalized_pod_function_state
        in mlrun.common.schemas.FunctionState.terminal_states()
    ):
        try:
            resp = server.api.utils.singletons.k8s.get_k8s_helper(silent=False).logs(
                pod
            )
        except ApiException as exc:
            logger.warning(
                "Failed to get build logs",
                function_name=name,
                function_state=normalized_pod_function_state,
                pod=pod,
                exc_info=exc,
            )
            resp = ""

        if (
            normalized_pod_function_state
            in mlrun.common.schemas.FunctionState.terminal_states()
        ):
            # TODO: move to log collector
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("wb") as fp:
                fp.write(resp.encode())

        if resp and logs:
            # begin from the offset number and then encode
            out = resp[offset:].encode()

    # check if the previous function state is different from the current build pod state, if that is the case then
    # update the function and store to the database
    if function_state != normalized_pod_function_state:
        update_in(fn, "status.state", normalized_pod_function_state)

        versioned = False
        if normalized_pod_function_state == mlrun.common.schemas.FunctionState.ready:
            update_in(fn, "spec.image", image)
            versioned = True

        server.api.crud.Functions().store_function(
            db_session,
            fn,
            name,
            project,
            tag,
            versioned=versioned,
        )

    return Response(
        content=out,
        media_type="text/plain",
        headers={
            "x-mlrun-function-status": normalized_pod_function_state,
            "function_status": normalized_pod_function_state,
            "function_image": image,
            "builder_pod": pod,
        },
    )


def _parse_start_function_body(db_session, data):
    url = data.get("functionUrl")
    if not url:
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="Runtime error: functionUrl not specified",
        )

    project, name, tag, hash_key = parse_versioned_object_uri(url)
    runtime = server.api.crud.Functions().get_function(
        db_session, name, project, tag, hash_key
    )
    if not runtime:
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: function {url} not found",
        )

    return new_function(runtime=runtime)


async def _start_function_wrapper(
    function,
    auth_info: mlrun.common.schemas.AuthInfo,
    client_version: str = None,
    client_python_version: str = None,
):
    await run_in_threadpool(
        _start_function,
        function,
        auth_info,
        client_version,
        client_python_version,
    )


def _start_function(
    function,
    auth_info: mlrun.common.schemas.AuthInfo,
    client_version: str = None,
    client_python_version: str = None,
):
    db_session = server.api.db.session.create_session()
    try:
        run_db = server.api.api.utils.get_run_db_instance(db_session)
        function.set_db_connection(run_db)
        server.api.api.utils.apply_enrichment_and_validation_on_function(
            function,
            auth_info,
        )

        server.api.crud.Functions().start_function(
            function, client_version, client_python_version
        )
        logger.info("Fn:\n %s", function.to_yaml())

    except mlrun.errors.MLRunBadRequestError:
        raise

    except Exception as err:
        logger.error(traceback.format_exc())
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {err_to_str(err)}",
        )
    finally:
        server.api.db.session.close_session(db_session)


async def _get_function_status(data, auth_info: mlrun.common.schemas.AuthInfo):
    logger.info(f"Getting function status:\n{data}")
    selector = data.get("selector")
    kind = data.get("kind")
    if not selector or not kind:
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="Runtime error: selector or runtime kind not specified",
        )
    project, name = data.get("project"), data.get("name")

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    try:
        status = server.api.crud.Functions().get_function_status(
            kind,
            selector,
        )
        logger.info("Got function status", status=status)
        return status

    except mlrun.errors.MLRunBadRequestError:
        raise

    except Exception as err:
        logger.error(traceback.format_exc())
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {err_to_str(err)}",
        )
