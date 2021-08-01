import traceback
from distutils.util import strtobool
from http import HTTPStatus
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.background_tasks
import mlrun.api.utils.clients.opa
import mlrun.api.utils.singletons.project_member
from mlrun.api.api import deps
from mlrun.api.api.utils import get_run_db_instance, log_and_raise
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.builder import build_runtime
from mlrun.config import config
from mlrun.run import new_function
from mlrun.runtimes import RuntimeKinds, runtime_resources_map
from mlrun.runtimes.function import deploy_nuclio_function, get_nuclio_deploy_status
from mlrun.utils import get_in, logger, parse_versioned_object_uri, update_in

router = APIRouter()


# curl -d@/path/to/func.json http://localhost:8080/func/prj/7?tag=0.3.2
@router.post("/func/{project}/{name}")
async def store_function(
    request: Request,
    project: str,
    name: str,
    tag: str = "",
    versioned: bool = False,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_verifier.auth_info,
    )
    await run_in_threadpool(
        mlrun.api.utils.clients.opa.Client().query_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_verifier.auth_info,
    )
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.debug("Storing function", project=project, name=name, tag=tag, data=data)
    hash_key = await run_in_threadpool(
        mlrun.api.crud.Functions().store_function,
        db_session,
        data,
        name,
        project,
        tag=tag,
        versioned=versioned,
    )
    return {
        "hash_key": hash_key,
    }


# curl http://localhost:8080/log/prj/7?tag=0.2.3
@router.get("/func/{project}/{name}")
def get_function(
    project: str,
    name: str,
    tag: str = "",
    hash_key="",
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_verifier.auth_info,
    )
    func = mlrun.api.crud.Functions().get_function(
        db_session, name, project, tag, hash_key
    )
    return {
        "func": func,
    }


@router.delete(
    "/projects/{project}/functions/{name}", status_code=HTTPStatus.NO_CONTENT.value
)
def delete_function(
    project: str,
    name: str,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_verifier.auth_info,
    )
    mlrun.api.crud.Functions().delete_function(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@router.get("/funcs")
def list_functions(
    project: str = config.default_project,
    name: str = None,
    tag: str = None,
    labels: List[str] = Query([], alias="label"),
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    functions = mlrun.api.crud.Functions().list_functions(
        db_session, project, name, tag, labels
    )
    functions = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        functions,
        lambda function: (
            function.get("metadata", {}).get("project", mlrun.mlconf.default_project),
            function["metadata"]["name"],
        ),
        auth_verifier.auth_info,
    )
    return {
        "funcs": functions,
    }


# curl -d@/path/to/job.json http://localhost:8080/build/function
@router.post("/build/function")
@router.post("/build/function/")
async def build_function(
    request: Request,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info(f"build_function:\n{data}")
    function = data.get("function")
    await run_in_threadpool(
        mlrun.api.utils.clients.opa.Client().query_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        function.get("metadata", {}).get("project", mlrun.mlconf.default_project),
        function.get("metadata", {}).get("name"),
        mlrun.api.schemas.AuthorizationAction.update,
        auth_verifier.auth_info,
    )
    with_mlrun = strtobool(data.get("with_mlrun", "on"))
    skip_deployed = data.get("skip_deployed", False)
    mlrun_version_specifier = data.get("mlrun_version_specifier")
    fn, ready = await run_in_threadpool(
        _build_function,
        db_session,
        auth_verifier.auth_info,
        function,
        with_mlrun,
        skip_deployed,
        mlrun_version_specifier,
    )
    return {
        "data": fn.to_dict(),
        "ready": ready,
    }


# curl -d@/path/to/job.json http://localhost:8080/start/function
@router.post("/start/function", response_model=mlrun.api.schemas.BackgroundTask)
@router.post("/start/function/", response_model=mlrun.api.schemas.BackgroundTask)
async def start_function(
    request: Request,
    background_tasks: BackgroundTasks,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    # TODO: ensure project here !!! for background task
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info("Got request to start function", body=data)

    function = await run_in_threadpool(_parse_start_function_body, db_session, data)
    await run_in_threadpool(
        mlrun.api.utils.clients.opa.Client().query_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        function.metadata.project,
        function.metadata.name,
        mlrun.api.schemas.AuthorizationAction.update,
        auth_verifier.auth_info,
    )

    background_task = await run_in_threadpool(
        mlrun.api.utils.background_tasks.Handler().create_background_task,
        function.metadata.project,
        background_tasks,
        _start_function,
        function,
        auth_verifier.auth_info,
    )

    return background_task


# curl -d@/path/to/job.json http://localhost:8080/status/function
@router.post("/status/function")
@router.post("/status/function/")
async def function_status(
    request: Request, auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    resp = await run_in_threadpool(_get_function_status, data, auth_verifier.auth_info)
    return {
        "data": resp,
    }


# curl -d@/path/to/job.json http://localhost:8080/build/status
@router.get("/build/status")
@router.get("/build/status/")
def build_status(
    name: str = "",
    project: str = "",
    tag: str = "",
    offset: int = 0,
    logs: bool = True,
    last_log_timestamp: float = 0.0,
    verbose: bool = False,
    auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project or mlrun.mlconf.default_project,
        name,
        # store since with the current mechanism we update the status (and store the function) in the DB when a client
        # query for the status
        mlrun.api.schemas.AuthorizationAction.store,
        auth_verifier.auth_info,
    )
    fn = mlrun.api.crud.Functions().get_function(db_session, name, project, tag)
    if not fn:
        log_and_raise(HTTPStatus.NOT_FOUND.value, name=name, project=project, tag=tag)

    # nuclio deploy status
    if fn.get("kind") in RuntimeKinds.nuclio_runtimes():
        (
            state,
            address,
            nuclio_name,
            last_log_timestamp,
            text,
            status,
        ) = get_nuclio_deploy_status(
            name,
            project,
            tag,
            last_log_timestamp=last_log_timestamp,
            verbose=verbose,
            auth_info=auth_verifier.auth_info,
        )
        if state == "ready":
            logger.info("Nuclio function deployed successfully", name=name)
        if state in ["error", "unhealthy"]:
            logger.error(f"Nuclio deploy error, {text}", name=name)

        internal_invocation_urls = status.get("internalInvocationUrls", [])
        external_invocation_urls = status.get("externalInvocationUrls", [])

        # on earlier versions of mlrun, address used to represent the nodePort external invocation url
        # now that functions can be not exposed (using service_type clusterIP) this no longer relevant
        # and hence, for BC it would be filled with the external invocation url first item
        # or completely empty.
        address = external_invocation_urls[0] if external_invocation_urls else ""

        update_in(fn, "status.nuclio_name", nuclio_name)
        update_in(fn, "status.internal_invocation_urls", internal_invocation_urls)
        update_in(fn, "status.external_invocation_urls", external_invocation_urls)
        update_in(fn, "status.state", state)
        update_in(fn, "status.address", address)

        versioned = False
        if state == "ready":
            # Versioned means the version will be saved in the DB forever, we don't want to spam
            # the DB with intermediate or unusable versions, only successfully deployed versions
            versioned = True
        mlrun.api.crud.Functions().store_function(
            db_session, fn, name, project, tag, versioned=versioned,
        )
        return Response(
            content=text,
            media_type="text/plain",
            headers={
                "x-mlrun-function-status": state,
                "x-mlrun-last-timestamp": str(last_log_timestamp),
                "x-mlrun-address": address,
                "x-mlrun-internal-invocation-urls": ",".join(internal_invocation_urls),
                "x-mlrun-external-invocation-urls": ",".join(external_invocation_urls),
                "x-mlrun-name": nuclio_name,
            },
        )

    # job deploy status
    state = get_in(fn, "status.state", "")
    pod = get_in(fn, "status.build_pod", "")
    image = get_in(fn, "spec.build.image", "")
    out = b""
    if not pod:
        if state == "ready":
            image = image or get_in(fn, "spec.image")
        return Response(
            content=out,
            media_type="text/plain",
            headers={
                "function_status": state,
                "function_image": image,
                "builder_pod": pod,
            },
        )

    logger.info(f"get pod {pod} status")
    state = get_k8s().get_pod_status(pod)
    logger.info(f"pod state={state}")

    if state == "succeeded":
        logger.info("build completed successfully")
        state = mlrun.api.schemas.FunctionState.ready
    if state in ["failed", "error"]:
        logger.error(f"build {state}, watch the build pod logs: {pod}")
        state = mlrun.api.schemas.FunctionState.error

    if logs and state != "pending":
        resp = get_k8s().logs(pod)
        if resp:
            out = resp.encode()[offset:]

    update_in(fn, "status.state", state)
    if state == mlrun.api.schemas.FunctionState.ready:
        update_in(fn, "spec.image", image)

    versioned = False
    if state == mlrun.api.schemas.FunctionState.ready:
        versioned = True
    mlrun.api.crud.Functions().store_function(
        db_session, fn, name, project, tag, versioned=versioned,
    )

    return Response(
        content=out,
        media_type="text/plain",
        headers={
            "x-mlrun-function-status": state,
            "function_status": state,
            "function_image": image,
            "builder_pod": pod,
        },
    )


def _build_function(
    db_session,
    auth_info: mlrun.api.schemas.AuthInfo,
    function,
    with_mlrun,
    skip_deployed,
    mlrun_version_specifier,
):
    fn = None
    ready = None
    try:
        fn = new_function(runtime=function)
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
    try:
        run_db = get_run_db_instance(db_session, auth_info)
        fn.set_db_connection(run_db)
        fn.save(versioned=False)
        if fn.kind in RuntimeKinds.nuclio_runtimes():
            mlrun.api.api.utils.ensure_function_has_auth_set(fn, auth_info)
            deploy_nuclio_function(fn, auth_info=auth_info)
            # deploy only start the process, the get status API is used to check readiness
            ready = False
        else:
            ready = build_runtime(
                fn, with_mlrun, mlrun_version_specifier, skip_deployed
            )
        fn.save(versioned=True)
        logger.info("Fn:\n %s", fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
    return fn, ready


def _parse_start_function_body(db_session, data):
    url = data.get("functionUrl")
    if not url:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="runtime error: functionUrl not specified",
        )

    project, name, tag, hash_key = parse_versioned_object_uri(url)
    runtime = mlrun.api.crud.Functions().get_function(
        db_session, name, project, tag, hash_key
    )
    if not runtime:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"runtime error: function {url} not found",
        )

    return new_function(runtime=runtime)


def _start_function(function, auth_info: mlrun.api.schemas.AuthInfo):
    db_session = mlrun.api.db.session.create_session()
    try:
        resource = runtime_resources_map.get(function.kind)
        if "start" not in resource:
            log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason="runtime error: 'start' not supported by this runtime",
            )
        try:
            run_db = get_run_db_instance(db_session, auth_info)
            function.set_db_connection(run_db)
            mlrun.api.api.utils.ensure_function_has_auth_set(function, auth_info)
            #  resp = resource["start"](fn)  # TODO: handle resp?
            resource["start"](function)
            function.save(versioned=False)
            logger.info("Fn:\n %s", function.to_yaml())
        except Exception as err:
            logger.error(traceback.format_exc())
            log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
    finally:
        mlrun.api.db.session.close_session(db_session)


def _get_function_status(data, auth_info: mlrun.api.schemas.AuthInfo):
    logger.info(f"function_status:\n{data}")
    selector = data.get("selector")
    kind = data.get("kind")
    if not selector or not kind:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="runtime error: selector or runtime kind not specified",
        )
    project, name = data.get("project"), data.get("name")
    # Only after 0.6.6 the client start sending the project and name, as long as 0.6.6 is a valid version we'll need
    # to try and resolve them from the selector. TODO: remove this when 0.6.6 is not relevant anymore
    if not project or not name:
        project, name, _ = mlrun.runtimes.utils.parse_function_selector(selector)

    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )

    resource = runtime_resources_map.get(kind)
    if "status" not in resource:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="runtime error: 'status' not supported by this runtime",
        )

    resp = None
    try:
        resp = resource["status"](selector)
        logger.info("status: %s", resp)
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
