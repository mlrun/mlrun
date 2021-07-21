import json
import os
import traceback
from distutils.util import strtobool
from http import HTTPStatus
from typing import List

import v3io
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.background_tasks
from mlrun.api.api import deps
from mlrun.api.api.utils import get_run_db_instance, log_and_raise
from mlrun.api.crud.model_endpoints import ModelEndpoints
from mlrun.api.schemas import (
    ModelEndpoint,
    ModelEndpointMetadata,
    ModelEndpointStatus,
    ModelEndpointSpec,
)
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.builder import build_runtime
from mlrun.config import config
from mlrun.run import new_function
from mlrun.runtimes import (
    RuntimeKinds,
    runtime_resources_map,
    ServingRuntime,
)
from mlrun.runtimes.function import (
    deploy_nuclio_function,
    get_nuclio_deploy_status,
    resolve_function_internal_invocation_url,
)
from mlrun.utils import get_in, logger, parse_versioned_object_uri, update_in
from mlrun.utils.model_monitoring import parse_model_endpoint_store_prefix

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
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.debug(data)
    logger.info("store function: project=%s, name=%s, tag=%s", project, name, tag)
    hash_key = await run_in_threadpool(
        get_db().store_function,
        db_session,
        data,
        name,
        project,
        tag=tag,
        versioned=versioned,
        leader_session=auth_verifier.auth_info.session,
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
    db_session: Session = Depends(deps.get_db_session),
):
    func = get_db().get_function(db_session, name, project, tag, hash_key)
    return {
        "func": func,
    }


@router.delete(
    "/projects/{project}/functions/{name}", status_code=HTTPStatus.NO_CONTENT.value
)
def delete_function(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session),
):
    get_db().delete_function(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@router.get("/funcs")
def list_functions(
    project: str = config.default_project,
    name: str = None,
    tag: str = None,
    labels: List[str] = Query([], alias="label"),
    db_session: Session = Depends(deps.get_db_session),
):
    funcs = get_db().list_functions(db_session, name, project, tag, labels)
    return {
        "funcs": list(funcs),
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
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info("Got request to start function", body=data)

    function = await run_in_threadpool(_parse_start_function_body, db_session, data)

    background_task = await run_in_threadpool(
        mlrun.api.utils.background_tasks.Handler().create_background_task,
        db_session,
        auth_verifier.auth_info.session,
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
async def function_status(request: Request):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    resp = await run_in_threadpool(_get_function_status, data)
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
    fn = get_db().get_function(db_session, name, project, tag)
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
            name, project, tag, last_log_timestamp=last_log_timestamp, verbose=verbose
        )
        if state == "ready":
            logger.info("Nuclio function deployed successfully", name=name)
        if state in ["error", "unhealthy"]:
            logger.error(f"Nuclio deploy error, {text}", name=name)

        # internal / external invocation urls were added on nuclio 1.6.x
        # and hence, it might be empty
        # to backward compatible with older nuclio versions, we use hard-coded default values
        internal_invocation_urls = status.get(
            "internalInvocationUrls", [resolve_function_internal_invocation_url(name)]
        )
        external_invocation_urls = status.get(
            "externalInvocationUrls", [address] if address else []
        )

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
        get_db().store_function(
            db_session,
            fn,
            name,
            project,
            tag,
            versioned=versioned,
            leader_session=auth_verifier.auth_info.session,
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
    get_db().store_function(
        db_session,
        fn,
        name,
        project,
        tag,
        versioned=versioned,
        leader_session=auth_verifier.auth_info.session,
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

        run_db = get_run_db_instance(db_session, auth_info.session)
        fn.set_db_connection(run_db)
        fn.save(versioned=False)
        if fn.kind in RuntimeKinds.nuclio_runtimes():
            mlrun.api.api.utils.ensure_function_has_auth_set(fn, auth_info)

            if fn.kind == RuntimeKinds.serving:
                # Handle model monitoring
                try:
                    _init_model_monitoring_endpoint_records(
                        fn, db_session=db_session, auth_info=auth_info,
                    )
                    if fn.spec.track_models:
                        logger.info("Tracking enabled, initializing model monitoring")
                        _init_serving_function_stream_args(fn=fn)
                        _create_model_monitoring_stream(project=fn.metadata.project)
                        ModelEndpoints.deploy_monitoring_functions(
                            project=fn.metadata.project,
                            db_session=db_session,
                            auth_info=auth_info,
                        )
                except Exception as e:
                    logger.exception(e)

            deploy_nuclio_function(fn)
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
    runtime = get_db().get_function(db_session, name, project, tag, hash_key)
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
            run_db = get_run_db_instance(db_session, auth_info.session)
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


def _get_function_status(data):
    logger.info(f"function_status:\n{data}")
    selector = data.get("selector")
    kind = data.get("kind")
    if not selector or not kind:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="runtime error: selector or runtime kind not specified",
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


def _create_model_monitoring_stream(project: str):

    stream_path = config.model_endpoint_monitoring.store_prefixes.default.format(
        project=project, kind="stream"
    )

    _, container, stream_path = parse_model_endpoint_store_prefix(stream_path)

    # TODO: How should we configure sharding here?
    logger.info(
        "Creating endpoint stream",
        stream_path=stream_path,
        container=container,
        endpoint=config.v3io_api,
    )

    v3io_client = v3io.dataplane.Client(
        endpoint=config.v3io_api, access_key=os.environ.get("V3IO_ACCESS_KEY")
    )
    response = v3io_client.create_stream(
        container=container,
        path=stream_path,
        shard_count=1,
        retention_period_hours=24,
        raise_for_status=v3io.dataplane.RaiseForStatus.never,
    )

    if not (response.status_code == 400 and "ResourceInUse" in str(response.body)):
        response.raise_for_status([409, 204])


def _init_model_monitoring_endpoint_records(
    fn: ServingRuntime, db_session, auth_info: mlrun.api.schemas.AuthInfo
):

    stream_path = config.model_endpoint_monitoring.store_prefixes.default.format(
        project=fn.metadata.project, kind="stream"
    )

    for model_name, values in fn.spec.graph.routes.items():
        class_args = values.class_args or {}
        model_path = class_args.get("model_path", None)
        if not model_path:
            continue

        try:
            path_parts = model_path.split("/")
            file_name = path_parts[-1]
            _, version = file_name.split(":")
            if version:
                model_name = f"{model_name}:{version}"
        except ValueError:
            pass

        try:
            model_endpoint = ModelEndpoint(
                metadata=ModelEndpointMetadata(
                    project=fn.metadata.project, labels=fn.metadata.labels
                ),
                spec=ModelEndpointSpec(
                    function_uri=fn.uri,
                    model=model_name,
                    model_class=values.class_name,
                    model_uri=model_path,
                    stream_path=stream_path,
                    active=True,
                ),
                status=ModelEndpointStatus(),
            )

            ModelEndpoints.create_or_patch(
                db_session=db_session,
                access_key=os.environ.get("V3IO_ACCESS_KEY"),
                model_endpoint=model_endpoint,
                leader_session=auth_info.session,
            )
        except Exception as e:
            logger.error("Failed to create endpoint record", exc=e)


def _init_serving_function_stream_args(fn: ServingRuntime):
    logger.debug("Initializing serving function stream args")
    if "stream_args" in fn.spec.parameters:
        logger.debug("stream args found in function spec parameters")
        if "access_key" not in fn.spec.parameters["stream_args"]:
            logger.debug("pipelines access key added to stream args")
            fn.spec.parameters["stream_args"]["access_key"] = os.environ.get(
                "V3IO_ACCESS_KEY"
            )
    else:
        logger.debug("pipelines access key added to stream args")
        fn.spec.parameters["stream_args"] = {
            "access_key": os.environ.get("V3IO_ACCESS_KEY")
        }

    fn.save(versioned=True)
