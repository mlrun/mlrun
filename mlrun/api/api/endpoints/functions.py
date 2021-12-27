import base64  # noqa: F401
import os
import traceback
from distutils.util import strtobool
from http import HTTPStatus
from typing import List

import v3io
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.background_tasks
import mlrun.api.utils.singletons.project_member
from mlrun.api.api import deps
from mlrun.api.api.utils import get_run_db_instance, log_and_raise
from mlrun.api.crud.secrets import Secrets
from mlrun.api.schemas import SecretProviderName, SecretsData
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.builder import build_runtime
from mlrun.config import config
from mlrun.errors import MLRunRuntimeError
from mlrun.run import new_function
from mlrun.runtimes import RuntimeKinds, ServingRuntime, runtime_resources_map
from mlrun.runtimes.function import deploy_nuclio_function, get_nuclio_deploy_status
from mlrun.runtimes.utils import get_item_name
from mlrun.utils import get_in, logger, parse_versioned_object_uri, update_in
from mlrun.utils.model_monitoring import parse_model_endpoint_store_prefix

router = APIRouter()


@router.post("/func/{project}/{name}")
async def store_function(
    request: Request,
    project: str,
    name: str,
    tag: str = "",
    versioned: bool = False,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )
    await run_in_threadpool(
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_info,
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


@router.get("/func/{project}/{name}")
def get_function(
    project: str,
    name: str,
    tag: str = "",
    hash_key="",
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    func = mlrun.api.crud.Functions().get_function(
        db_session, name, project, tag, hash_key
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
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
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    mlrun.api.crud.Functions().delete_function(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/funcs")
def list_functions(
    project: str = None,
    name: str = None,
    tag: str = None,
    labels: List[str] = Query([], alias="label"),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    if project is None:
        project = config.default_project
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project, mlrun.api.schemas.AuthorizationAction.read, auth_info,
    )
    functions = mlrun.api.crud.Functions().list_functions(
        db_session, project, name, tag, labels
    )
    functions = mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        functions,
        lambda function: (
            function.get("metadata", {}).get("project", mlrun.mlconf.default_project),
            function["metadata"]["name"],
        ),
        auth_info,
    )
    return {
        "funcs": functions,
    }


@router.post("/build/function")
@router.post("/build/function/")
async def build_function(
    request: Request,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
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
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        function.get("metadata", {}).get("project", mlrun.mlconf.default_project),
        auth_info=auth_info,
    )
    await run_in_threadpool(
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        function.get("metadata", {}).get("project", mlrun.mlconf.default_project),
        function.get("metadata", {}).get("name"),
        mlrun.api.schemas.AuthorizationAction.update,
        auth_info,
    )
    if isinstance(data.get("with_mlrun"), bool):
        with_mlrun = data.get("with_mlrun")
    else:
        with_mlrun = strtobool(data.get("with_mlrun", "on"))
    skip_deployed = data.get("skip_deployed", False)
    mlrun_version_specifier = data.get("mlrun_version_specifier")
    fn, ready = await run_in_threadpool(
        _build_function,
        db_session,
        auth_info,
        function,
        with_mlrun,
        skip_deployed,
        mlrun_version_specifier,
        data.get("builder_env"),
    )
    return {
        "data": fn.to_dict(),
        "ready": ready,
    }


@router.post("/start/function", response_model=mlrun.api.schemas.BackgroundTask)
@router.post("/start/function/", response_model=mlrun.api.schemas.BackgroundTask)
async def start_function(
    request: Request,
    background_tasks: BackgroundTasks,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
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
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        function.metadata.project,
        function.metadata.name,
        mlrun.api.schemas.AuthorizationAction.update,
        auth_info,
    )

    background_task = await run_in_threadpool(
        mlrun.api.utils.background_tasks.Handler().create_project_background_task,
        function.metadata.project,
        background_tasks,
        _start_function,
        function,
        auth_info,
    )

    return background_task


@router.post("/status/function")
@router.post("/status/function/")
async def function_status(
    request: Request,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    resp = await run_in_threadpool(_get_function_status, data, auth_info)
    return {
        "data": resp,
    }


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
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
):
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.function,
        project or mlrun.mlconf.default_project,
        name,
        # store since with the current mechanism we update the status (and store the function) in the DB when a client
        # query for the status
        mlrun.api.schemas.AuthorizationAction.store,
        auth_info,
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
            auth_info=auth_info,
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
    with_mlrun=True,
    skip_deployed=False,
    mlrun_version_specifier=None,
    builder_env=None,
):
    fn = None
    ready = None
    try:
        fn = new_function(runtime=function)
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
    try:
        run_db = get_run_db_instance(db_session)
        fn.set_db_connection(run_db)
        fn.save(versioned=False)
        if fn.kind in RuntimeKinds.nuclio_runtimes():
            mlrun.api.api.utils.ensure_function_has_auth_set(fn, auth_info)
            mlrun.api.api.utils.process_function_service_account(fn)

            if fn.kind == RuntimeKinds.serving:
                # Handle model monitoring
                try:
                    if fn.spec.track_models:
                        logger.info("Tracking enabled, initializing model monitoring")
                        _init_serving_function_stream_args(fn=fn)
                        model_monitoring_access_key = _process_model_monitoring_secret(
                            db_session,
                            fn.metadata.project,
                            "MODEL_MONITORING_ACCESS_KEY",
                        )

                        _create_model_monitoring_stream(project=fn.metadata.project)
                        mlrun.api.crud.ModelEndpoints().deploy_monitoring_functions(
                            project=fn.metadata.project,
                            model_monitoring_access_key=model_monitoring_access_key,
                            db_session=db_session,
                            auth_info=auth_info,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed deploying model monitoring infrastructure for the project",
                        project=fn.metadata.project,
                        exc=exc,
                        traceback=traceback.format_exc(),
                    )

            deploy_nuclio_function(fn, auth_info=auth_info)
            # deploy only start the process, the get status API is used to check readiness
            ready = False
        else:
            ready = build_runtime(
                auth_info,
                fn,
                with_mlrun,
                mlrun_version_specifier,
                skip_deployed,
                builder_env=builder_env,
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
            run_db = get_run_db_instance(db_session)
            function.set_db_connection(run_db)
            mlrun.api.api.utils.ensure_function_has_auth_set(function, auth_info)
            mlrun.api.api.utils.process_function_service_account(function)
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

    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
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


def _create_model_monitoring_stream(project: str):

    stream_path = config.model_endpoint_monitoring.store_prefixes.default.format(
        project=project, kind="stream"
    )

    _, container, stream_path = parse_model_endpoint_store_prefix(stream_path)

    # TODO: How should we configure sharding here?
    logger.info(
        "Creating model endpoint stream for project",
        project=project,
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
        shard_count=config.model_endpoint_monitoring.serving_stream_args.shard_count,
        retention_period_hours=config.model_endpoint_monitoring.serving_stream_args.retention_period_hours,
        raise_for_status=v3io.dataplane.RaiseForStatus.never,
    )

    if not (response.status_code == 400 and "ResourceInUse" in str(response.body)):
        response.raise_for_status([409, 204])


def _init_serving_function_stream_args(fn: ServingRuntime):
    logger.debug("Initializing serving function stream args")
    if "stream_args" in fn.spec.parameters:
        logger.debug("Adding access key to pipelines stream args")
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


def _get_function_env_var(fn: ServingRuntime, var_name: str):
    for env_var in fn.spec.env:
        if get_item_name(env_var) == var_name:
            return env_var
    return None


def _process_model_monitoring_secret(db_session, project_name: str, secret_key: str):
    # The expected result of this method is an access-key placed in an internal project-secret.
    # If the user provided an access-key as the "regular" secret_key, then we delete this secret and move contents
    # to the internal secret instead. Else, if the internal secret already contained a value, keep it. Last option
    # (which is the recommended option for users) is to retrieve a new access-key from the project owner and use it.
    logger.info(
        "Getting project secret", project_name=project_name, namespace=config.namespace
    )

    provider = SecretProviderName.kubernetes
    secret_value = Secrets().get_secret(
        project_name, provider, secret_key, allow_secrets_from_k8s=True,
    )
    user_provided_key = secret_value is not None
    internal_key_name = Secrets().generate_model_monitoring_secret_key(secret_key)

    if not user_provided_key:
        secret_value = Secrets().get_secret(
            project_name,
            provider,
            internal_key_name,
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
        )
        if not secret_value:
            import mlrun.api.utils.singletons.project_member

            project_owner = mlrun.api.utils.singletons.project_member.get_project_member().get_project_owner(
                db_session, project_name
            )

            secret_value = project_owner.session
            if not secret_value:
                raise MLRunRuntimeError(
                    f"No model monitoring access key. Failed to generate one for owner of project {project_name}",
                )

            logger.info(
                "Filling model monitoring access-key from project owner",
                project_name=project_name,
                project_owner=project_owner.username,
            )

    secrets = SecretsData(provider=provider, secrets={internal_key_name: secret_value})
    Secrets().store_secrets(project_name, secrets, allow_internal_secrets=True)
    if user_provided_key:
        logger.info(
            "Deleting user-provided access-key - replaced with an internal secret"
        )
        Secrets().delete_secret(project_name, provider, secret_key)

    return secret_value
