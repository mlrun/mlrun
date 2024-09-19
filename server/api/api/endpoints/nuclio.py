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
#
import asyncio
import traceback
import typing
from http import HTTPStatus

import semver
import sqlalchemy.orm
from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import server.api.api.utils
import server.api.crud.model_monitoring.deployment
import server.api.crud.runtimes.nuclio.function
import server.api.db.session
import server.api.launcher
import server.api.utils.auth.verifier
import server.api.utils.clients.async_nuclio
import server.api.utils.clients.chief
import server.api.utils.singletons.project_member
from mlrun.common.model_monitoring.helpers import parse_model_endpoint_store_prefix
from mlrun.utils import logger
from mlrun.utils.helpers import generate_object_uri
from server.api import MINIMUM_CLIENT_VERSION_FOR_MM
from server.api.api import deps
from server.api.crud.secrets import Secrets, SecretsClientType

router = APIRouter()


@router.get(
    "/projects/{project}/api-gateways",
    response_model=mlrun.common.schemas.APIGatewaysOutput,
    response_model_exclude_none=True,
)
async def list_api_gateways(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        api_gateways = await client.list_api_gateways(project)

    allowed_api_gateways = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateway,
        list(api_gateways.keys()) if api_gateways else [],
        lambda _api_gateway: (
            project,
            _api_gateway,
        ),
        auth_info,
    )
    allowed_api_gateways = {
        api_gateway: api_gateways[api_gateway] for api_gateway in allowed_api_gateways
    }
    return mlrun.common.schemas.APIGatewaysOutput(api_gateways=allowed_api_gateways)


@router.get(
    "/projects/{project}/api-gateways/{gateway}",
    response_model=mlrun.common.schemas.APIGateway,
    response_model_exclude_none=True,
)
async def get_api_gateway(
    project: str,
    gateway: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateway,
        project,
        gateway,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        api_gateway = await client.get_api_gateway(project_name=project, name=gateway)

    return api_gateway


@router.put(
    "/projects/{project}/api-gateways/{name}",
    response_model=mlrun.common.schemas.APIGateway,
    response_model_exclude_none=True,
)
async def store_api_gateway(
    project: str,
    name: str,
    api_gateway: mlrun.common.schemas.APIGateway,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateway,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        create = False
        try:
            existing_api_gateway = await client.get_api_gateway(
                project_name=project, name=name
            )
            # check if any functions were removed from the api gateway
            unused_functions = [
                func
                for func in existing_api_gateway.get_function_names()
                if func not in api_gateway.get_function_names()
            ]
            # if invocation URL has changed, delete URL from all the functions
            if existing_api_gateway.get_invoke_url != api_gateway.get_invoke_url:
                await _delete_functions_external_invocation_url(
                    project=project,
                    url=existing_api_gateway.get_invoke_url(),
                    function_names=existing_api_gateway.get_function_names(),
                )
            # if only functions list has changed, then delete URL only from those functions
            # which are not used in api gateway anymore
            elif unused_functions:
                # delete api gateway url from those functions which are not used in api gateway anymore
                await _delete_functions_external_invocation_url(
                    project=project,
                    url=existing_api_gateway.get_invoke_url(),
                    function_names=unused_functions,
                )

        except mlrun.errors.MLRunNotFoundError:
            create = True

        await client.store_api_gateway(
            project_name=project, api_gateway=api_gateway, create=create
        )
        api_gateway = await client.get_api_gateway(
            name=name,
            project_name=project,
        )
    if api_gateway:
        await _add_functions_external_invocation_url(
            project=project,
            url=api_gateway.get_invoke_url(),
            function_names=api_gateway.get_function_names(),
        )
    return api_gateway


@router.delete(
    "/projects/{project}/api-gateways/{name}",
)
async def delete_api_gateway(
    project: str,
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.api_gateway,
        project,
        name,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        api_gateway = await client.get_api_gateway(project_name=project, name=name)

        if api_gateway:
            await _delete_functions_external_invocation_url(
                project=project,
                url=api_gateway.get_invoke_url(),
                function_names=api_gateway.get_function_names(),
            )
            return await client.delete_api_gateway(project_name=project, name=name)
        return await client.delete_api_gateway(project_name=project, name=name)


@router.post("/projects/{project}/nuclio/{name}/deploy")
async def deploy_function(
    project: str,
    name: str,
    request: Request,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: sqlalchemy.orm.Session = Depends(deps.get_db_session),
    client_version: typing.Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
    client_python_version: typing.Optional[str] = Header(
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

    logger.info("Deploying function", data=data, project=project, name=name)
    function = data.get("function")
    function.setdefault("metadata", {})
    function["metadata"]["name"] = name
    function["metadata"]["project"] = project
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
                name=name,
                project=project,
                function=function,
            )
            chief_client = server.api.utils.clients.chief.Client()
            return await chief_client.build_function(request=request, json=data)

    fn = await run_in_threadpool(
        _deploy_function,
        db_session,
        auth_info,
        project,
        name,
        function,
        data.get("builder_env"),
        client_version,
        client_python_version,
    )

    return {
        "data": fn.to_dict(),
    }


@router.get("/projects/{project}/nuclio/{name}/deploy")
async def deploy_status(
    project: str = "",
    name: str = "",
    tag: str = "",
    last_log_timestamp: float = 0.0,
    verbose: bool = False,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: sqlalchemy.orm.Session = Depends(deps.get_db_session),
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

    if fn.get("kind") not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime kind {fn.kind} is not a nuclio runtime",
        )
    api_gateways_urls = await _get_api_gateways_urls_for_function(
        auth_info, project, name, tag
    )
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
        api_gateways_urls,
    )


def process_model_monitoring_secret(
    db_session, project_name: str, secret_key: str, store: bool = True
):
    # The expected result of this method is an access-key placed in an internal project-secret.
    # If the user provided an access-key as the "regular" secret_key, then we delete this secret and move contents
    # to the internal secret instead. Else, if the internal secret already contained a value, keep it. Last option
    # (which is the recommended option for users) is to retrieve a new access-key from the project owner and use it.
    logger.info(
        "Getting project secret",
        project_name=project_name,
        namespace=mlrun.mlconf.namespace,
    )
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    secret_value = Secrets().get_project_secret(
        project_name,
        provider,
        secret_key,
        allow_secrets_from_k8s=True,
    )
    user_provided_key = secret_value is not None
    internal_key_name = Secrets().generate_client_project_secret_key(
        SecretsClientType.model_monitoring, secret_key
    )

    if not user_provided_key:
        secret_value = Secrets().get_project_secret(
            project_name,
            provider,
            internal_key_name,
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
        )
        if not secret_value:
            try:
                project_owner = server.api.utils.singletons.project_member.get_project_member().get_project_owner(
                    db_session, project_name
                )
            except mlrun.errors.MLRunNotFoundError:
                logger.debug(
                    "Failed to retrieve project owner, the project does not exist in Iguazio.",
                    project_name=project_name,
                )
                raise

            secret_value = project_owner.access_key
            if not secret_value:
                raise mlrun.errors.MLRunRuntimeError(
                    f"No model monitoring access key. Failed to generate one for owner of project {project_name}",
                )

            logger.info(
                "Filling model monitoring access-key from project owner",
                project_name=project_name,
                project_owner=project_owner.username,
            )
    if store:
        secrets = mlrun.common.schemas.SecretsData(
            provider=provider, secrets={internal_key_name: secret_value}
        )
        Secrets().store_project_secrets(
            project_name, secrets, allow_internal_secrets=True
        )
        if user_provided_key:
            logger.info(
                "Deleting user-provided access-key - replaced with an internal secret"
            )
            Secrets().delete_project_secret(project_name, provider, secret_key)

    return secret_value


def create_model_monitoring_stream(
    project: str,
    stream_path: str,
    access_key: str = None,
    stream_args: dict = None,
):
    if stream_path.startswith("v3io://"):
        import v3io.dataplane

        _, container, stream_path = parse_model_endpoint_store_prefix(stream_path)

        # TODO: How should we configure sharding here?
        logger.info(
            "Creating stream",
            project=project,
            stream_path=stream_path,
            container=container,
            endpoint=mlrun.mlconf.v3io_api,
        )

        v3io_client = v3io.dataplane.Client(
            endpoint=mlrun.mlconf.v3io_api, access_key=access_key
        )

        response = v3io_client.stream.create(
            container=container,
            stream_path=stream_path,
            shard_count=stream_args.shard_count,
            retention_period_hours=stream_args.retention_period_hours,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
            access_key=access_key,
        )

        if not (response.status_code == 400 and "ResourceInUse" in str(response.body)):
            response.raise_for_status([409, 204])


def _deploy_function(
    db_session: sqlalchemy.orm.Session,
    auth_info: mlrun.common.schemas.AuthInfo,
    project: str,
    name: str,
    function: dict,
    builder_env: dict,
    client_version: str,
    client_python_version: str,
):
    fn = None
    try:
        fn = mlrun.new_function(runtime=function, project=project, name=name)
    except Exception as err:
        logger.error(traceback.format_exc())
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {mlrun.errors.err_to_str(err)}",
        )

    if fn.kind not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime kind {fn.kind} is not a nuclio runtime",
        )

    fn: mlrun.runtimes.RemoteRuntime
    try:
        # Connect to run db
        run_db = server.api.api.utils.get_run_db_instance(db_session)
        fn.set_db_connection(run_db)

        # Enrich runtime
        launcher = server.api.launcher.ServerSideLauncher(auth_info=auth_info)
        launcher.enrich_runtime(runtime=fn, full=True)

        fn.pre_deploy_validation()
        fn.save(versioned=False)

        fn = _deploy_nuclio_runtime(
            auth_info,
            builder_env,
            client_python_version,
            client_version,
            db_session,
            fn,
        )
        fn.save(versioned=True)
        logger.info("Resolved function", fn=fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {mlrun.errors.err_to_str(err)}",
        )
    return fn


def _deploy_nuclio_runtime(
    auth_info, builder_env, client_python_version, client_version, db_session, fn
):
    monitoring_application = (
        fn.metadata.labels.get(mm_constants.ModelMonitoringAppLabel.KEY)
        == mm_constants.ModelMonitoringAppLabel.VAL
    )
    serving_to_monitor = (
        fn.kind == mlrun.runtimes.RuntimeKinds.serving and fn.spec.track_models
    )
    if monitoring_application or serving_to_monitor:
        if not mlrun.mlconf.is_ce_mode():
            model_monitoring_access_key = process_model_monitoring_secret(
                db_session,
                fn.metadata.project,
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
            )
        else:
            model_monitoring_access_key = None

        monitoring_deployment = (
            server.api.crud.model_monitoring.deployment.MonitoringDeployment(
                project=fn.metadata.project,
                auth_info=auth_info,
                db_session=db_session,
                model_monitoring_access_key=model_monitoring_access_key,
            )
        )
        try:
            monitoring_deployment.check_if_credentials_are_set()
        except mlrun.errors.MLRunBadRequestError as exc:
            if monitoring_application:
                err_txt = f"Can not deploy model monitoring application due to: {exc}"
            else:
                err_txt = (
                    f"Can not deploy serving function with track models due to: {exc}"
                )
            server.api.api.utils.log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason=err_txt,
            )
        if monitoring_application:
            fn = monitoring_deployment.apply_and_create_stream_trigger(
                function=fn, function_name=fn.metadata.name
            )

        if serving_to_monitor:
            if not client_version:
                server.api.api.utils.log_and_raise(
                    HTTPStatus.BAD_REQUEST.value,
                    reason=f"On deployment of serving-functions that are based on mlrun image "
                    f"('mlrun/') and set-tracking is enabled, "
                    f"client version must be specified and  >= {MINIMUM_CLIENT_VERSION_FOR_MM}",
                )
            elif fn.spec.image.startswith("mlrun/") and (
                semver.Version.parse(client_version)
                < semver.Version.parse(MINIMUM_CLIENT_VERSION_FOR_MM)
                and "unstable" not in client_version
            ):
                server.api.api.utils.log_and_raise(
                    HTTPStatus.BAD_REQUEST.value,
                    reason=f"On deployment of serving-functions that are based on mlrun image "
                    f"('mlrun/') and set-tracking is enabled, "
                    f"client version must be >= {MINIMUM_CLIENT_VERSION_FOR_MM}",
                )

    server.api.crud.runtimes.nuclio.function.deploy_nuclio_function(
        fn,
        auth_info=auth_info,
        client_version=client_version,
        client_python_version=client_python_version,
        builder_env=builder_env,
    )
    return fn


def _handle_nuclio_deploy_status(
    db_session,
    auth_info,
    fn,
    name: str,
    project: str,
    tag: str,
    last_log_timestamp: int,
    verbose: bool,
    api_gateway_urls: list[str],
):
    (
        state,
        _,
        nuclio_name,
        last_log_timestamp,
        text,
        status,
    ) = server.api.crud.runtimes.nuclio.function.get_nuclio_deploy_status(
        name,
        project,
        tag,
        # Workaround since when passing 0.0 to nuclio current timestamp is used and no logs are returned
        last_log_timestamp=last_log_timestamp or 1.0,
        verbose=verbose,
        auth_info=auth_info,
    )
    if state in ["ready", "scaledToZero"]:
        logger.info("Nuclio function deployed successfully", name=name)
    if state in ["error", "unhealthy"]:
        logger.error(f"Nuclio deploy error, {text}", name=name)

    internal_invocation_urls = (
        status.get("internalInvocationUrls")
        if status.get("internalInvocationUrls")
        else []
    )
    external_invocation_urls = (
        status.get("externalInvocationUrls")
        if status.get("externalInvocationUrls")
        else []
    )

    # add api gateway's URLs
    if api_gateway_urls:
        external_invocation_urls += api_gateway_urls

    # on earlier versions of mlrun, address used to represent the nodePort external invocation url
    # now that functions can be not exposed (using service_type clusterIP) this no longer relevant
    # and hence, for BC it would be filled with the external invocation url first item
    # or completely empty.
    address = external_invocation_urls[0] if external_invocation_urls else ""

    # the built and pushed image name used to run the nuclio function container
    container_image = status.get("containerImage", "")

    # we don't want to store the function on all requests to get the deploy status, therefore we verify
    # that changes were actually made and if that's the case then we store the function
    if _is_nuclio_deploy_status_changed(
        previous_status=fn.get("status", {}),
        new_status=status,
        new_state=state,
        new_nuclio_name=nuclio_name,
    ):
        mlrun.utils.update_in(fn, "status.nuclio_name", nuclio_name)
        mlrun.utils.update_in(
            fn, "status.internal_invocation_urls", internal_invocation_urls
        )
        mlrun.utils.update_in(
            fn, "status.external_invocation_urls", external_invocation_urls
        )
        mlrun.utils.update_in(fn, "status.state", state)
        mlrun.utils.update_in(fn, "status.address", address)
        mlrun.utils.update_in(fn, "status.container_image", container_image)

        versioned = False
        if state == "ready":
            # Versioned means the version will be saved in the DB forever, we don't want to spam
            # the DB with intermediate or unusable versions, only successfully deployed versions
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
        content=text,
        media_type="text/plain",
        headers={
            "x-mlrun-function-status": state,
            "x-mlrun-last-timestamp": str(last_log_timestamp),
            "x-mlrun-address": address,
            "x-mlrun-internal-invocation-urls": ",".join(internal_invocation_urls),
            "x-mlrun-external-invocation-urls": ",".join(external_invocation_urls),
            "x-mlrun-container-image": container_image,
            "x-mlrun-name": nuclio_name,
        },
    )


async def _get_api_gateways_urls_for_function(
    auth_info, project, name, tag
) -> list[str]:
    function_uri = generate_object_uri(project, name, tag)
    async with server.api.utils.clients.async_nuclio.Client(auth_info) as client:
        api_gateways = await client.list_api_gateways(project)
        # if there are any API gateways, filter the ones associated with the function
        # extract the hosts from the API gateway specifications and return them as a list
        # TODO: optimise the way we request api gateways by filtering on Nuclio side
        return (
            [
                api_gateway.spec.host
                for api_gateway in api_gateways.values()
                if function_uri in api_gateway.get_function_names()
            ]
            if api_gateways
            else []
        )


def _is_nuclio_deploy_status_changed(
    previous_status: dict, new_status: dict, new_state: str, new_nuclio_name: str = None
) -> bool:
    # get relevant fields from the new status
    new_container_image = new_status.get("containerImage", "")
    new_internal_invocation_urls = new_status.get("internalInvocationUrls", [])
    new_external_invocation_urls = new_status.get("externalInvocationUrls", [])
    address = new_external_invocation_urls[0] if new_external_invocation_urls else ""

    # Determine if any of the relevant fields have changed
    has_changed = (
        previous_status.get("nuclio_name", "") != new_nuclio_name
        or previous_status.get("state") != new_state
        or previous_status.get("container_image", "") != new_container_image
        or previous_status.get("internal_invocation_urls", [])
        != new_internal_invocation_urls
        or previous_status.get("external_invocation_urls", [])
        != new_external_invocation_urls
        or previous_status.get("address", "") != address
    )
    return has_changed


async def _delete_functions_external_invocation_url(
    project: str, url: str, function_names: list[str]
) -> None:
    tasks = [
        asyncio.create_task(
            run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                server.api.crud.Functions().delete_function_external_invocation_url,
                function_uri=function,
                project=project,
                invocation_url=url,
            )
        )
        for function in function_names
    ]
    await asyncio.gather(*tasks)


async def _add_functions_external_invocation_url(
    project: str, url: str, function_names: list[str]
) -> None:
    tasks = [
        asyncio.create_task(
            run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                server.api.crud.Functions().add_function_external_invocation_url,
                function_uri=function,
                project=project,
                invocation_url=url,
            )
        )
        for function in function_names
    ]
    await asyncio.gather(*tasks)
