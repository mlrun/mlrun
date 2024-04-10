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

import os
import traceback
from http import HTTPStatus

import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import server.api.api.utils
import server.api.crud.model_monitoring.deployment
import server.api.crud.runtimes.nuclio.function
import server.api.launcher
import server.api.utils.singletons.project_member
from mlrun.common.model_monitoring.helpers import parse_model_endpoint_store_prefix
from mlrun.config import config
from mlrun.errors import MLRunRuntimeError, err_to_str
from mlrun.run import new_function
from mlrun.runtimes import RuntimeKinds
from mlrun.utils import logger
from server.api.crud.secrets import Secrets, SecretsClientType
from server.api.utils.builder import build_runtime


def process_model_monitoring_secret(db_session, project_name: str, secret_key: str):
    # The expected result of this method is an access-key placed in an internal project-secret.
    # If the user provided an access-key as the "regular" secret_key, then we delete this secret and move contents
    # to the internal secret instead. Else, if the internal secret already contained a value, keep it. Last option
    # (which is the recommended option for users) is to retrieve a new access-key from the project owner and use it.
    logger.info(
        "Getting project secret", project_name=project_name, namespace=config.namespace
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
            project_owner = server.api.utils.singletons.project_member.get_project_member().get_project_owner(
                db_session, project_name
            )

            secret_value = project_owner.access_key
            if not secret_value:
                raise MLRunRuntimeError(
                    f"No model monitoring access key. Failed to generate one for owner of project {project_name}",
                )

            logger.info(
                "Filling model monitoring access-key from project owner",
                project_name=project_name,
                project_owner=project_owner.username,
            )

    secrets = mlrun.common.schemas.SecretsData(
        provider=provider, secrets={internal_key_name: secret_value}
    )
    Secrets().store_project_secrets(project_name, secrets, allow_internal_secrets=True)
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
            endpoint=config.v3io_api,
        )

        v3io_client = v3io.dataplane.Client(
            endpoint=config.v3io_api, access_key=access_key
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


def build_function(
    db_session,
    auth_info: mlrun.common.schemas.AuthInfo,
    function,
    with_mlrun=True,
    skip_deployed=False,
    mlrun_version_specifier=None,
    builder_env=None,
    client_version=None,
    client_python_version=None,
    force_build=False,
):
    fn = None
    ready = None
    try:
        fn = new_function(runtime=function)
    except Exception as err:
        logger.error(traceback.format_exc())
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {err_to_str(err)}",
        )
    try:
        # connect to run db
        run_db = server.api.api.utils.get_run_db_instance(db_session)
        fn.set_db_connection(run_db)

        is_nuclio_runtime = fn.kind in RuntimeKinds.nuclio_runtimes()

        # Enrich runtime with project defaults
        launcher = server.api.launcher.ServerSideLauncher(auth_info=auth_info)
        # When runtime is nuclio, building means we deploy the function and not just build its image
        # so we need full enrichment
        launcher.enrich_runtime(runtime=fn, full=is_nuclio_runtime)

        fn.save(versioned=False)
        if is_nuclio_runtime:
            fn: mlrun.runtimes.RemoteRuntime
            fn.pre_deploy_validation()
            fn = _deploy_nuclio_runtime(
                auth_info,
                builder_env,
                client_python_version,
                client_version,
                db_session,
                fn,
            )
            # deploy only start the process, the get status API is used to check readiness
            ready = False
        else:
            log_file = server.api.api.utils.log_path(
                fn.metadata.project,
                f"build_{fn.metadata.name}__{fn.metadata.tag or 'latest'}",
            )
            if log_file.exists() and not (skip_deployed and fn.is_deployed()):
                # delete old build log file if exist and build is not skipped
                os.remove(str(log_file))

            ready = build_runtime(
                auth_info,
                fn,
                with_mlrun,
                mlrun_version_specifier,
                skip_deployed,
                builder_env=builder_env,
                client_version=client_version,
                client_python_version=client_python_version,
                force_build=force_build,
            )
        fn.save(versioned=True)
        logger.info("Resolved function", fn=fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        server.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {err_to_str(err)}",
        )
    return fn, ready


def _deploy_nuclio_runtime(
    auth_info, builder_env, client_python_version, client_version, db_session, fn
):
    monitoring_application = (
        fn.metadata.labels.get(mm_constants.ModelMonitoringAppLabel.KEY)
        == mm_constants.ModelMonitoringAppLabel.VAL
    )
    serving_to_monitor = fn.kind == RuntimeKinds.serving and fn.spec.track_models
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
        if monitoring_application:
            fn = monitoring_deployment._apply_and_create_stream_trigger(
                function=fn,
                function_name=fn.metadata.name,
            )

        if serving_to_monitor:
            if not mlrun.mlconf.is_ce_mode():
                if not monitoring_deployment.is_monitoring_stream_has_the_new_stream_trigger():
                    monitoring_deployment.deploy_model_monitoring_stream_processing(
                        overwrite=True
                    )

    server.api.crud.runtimes.nuclio.function.deploy_nuclio_function(
        fn,
        auth_info=auth_info,
        client_version=client_version,
        client_python_version=client_python_version,
        builder_env=builder_env,
    )
    return fn
