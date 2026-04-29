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

import os
import traceback
from http import HTTPStatus

import mlrun.common.schemas
import mlrun.datastore
import mlrun.errors
import mlrun.utils
from mlrun.errors import err_to_str
from mlrun.run import new_function
from mlrun.runtimes import RuntimeKinds
from mlrun.utils import logger

import framework.api.utils
import services.api.launcher
from services.api.api.endpoints.nuclio import _deploy_nuclio_runtime
from services.api.utils.builder import build_runtime


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
        framework.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {err_to_str(err)}",
        )
    try:
        # connect to run db
        run_db = framework.api.utils.get_run_db_instance(db_session)
        fn.set_db_connection(run_db)

        # TODO:  nuclio deploy moved to new endpoint, this flow is about to be deprecated
        is_nuclio_deploy = fn.kind in RuntimeKinds.pure_nuclio_deployed_runtimes()

        # Enrich runtime with project defaults
        launcher = services.api.launcher.ServerSideLauncher(auth_info=auth_info)
        # When runtime is nuclio, building means we deploy the function and not just build its image,
        # so we need full enrichment
        launcher.enrich_runtime(
            runtime=fn, full=is_nuclio_deploy, client_version=client_version
        )

        # only validate
        framework.api.utils.apply_enrichment_and_validation_on_function(
            function=fn,
            auth_info=auth_info,
            ensure_auth=False,
            perform_auto_mount=False,
            mask_sensitive_data=False,
            ensure_security_context=False,
        )

        if is_nuclio_deploy:
            fn: mlrun.runtimes.RemoteRuntime
            # before saving function to DB, we need to mask some nuclio-specific fields
            # which later in Nuclio will be masked and saved to secrets
            raw_config = fn.mask_sensitive_data_in_config()

            # save without sensitive data
            fn.save(versioned=False)

            # after saving function to DB, we need to restore the original config
            # so that the sensitive data won't be stored
            fn.spec.config = raw_config

            fn.pre_deploy_validation()
            fn = _deploy_nuclio_runtime(
                auth_info,
                builder_env,
                client_python_version,
                client_version,
                db_session,
                fn,
            )
            # after deploying the function, we need to re-mask the sensitive data again and save to the db
            fn.mask_sensitive_data_in_config()

            # deploy only start the process, the get status API is used to check readiness
            ready = False
        else:
            fn.save(versioned=False)
            log_file = framework.api.utils.log_path(
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
        logger.info("Resolved function", fn=fn.to_dict())
    except Exception as err:
        logger.error(traceback.format_exc())
        framework.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"Runtime error: {err_to_str(err)}",
        )
    return fn, ready


def enrich_function_from_code_artifact(
    function: "mlrun.runtimes.base.BaseRuntime",
    project: str,
):
    """Resolve store:// code artifact and enrich the function build spec from it.

    When ``function.spec.build.source`` is a store:// URI:

    1. Validates that the artifact is a CodeArtifact (kind == "code").
    2. Merges artifact ``spec.requirements`` into ``function.spec.build.requirements``
       (user requirements win on conflict).
    3. Defaults ``function.spec.build.load_source_on_run`` to True when unset, so
       the source is fetched at runtime rather than baked into a build image.
       An explicit user value (True/False) is preserved.

    :param function: The function object to enrich
    :param project:  Project name for artifact resolution
    """
    # Falls back to status.application_source for Nuclio (Application runtime)
    # redeploys where spec.build.source is cleared mid-deploy. No-op for jobs.
    source = function.spec.build.source or getattr(
        function.status, "application_source", None
    )
    if not source or not mlrun.utils.is_store_uri(source):
        return

    try:
        artifact = mlrun.datastore.get_store_resource(source, project=project)
    except mlrun.errors.MLRunBaseError:
        # Preserve typed MLRun errors so HTTP status mapping (e.g. 404 for
        # MLRunNotFoundError) is not collapsed into a generic 400.
        raise
    except Exception as exc:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Cannot resolve code artifact {source}: {err_to_str(exc)}"
        ) from exc

    if artifact.kind != "code":
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Source {source} resolves to a {artifact.kind!r} artifact; "
            "expected a code artifact (kind='code')."
        )

    artifact_requirements = getattr(artifact.spec, "requirements", None)
    if artifact_requirements:
        function.spec.build.requirements = mlrun.utils.merge_requirements(
            reqs_priority=function.spec.build.requirements or [],
            reqs_secondary=artifact_requirements,
        )

    # Default load_source_on_run for store:// jobs so the pod fetches the
    # artifact at startup. An explicit user value (True/False) is preserved.
    if function.spec.build.load_source_on_run is None:
        logger.debug(
            "Defaulting load_source_on_run=True for store:// source",
            source=source,
        )
        function.spec.build.load_source_on_run = True
