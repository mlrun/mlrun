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
import server.api.api.utils
import server.api.launcher
from mlrun.errors import err_to_str
from mlrun.run import new_function
from mlrun.runtimes import RuntimeKinds
from mlrun.utils import logger
from server.api.api.endpoints.nuclio import _deploy_nuclio_runtime
from server.api.utils.builder import build_runtime


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

        # TODO:  nuclio deploy moved to new endpoint, this flow is about to be deprecated
        is_nuclio_deploy = fn.kind in RuntimeKinds.pure_nuclio_deployed_runtimes()

        # Enrich runtime with project defaults
        launcher = server.api.launcher.ServerSideLauncher(auth_info=auth_info)
        # When runtime is nuclio, building means we deploy the function and not just build its image,
        # so we need full enrichment
        launcher.enrich_runtime(runtime=fn, full=is_nuclio_deploy)

        fn.save(versioned=False)
        if is_nuclio_deploy:
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
