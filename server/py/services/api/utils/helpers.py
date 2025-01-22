# Copyright 2024 Iguazio
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

import typing

import semver

import mlrun
import mlrun.common.schemas
from mlrun.common.schemas import ProjectOut, WorkflowSpec


def resolve_client_default_kfp_image(
    project: typing.Optional[ProjectOut] = None,
    workflow_spec: typing.Optional[WorkflowSpec] = None,
    client_version: typing.Optional[str] = None,
) -> str:
    if workflow_spec and workflow_spec.image:
        return workflow_spec.image
    elif project and project.spec.default_image:
        return project.spec.default_image

    # Determines the default KFP image based on the engine type and client version:
    # - For engine=KFP or REMOTE_KFP:
    #     - By default, use mlrun.mlconf.kfp_image (e.g. "mlrun/mlrun-kfp").
    #     - But for clients <1.8.0 (unless "unstable"), revert to mlrun.mlconf.default_base_image
    #     (e.g. "mlrun/mlrun") for backward compatibility.
    # - For engine=LOCAL or REMOTE: use mlrun.mlconf.default_base_image.
    #
    # This function is used in:
    # 1) get_client_spec(...) => The engine is KFP, older clients still get "mlrun/mlrun",
    #       and newer ones get "mlrun/mlrun-kfp".
    # 2) submit_workflow(...) => Chooses the correct runner image for workflows.
    #       (server side also uses "mlrun/mlrun-kfp").

    pre_kfp_image_mlrun_version = False
    if client_version and "unstable" not in client_version.lower():
        try:
            parsed_version = semver.VersionInfo.parse(client_version)
            if parsed_version < semver.VersionInfo(1, 8, 0):
                pre_kfp_image_mlrun_version = True
        except ValueError:
            pass

    engine = (
        workflow_spec.engine.lower()
        if (workflow_spec and workflow_spec.engine)
        else mlrun.common.schemas.workflow.EngineType.LOCAL
    )

    if (
        pre_kfp_image_mlrun_version
        and engine == mlrun.common.schemas.workflow.EngineType.KFP
    ):
        return mlrun.mlconf.default_base_image

    if engine in (
        mlrun.common.schemas.workflow.EngineType.REMOTE,
        mlrun.common.schemas.workflow.EngineType.REMOTE_KFP,
        mlrun.common.schemas.workflow.EngineType.KFP,
    ):
        return mlrun.mlconf.kfp_image

    return mlrun.mlconf.default_base_image
