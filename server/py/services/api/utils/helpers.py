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

import mlrun.common.schemas


def resolve_client_default_kfp_image(
    project: typing.Optional[mlrun.common.schemas.ProjectOut] = None,
    workflow_spec: typing.Optional[mlrun.common.schemas.WorkflowSpec] = None,
    client_version: typing.Optional[str] = None,
) -> str:
    # override image by either workflow or project
    if workflow_spec and workflow_spec.image:
        return workflow_spec.image
    elif project and project.spec.default_image:
        return project.spec.default_image

    # set mlrun/mlrun-kfp if engine has KFP in it, else default to mlrun/mlrun
    must_use_mlrun_image = False
    if client_version and "unstable" not in client_version:
        try:
            client_version = semver.Version.parse(client_version)
            if client_version <= semver.Version.parse("1.7.9999"):
                must_use_mlrun_image = True
        except ValueError:
            # client version is not semver, pass
            pass

    # client is olden than (<) 1.8, must use mlrun image for kfp
    if must_use_mlrun_image:
        return mlrun.mlconf.default_base_image

    # "kfp" or "remote:kfp"
    kfp_engine = (
        workflow_spec
        and mlrun.common.schemas.workflow.EngineType.KFP.lower()
        in (workflow_spec.engine or "").lower()
    )
    return mlrun.mlconf.kfp_image if kfp_engine else mlrun.mlconf.default_base_image
