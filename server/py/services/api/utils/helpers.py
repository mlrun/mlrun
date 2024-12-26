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
    project: mlrun.common.schemas.ProjectOut,
    workflow_spec: mlrun.common.schemas.WorkflowSpec,
    client_version: typing.Optional[str],
) -> str:
    if override_image := workflow_spec.image or project.spec.default_image:
        return override_image

    # set mlrun/mlrun-kfp if engine has KFP in it, else default to mlrun/mlrun
    must_use_mlrun_image = False
    if client_version and "unstable" not in client_version:
        client_version = semver.Version.parse(client_version)
        if client_version <= semver.Version.parse("1.7.9999"):
            print("must_use_mlrun_image", client_version)
            must_use_mlrun_image = True

    # "kfp" or "remote:kfp"
    kfp_engine = (
        mlrun.common.schemas.workflow.EngineType.KFP.lower()
        in (workflow_spec.engine or "").lower()
    )
    return (
        mlrun.mlconf.kfp_image
        if kfp_engine and not must_use_mlrun_image
        else mlrun.mlconf.default_base_image
    )
