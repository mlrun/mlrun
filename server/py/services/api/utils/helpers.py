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
from mlrun.utils import logger


def resolve_client_default_kfp_image(
    project: typing.Optional[ProjectOut] = None,
    workflow_spec: typing.Optional[WorkflowSpec] = None,
    client_version: typing.Optional[str] = None,
) -> str:
    if workflow_spec and workflow_spec.image:
        image = workflow_spec.image
    elif project and project.spec.default_image:
        image = project.spec.default_image
    else:
        must_use_mlrun_image = False
        if client_version and "unstable" not in client_version:
            try:
                client_version = semver.Version.parse(client_version)
                # client is olden than (<) 1.8, must use mlrun image for kfp
                if client_version < semver.Version.parse("1.7.9999"):
                    must_use_mlrun_image = True
            except ValueError:
                # client version is not semver, pass
                pass

        if must_use_mlrun_image:
            image = mlrun.mlconf.default_base_image
        else:
            image = mlrun.mlconf.kfp_image

        logger.debug(
            "Resolved KFP image for workflow",
            project=project,
            client_version=client_version,
            workflow_spec_image=workflow_spec.image,
            project_spec_default_image=project.spec.default_image,
            resolved_image=image,
        )
    return image
