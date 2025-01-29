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

import mlrun
import mlrun.common.schemas
from mlrun.common.schemas import ProjectOut, WorkflowSpec
from mlrun.utils import logger


def resolve_client_default_kfp_image(
    project: typing.Optional[ProjectOut] = None,
    workflow_spec: typing.Optional[WorkflowSpec] = None,
) -> str:
    if workflow_spec and workflow_spec.image:
        image = workflow_spec.image
    elif project and project.spec.default_image:
        image = project.spec.default_image
    else:
        image = mlrun.mlconf.kfp_image
    logger.debug(
        "Resolved KFP image for workflow",
        project=project,
        workflow_spec_image=workflow_spec.image,
        project_spec_default_image=project.spec.default_image,
        resolved_image=image,
    )
    return image
