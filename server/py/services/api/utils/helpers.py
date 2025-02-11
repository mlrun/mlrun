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
    project: ProjectOut,
    workflow_spec: WorkflowSpec,
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
                # client is older than (<) 1.8, must use mlrun image for kfp
                if semver.Version.parse(client_version) < semver.Version.parse(
                    "1.8.0-rc0"
                ):
                    must_use_mlrun_image = True
            except ValueError:
                # client version is not semver, pass
                pass

        if must_use_mlrun_image:
            image = mlrun.mlconf.default_base_image
            if ":" not in image:
                # enrich the image with the client version to ensure that
                # client < 1.8 will use the correct mlrun image and version.
                # https://iguazio.atlassian.net/browse/ML-9292
                enriched_image = mlrun.utils.enrich_image_url(
                    image, client_version=client_version
                )
                logger.debug(
                    "Ensuring KFP image has fixed client version",
                    enriched_image=enriched_image,
                    image=image,
                )
                image = enriched_image
        else:
            image = mlrun.mlconf.kfp_image

    logger.debug(
        "Resolved KFP image for workflow",
        project_name=project.metadata.name,
        client_version=client_version,
        workflow_spec_image=workflow_spec.image,
        project_spec_default_image=project.spec.default_image,
        resolved_image=image,
    )
    return image
