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

import typing

import semver

import mlrun
import mlrun.auth.utils
import mlrun.common.constants
import mlrun.common.schemas
from mlrun.common.schemas import ProjectOut, WorkflowSpec
from mlrun.utils import logger

import framework.utils.singletons.k8s


def resolve_client_default_kfp_image(
    project: ProjectOut,
    workflow_spec: typing.Optional[WorkflowSpec] = None,
    client_version: typing.Optional[str] = None,
) -> str:
    if workflow_spec and workflow_spec.image:
        image = workflow_spec.image
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
        workflow_spec_image=getattr(workflow_spec, "image", None),
        project_spec_default_image=project.spec.default_image,
        resolved_image=image,
    )
    return image


def resolve_auth_token_secret_name(
    provided_token_name: typing.Optional[str], username: typing.Optional[str]
) -> typing.Optional[str]:
    """
    Resolve the name of the secret that holds the user's auth token. Performs enrichment and validation of the
    token name.
    :param provided_token_name: The name of the token provided by the user, if any.
    :param username: The username for which the token is being resolved.

    :return: The name of the secret that holds the user's auth token.
    """
    if mlrun.mlconf.is_iguazio_v4_mode():
        # In ML-11600, we will implement a proper resolution logic that checks all secret tokens
        # of the user and finds a valid one if no token name is provided
        # If token name not provided, use default
        token_name = (
            provided_token_name
            or mlrun.common.constants.MLRUN_RUNTIME_AUTH_DEFAULT_TOKEN_NAME
        )
        secret = framework.utils.singletons.k8s.get_k8s_helper()._get_user_token_secret(
            username=username, token_name=token_name
        )
        return secret.metadata.name if secret else None
