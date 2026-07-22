# Copyright 2026 Iguazio
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
"""ECR/ACR push-credential minting, run inside a Buildah build's init container (ML-12886).

Invoked via ``python -m mlrun mint-registry-credentials`` by
:mod:`services.api.utils.builder.registry_auth` (server-side) - the init container runs on the
mlrun image, so this lives in the SDK rather than the server package. GAR/GCR credential exchange is
not here: it runs inline in the Buildah main container instead, which has no mlrun installed - see
the server-side module for that script.
"""

import base64
import json

_ACR_TOKEN_SCOPE = "https://containerregistry.azure.com/.default"
_ACR_ANONYMOUS_USERNAME = "00000000-0000-0000-0000-000000000000"


def mint_ecr_authfile(registry: str, dest: str, authfile_path: str) -> None:
    """Create the target ECR repository (idempotent) and write a push authfile.

    Credentials come from boto3's default chain (IRSA or instance role, resolved via the build
    pod's own service account/environment) - nothing else needs to be configured.

    :param registry: The ECR registry host.
    :param dest: The fully resolved destination image reference (for the repository name).
    :param authfile_path: Where to write the docker-config-shaped authfile.
    """
    import boto3

    region = registry.split(".")[3]
    repo = _ecr_repo_name(dest)
    client = boto3.client("ecr", region_name=region)
    for repo_name in (repo, f"{repo}/cache"):
        try:
            client.create_repository(repositoryName=repo_name)
        except client.exceptions.RepositoryAlreadyExistsException:
            pass

    authorization_data = client.get_authorization_token()["authorizationData"][0]
    token = authorization_data["authorizationToken"]
    with open(authfile_path, "w") as fh:
        json.dump({"auths": {registry: {"auth": token}}}, fh)


def mint_acr_authfile(registry: str, authfile_path: str) -> None:
    """Exchange the pod's Azure workload identity for an ACR push credential.

    Exchanges the pod's Azure workload identity (federated JWT, injected by the
    ``azure.workload.identity/use`` label - see
    :func:`~services.api.utils.builder.base.resolve_builder_pod_labels`) for an AAD access token via
    azure-identity, then exchanges that AAD token for an ACR refresh token via ACR's
    ``/oauth2/exchange`` endpoint (no SDK covers this ACR-specific endpoint).

    :param registry: The ACR registry host.
    :param authfile_path: Where to write the docker-config-shaped authfile.
    """
    import os

    import requests
    from azure.identity import DefaultAzureCredential

    aad_token = DefaultAzureCredential().get_token(_ACR_TOKEN_SCOPE).token
    response = requests.post(
        f"https://{registry}/oauth2/exchange",
        data={
            "grant_type": "access_token",
            "service": registry,
            "tenant": os.environ["AZURE_TENANT_ID"],
            "access_token": aad_token,
        },
    )
    response.raise_for_status()
    refresh_token = response.json()["refresh_token"]
    auth = base64.b64encode(
        f"{_ACR_ANONYMOUS_USERNAME}:{refresh_token}".encode()
    ).decode()
    with open(authfile_path, "w") as fh:
        json.dump({"auths": {registry: {"auth": auth}}}, fh)


def _ecr_repo_name(dest: str) -> str:
    end = dest.find(":")
    if end == -1:
        end = len(dest)
    return dest[dest.find("/") + 1 : end]
