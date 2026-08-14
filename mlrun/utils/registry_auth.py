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
"""ECR/ACR credential minting, run inside a Buildah build's init container (ML-12886, ML-12961).

Invoked via ``python -m mlrun mint-registry-credentials`` by
:mod:`services.api.utils.builder.registry_auth` (server-side) - the init container runs on the
mlrun image, so this lives in the SDK rather than the server package. GAR/GCR credential exchange is
not here: it runs inline in the Buildah main container instead, which has no mlrun installed - see
the server-side module for that script.
"""

import base64
import json
import os

import requests

import mlrun
import mlrun.common.types


class CloudRegistryProvider(mlrun.common.types.StrEnum):
    ECR = "ecr"
    ACR = "acr"
    GAR = "gar"


# the ACR-specific resource (https://containerregistry.azure.com) is Microsoft's documented scope
# for this flow, but its resource principal isn't provisioned in every tenant (AADSTS500011 if not -
# confirmed on a live cluster, ML-12886). ARM is provisioned in every tenant unconditionally, and
# ACR's /oauth2/exchange accepts ARM-scoped tokens the same way, so it's the more portable choice.
_ACR_TOKEN_SCOPE = "https://management.azure.com/.default"
_ACR_ANONYMOUS_USERNAME = "00000000-0000-0000-0000-000000000000"


def mint_ecr_authfile(
    registry: str, authfile_path: str, dest: str | None = None
) -> None:
    """Mint an ECR authorization token and merge it into the authfile.

    Unlike ACR/GAR, ECR does not auto-create a repository on first push - it must exist first, or
    the push fails outright. When ``dest`` is given (a push), the repository is created (idempotent)
    up front; when it's omitted (a base-image pull), repository creation is skipped - the pod may
    not even have create permissions on a registry it's only pulling from.

    Credentials come from boto3's default chain (IRSA or instance role, resolved via the build
    pod's own service account/environment) - nothing else needs to be configured.

    :param registry: The ECR registry host.
    :param authfile_path: Where to merge the docker-config-shaped authfile entry.
    :param dest: The fully resolved destination image reference, to derive the repository name for
        creation. Omit for a pull-only (base-image) credential exchange.
    """
    import boto3

    region = registry.split(".")[3]
    client = boto3.client("ecr", region_name=region)
    if dest:
        repo = _ecr_repo_name(dest)
        try:
            client.create_repository(repositoryName=repo)
        except client.exceptions.RepositoryAlreadyExistsException:
            pass

    authorization_data = client.get_authorization_token()["authorizationData"][0]
    token = authorization_data["authorizationToken"]
    _merge_auth_entry(authfile_path, registry, token)


def mint_acr_authfile(registry: str, authfile_path: str) -> None:
    """Exchange the pod's Azure workload identity for an ACR credential.

    Exchanges the pod's Azure workload identity (federated JWT, injected by the
    ``azure.workload.identity/use`` label - see
    :func:`~services.api.utils.builder.base.resolve_builder_pod_labels`) for an AAD access token via
    azure-identity, then exchanges that AAD token for an ACR refresh token via ACR's
    ``/oauth2/exchange`` endpoint (no SDK covers this ACR-specific endpoint). Used for both push and
    base-image-pull credentials - the exchange itself is identical, only the registry host differs.

    :param registry: The ACR registry host.
    :param authfile_path: Where to merge the docker-config-shaped authfile entry.
    """
    # lazy: azure-identity is only needed for ACR, not ECR/GAR - avoid requiring it unless this path
    # actually runs, same as boto3 in mint_ecr_authfile.
    from azure.identity import DefaultAzureCredential

    tenant = os.environ.get("AZURE_TENANT_ID")
    if not tenant:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "AZURE_TENANT_ID is not set - Azure Workload Identity does not appear to be configured "
            "on this pod"
        )
    aad_token = DefaultAzureCredential().get_token(_ACR_TOKEN_SCOPE).token
    response = requests.post(
        f"https://{registry}/oauth2/exchange",
        data={
            "grant_type": "access_token",
            "service": registry,
            "tenant": tenant,
            "access_token": aad_token,
        },
        timeout=10,
    )
    response.raise_for_status()
    refresh_token = response.json()["refresh_token"]
    auth = base64.b64encode(
        f"{_ACR_ANONYMOUS_USERNAME}:{refresh_token}".encode()
    ).decode()
    _merge_auth_entry(authfile_path, registry, auth)


def _merge_auth_entry(authfile_path: str, registry: str, auth: str) -> None:
    # a push-side and a pull-side credential exchange can both target this file (different
    # registries, e.g. the push destination and a differently-hosted base image), and a copied-in
    # static secret (ML-12988) may already be there too - init containers run sequentially, never
    # concurrently, so read-modify-write here is safe without locking. Preserves any other
    # top-level docker-config keys already in the file (credHelpers, credsStore, ...) - only the
    # entry for this one registry is ever added or replaced.
    if os.path.exists(authfile_path):
        with open(authfile_path) as fh:
            doc = json.load(fh)
    else:
        doc = {}
    doc.setdefault("auths", {})[registry] = {"auth": auth}
    with open(authfile_path, "w") as fh:
        json.dump(doc, fh)


def _ecr_repo_name(dest: str) -> str:
    end = dest.find(":")
    if end == -1:
        end = len(dest)
    return dest[dest.find("/") + 1 : end]
