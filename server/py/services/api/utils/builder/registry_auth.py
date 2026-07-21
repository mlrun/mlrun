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
"""Cloud-registry credential exchange for :class:`~services.api.utils.builder.buildah.BuildahBackend`.

Buildah's stock image has no built-in cloud-credential support the way Kaniko's Go binary does, so
each provider mints its own push credential from the pod's existing workload identity:

* **ECR** — an init container on the MLRun image (already carries boto3) calls
  ``ecr.get_authorization_token()`` and writes the result to a shared authfile.
* **ACR** — an init container on the MLRun image (already carries azure-identity) exchanges an AAD
  token for an ACR refresh token via ACR's ``/oauth2/exchange`` endpoint, and writes the result to
  the same shared authfile.
* **GAR / GCR** — no init container: GCP metadata-server tokens default to a 1-hour TTL, which can
  be shorter than a long build, so the token is minted just-in-time inside the Buildah main
  container, immediately before ``buildah push`` (see :mod:`services.api.utils.builder.buildah`).

None of the generated scripts ever log the minted token.
"""

from urllib.parse import urlparse

import mlrun.utils
import mlrun.utils.helpers
from mlrun.config import config

_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME = "registry-credential-exchange"

# the mlrun/mlrun image ships boto3 / azure-identity via the "complete" extra - the same image
# kaniko's source-fetch init container derives from by default (see
# kaniko_source_fetch_init_container_image), so ECR/ACR credential exchange reuses that convention.
_DEFAULT_CREDENTIAL_EXCHANGE_IMAGE = "mlrun/mlrun"

_ACR_TOKEN_SCOPE = "https://containerregistry.azure.com/.default"
_ACR_ANONYMOUS_USERNAME = "00000000-0000-0000-0000-000000000000"

_GCP_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/token"
)


def classify_cloud_registry(target: str) -> str | None:
    """Return which cloud provider's registry ``target`` belongs to, or ``None``.

    :param target: A registry host, or a full image reference (only the host matters).
    :return: ``"ecr"``, ``"acr"``, ``"gar"``, or ``None`` for anything else (Docker Hub, a
        private/self-signed registry, ...).
    """
    if not target:
        return None
    if mlrun.utils.helpers.is_ecr_url(target):
        return "ecr"
    if _is_acr_registry(target):
        return "acr"
    if _is_gar_registry(target):
        return "gar"
    return None


def append_ecr_credential_exchange_init_container(
    pod, registry: str, dest: str, authfile_path: str
) -> None:
    """Append the init container that mints ECR push credentials for ``pod``.

    Runs boto3 (bundled in the MLRun image) with the pod's own AWS credentials - IRSA or instance
    role, resolved via the build pod's service account, see
    :func:`~services.api.utils.builder.base.resolve_build_pod_spec_attributes` - to create the
    target ECR repository (idempotent) and mint a short-lived authorization token, writing it to
    ``authfile_path`` for the main Buildah container to push with.

    :param pod: The Buildah build pod being constructed.
    :param registry: The ECR registry host.
    :param dest: The fully resolved destination image reference (for the repository name).
    :param authfile_path: Where to write the docker-config-shaped authfile.
    """
    pod.append_init_container(
        _credential_exchange_image(),
        command=["python3", "-c"],
        args=[_ecr_credential_exchange_script(registry, dest, authfile_path)],
        name=_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
    )


def append_acr_credential_exchange_init_container(
    pod, registry: str, authfile_path: str
) -> None:
    """Append the init container that mints ACR push credentials for ``pod``.

    Exchanges the pod's Azure workload identity (federated JWT, injected by the
    ``azure.workload.identity/use`` label - see
    :func:`~services.api.utils.builder.base.resolve_builder_pod_labels`) for an AAD access token via
    azure-identity (bundled in the MLRun image), then exchanges that AAD token for an ACR refresh
    token via ACR's ``/oauth2/exchange`` endpoint (no SDK covers this ACR-specific endpoint), writing
    the result to ``authfile_path``.

    :param pod: The Buildah build pod being constructed.
    :param registry: The ACR registry host.
    :param authfile_path: Where to write the docker-config-shaped authfile.
    """
    pod.append_init_container(
        _credential_exchange_image(),
        command=["python3", "-c"],
        args=[_acr_credential_exchange_script(registry, authfile_path)],
        name=_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
    )


def gar_credential_exchange_script(registry: str, authfile_path: str) -> str:
    """Return the Python source that mints a GAR/GCR push token just-in-time.

    Unlike ECR/ACR, this does not run in an init container: GCP metadata-server tokens default to a
    1-hour TTL, which can be shorter than a long build, so it is minted directly in the Buildah main
    container immediately before ``buildah push`` (see
    :func:`~services.api.utils.builder.buildah._build_script`). Relies on GKE Workload Identity
    transparently intercepting the metadata-server call for the build pod's service account - no
    federated-token plumbing is needed on mlrun's side. Uses only the standard library, since the
    stock Buildah image has no cloud SDKs installed.

    :param registry: The GAR/GCR registry host.
    :param authfile_path: Where to write the docker-config-shaped authfile.
    :return: The Python source to run (e.g. via ``python3 -c``).
    """
    lines = [
        "import base64",
        "import json",
        "import urllib.request",
        "",
        f"req = urllib.request.Request({_GCP_METADATA_TOKEN_URL!r}, headers="
        "{'Metadata-Flavor': 'Google'})",
        "with urllib.request.urlopen(req) as resp:",
        "    token = json.load(resp)['access_token']",
        "auth = base64.b64encode(b'oauth2accesstoken:' + token.encode()).decode()",
        f"registry = {registry!r}",
        f"with open({authfile_path!r}, 'w') as fh:",
        "    json.dump({'auths': {registry: {'auth': auth}}}, fh)",
    ]
    return "\n".join(lines)


def _credential_exchange_image() -> str:
    image = config.httpdb.builder.buildah_registry_credential_exchange_image
    if not image:
        image = mlrun.utils.enrich_image_url(_DEFAULT_CREDENTIAL_EXCHANGE_IMAGE)
    return image


def _ecr_repo_name(dest: str) -> str:
    end = dest.find(":")
    if end == -1:
        end = len(dest)
    return dest[dest.find("/") + 1 : end]


def _ecr_credential_exchange_script(
    registry: str, dest: str, authfile_path: str
) -> str:
    region = registry.split(".")[3]
    repo = _ecr_repo_name(dest)
    lines = [
        "import boto3",
        "import json",
        "",
        f"client = boto3.client('ecr', region_name={region!r})",
        f"repo = {repo!r}",
        "for repo_name in (repo, repo + '/cache'):",
        "    try:",
        "        client.create_repository(repositoryName=repo_name)",
        "    except client.exceptions.RepositoryAlreadyExistsException:",
        "        pass",
        "",
        "authorization_data = client.get_authorization_token()['authorizationData'][0]",
        "token = authorization_data['authorizationToken']",
        f"registry = {registry!r}",
        f"with open({authfile_path!r}, 'w') as fh:",
        "    json.dump({'auths': {registry: {'auth': token}}}, fh)",
    ]
    return "\n".join(lines)


def _acr_credential_exchange_script(registry: str, authfile_path: str) -> str:
    lines = [
        "import base64",
        "import json",
        "import os",
        "",
        "import requests",
        "from azure.identity import DefaultAzureCredential",
        "",
        f"registry = {registry!r}",
        f"aad_token = DefaultAzureCredential().get_token({_ACR_TOKEN_SCOPE!r}).token",
        "response = requests.post(",
        "    'https://' + registry + '/oauth2/exchange',",
        "    data={",
        "        'grant_type': 'access_token',",
        "        'service': registry,",
        "        'tenant': os.environ['AZURE_TENANT_ID'],",
        "        'access_token': aad_token,",
        "    },",
        ")",
        "response.raise_for_status()",
        "refresh_token = response.json()['refresh_token']",
        "auth = base64.b64encode(",
        f"    ({_ACR_ANONYMOUS_USERNAME!r} + ':' + refresh_token).encode()",
        ").decode()",
        f"with open({authfile_path!r}, 'w') as fh:",
        "    json.dump({'auths': {registry: {'auth': auth}}}, fh)",
    ]
    return "\n".join(lines)


def _hostname(target: str) -> str | None:
    return urlparse(f"https://{target}").hostname


def _is_acr_registry(target: str) -> bool:
    hostname = _hostname(target)
    return bool(hostname) and hostname.endswith(".azurecr.io")


def _is_gar_registry(target: str) -> bool:
    hostname = _hostname(target)
    if not hostname:
        return False
    return (
        hostname.endswith("-docker.pkg.dev")
        or hostname == "gcr.io"
        or hostname.endswith(".gcr.io")
    )
