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

* **ECR** and **ACR** — an init container on the MLRun image (already carries boto3 /
  azure-identity, and mlrun itself) runs ``python -m mlrun mint-registry-credentials`` (see
  :mod:`mlrun.utils.registry_auth` for the actual credential-exchange logic) and writes the result
  to a shared authfile - the same ``python -m mlrun <subcommand>`` convention Kaniko's source-fetch
  init container uses.
* **GAR / GCR** — no init container: GCP tokens are cached and reused by the metadata server across
  callers until fewer than 5 minutes remain before expiry, so *any* mint can hand back a token with
  as little as ~5 minutes of remaining life, regardless of its original TTL. Minting just-in-time,
  immediately before each Buildah step that needs it (see :mod:`services.api.utils.builder.buildah`),
  minimizes the gap between mint and use rather than trying to outlast a fixed TTL. This path can't
  use the same init-container convention as ECR/ACR: it runs inline in the Buildah main container,
  which has no mlrun (or any cloud SDK) installed - only the stock image's python3/urllib.

None of the generated scripts or CLI args ever log the minted token.
"""

from urllib.parse import urlparse

import mlrun.utils
import mlrun.utils.helpers
from mlrun.config import config
from mlrun.utils.registry_auth import CloudRegistryProvider

_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME = "registry-credential-exchange"

# the mlrun/mlrun image ships boto3 / azure-identity via the "complete" extra - the same image
# kaniko's source-fetch init container derives from by default (see
# kaniko_source_fetch_init_container_image), so ECR/ACR credential exchange reuses that convention.
_DEFAULT_CREDENTIAL_EXCHANGE_IMAGE = "mlrun/mlrun"

_GCP_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/token"
)


def classify_cloud_registry(target: str) -> CloudRegistryProvider | None:
    """Return which cloud provider's registry ``target`` belongs to, or ``None``.

    :param target: A registry host, or a full image reference (only the host matters).
    :return: The registry's :class:`CloudRegistryProvider`, or ``None`` for anything else (Docker
        Hub, a private/self-signed registry, ...).
    """
    if not target:
        return None
    if mlrun.utils.helpers.is_ecr_url(target):
        return CloudRegistryProvider.ECR
    if _is_acr_registry(target):
        return CloudRegistryProvider.ACR
    if _is_gar_registry(target):
        return CloudRegistryProvider.GAR
    return None


def append_ecr_credential_exchange_init_container(
    pod, registry: str, dest: str, authfile_path: str
) -> None:
    """Append the init container that mints ECR push credentials for ``pod``.

    Runs ``python -m mlrun mint-registry-credentials --provider ecr`` (see
    :func:`mlrun.utils.registry_auth.mint_ecr_authfile`), which uses boto3 with the pod's own AWS
    credentials - IRSA or instance role, resolved via the build pod's service account, see
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
        command=["python"],
        args=[
            "-m",
            "mlrun",
            "mint-registry-credentials",
            "--provider",
            CloudRegistryProvider.ECR.value,
            "--registry",
            registry,
            "--dest",
            dest,
            "--authfile",
            authfile_path,
        ],
        name=_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
    )


def append_acr_credential_exchange_init_container(
    pod, registry: str, authfile_path: str
) -> None:
    """Append the init container that mints ACR push credentials for ``pod``.

    Runs ``python -m mlrun mint-registry-credentials --provider acr`` (see
    :func:`mlrun.utils.registry_auth.mint_acr_authfile`), which exchanges the pod's Azure workload
    identity (federated JWT, injected by the ``azure.workload.identity/use`` label - see
    :func:`~services.api.utils.builder.base.resolve_builder_pod_labels`) for an AAD access token via
    azure-identity, then exchanges that AAD token for an ACR refresh token via ACR's
    ``/oauth2/exchange`` endpoint (no SDK covers this ACR-specific endpoint), writing the result to
    ``authfile_path``.

    :param pod: The Buildah build pod being constructed.
    :param registry: The ACR registry host.
    :param authfile_path: Where to write the docker-config-shaped authfile.
    """
    pod.append_init_container(
        _credential_exchange_image(),
        command=["python"],
        args=[
            "-m",
            "mlrun",
            "mint-registry-credentials",
            "--provider",
            CloudRegistryProvider.ACR.value,
            "--registry",
            registry,
            "--authfile",
            authfile_path,
        ],
        name=_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
    )


def gar_credential_exchange_script(registry: str, authfile_path: str) -> str:
    """Return the Python source that mints a GAR/GCR push token just-in-time.

    Unlike ECR/ACR, this does not run via the ``mint-registry-credentials`` CLI in an init
    container: it runs directly in the Buildah main container, which has no mlrun (or any cloud SDK)
    installed - only the stock image's python3/urllib. Minted immediately before both ``buildah bud``
    and ``buildah push`` (see :func:`~services.api.utils.builder.buildah._build_script`) to minimize
    the gap between mint and use - see the module docstring for why GCP token caching makes that gap
    the actual risk, not a fixed TTL window. Relies on GKE Workload Identity transparently
    intercepting the metadata-server call for the build pod's service account - no federated-token
    plumbing is needed on mlrun's side.

    :param registry: The GAR/GCR registry host.
    :param authfile_path: Where to write the docker-config-shaped authfile.
    :return: The Python source to run (e.g. via ``python3 -c``).
    """
    # kept terse - this is base64-embedded into a pod env var, so every byte here is sent over the
    # wire. No `with` blocks: the process exits right after, so explicit close/flush isn't needed.
    lines = [
        "import base64, json, urllib.request",
        f"req = urllib.request.Request({_GCP_METADATA_TOKEN_URL!r}, headers="
        "{'Metadata-Flavor': 'Google'})",
        "token = json.load(urllib.request.urlopen(req, timeout=10))['access_token']",
        "auth = base64.b64encode(b'oauth2accesstoken:' + token.encode()).decode()",
        f"json.dump({{'auths': {{{registry!r}: {{'auth': auth}}}}}}, open({authfile_path!r}, 'w'))",
    ]
    return "\n".join(lines)


def _credential_exchange_image() -> str:
    image = config.httpdb.builder.buildah_registry_credential_exchange_image
    if not image:
        image = mlrun.utils.enrich_image_url(_DEFAULT_CREDENTIAL_EXCHANGE_IMAGE)
    return image


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
