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
each provider mints its own credential from the pod's existing workload identity - for both the
push destination and the base image being pulled (ML-12961):

* **ECR** and **ACR** — an init container on the MLRun image (already carries boto3 /
  azure-identity, and mlrun itself) runs ``python -m mlrun mint-registry-credentials`` (see
  :mod:`mlrun.utils.registry_auth` for the actual credential-exchange logic) and merges the result
  into a shared authfile - the same ``python -m mlrun <subcommand>`` convention Kaniko's
  source-fetch init container uses. A push-side and a pull-side exchange run as two distinctly-named
  init containers against the same authfile, unless they're the same host, in which case the
  pull-side one is skipped as redundant.
* **GAR / GCR** — no init container: GCP tokens are cached and reused by the metadata server across
  callers until fewer than 5 minutes remain before expiry, so *any* mint can hand back a token with
  as little as ~5 minutes of remaining life, regardless of its original TTL. Minting just-in-time,
  immediately before each Buildah step that needs it (see :mod:`services.api.utils.builder.buildah`),
  minimizes the gap between mint and use rather than trying to outlast a fixed TTL. This path can't
  use the same init-container convention as ECR/ACR: it runs inline in the Buildah main container,
  which has no mlrun (or any cloud SDK) installed - only the stock image's python3/urllib. A
  push-side and a pull-side script are minted independently - even when they're the same host -
  and both merge into the same authfile.

A static docker-config secret (``secret_name``) may authenticate a *different* host than the ones
above - e.g. a private base-image registry alongside a cloud push destination (ML-12988). It's
copied into the shared authfile via its own init container rather than mounted there directly - see
:func:`append_secret_authfile_init_container`.

Every credential-exchange step - ECR/ACR's init containers and GAR's inline script alike - soft-fails
(warns, doesn't fail the pod) on a mint failure rather than aborting the build, matching nuclio's own
Buildah registry-login containers (NUC-688) - see :func:`soft_fail_script`.

None of the generated scripts or CLI args ever log the minted token.
"""

import shlex
from urllib.parse import urlparse

import mlrun.utils
import mlrun.utils.helpers
from mlrun.config import config
from mlrun.utils.registry_auth import CloudRegistryProvider

_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME = "registry-credential-exchange"
# used when a base-image (pull) exchange runs alongside a push-side one in the same pod - container
# names must be unique within a pod.
PULL_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME = "registry-credential-exchange-pull"

_COPY_SECRET_AUTHFILE_INIT_CONTAINER_NAME = "copy-registry-auth-secret"

# where a configured docker-config secret is mounted read-only before
# append_secret_authfile_init_container copies it into the shared authfile.
SECRET_AUTHFILE_DIR = "/auth-secret"
SECRET_AUTHFILE_PATH = f"{SECRET_AUTHFILE_DIR}/config.json"

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


def registry_from_image(image: str) -> str | None:
    """Return the explicit registry host in ``image``, or ``None`` if it has none.

    Mirrors Docker's own reference-parsing rule: the segment before the first ``/`` is a registry
    host only if it contains a ``.`` or ``:``, or is exactly ``localhost`` - otherwise (e.g.
    ``python:3.11-slim``, or a Docker Hub ``some-org/some-image:tag``) there's no explicit registry,
    the image is implicitly on Docker Hub.

    :param image: A full image reference (``[registry/]repository[:tag]``).
    :return: The registry host, or ``None``.
    """
    host, sep, _ = image.partition("/")
    if not sep or ("." not in host and ":" not in host and host != "localhost"):
        return None
    return host


def soft_fail_script(command: list[str], kind: str) -> str:
    """Wrap ``command`` so a failure logs a warning and exits 0 instead of failing the pod.

    Mirrors nuclio's own Buildah registry-login containers (NUC-688): a cloud credential mint
    failing - e.g. workload identity genuinely isn't configured for this host - must not abort the
    build outright (ML-12988). The build falls back to whatever a secret already provided for that
    host, or, if nothing covers it, buildah's own push/pull surfaces a clear auth error instead of
    an opaque credential-helper traceback. Used by ECR/ACR's init containers below and, inline, by
    :func:`gar_credential_exchange_script`'s callers in
    :mod:`services.api.utils.builder.buildah`.

    :param command: The command to run, as argv (shell-quoted here, never shell-interpreted itself).
    :param kind: The provider name, for the warning message (e.g. ``"ECR"``).
    :return: A ``/bin/sh -c`` script string.
    """
    return (
        f"{shlex.join(command)} || "
        f"echo 'WARNING: failed to mint {kind} registry credentials' >&2"
    )


def append_secret_authfile_init_container(pod, authfile_path: str) -> None:
    """Append the init container that copies a mounted docker-config secret into ``authfile_path``.

    Used when a static secret and cloud credential exchange are both configured (ML-12988): the
    secret must already be mounted at :data:`SECRET_AUTHFILE_DIR` (read-only) rather than directly
    at ``authfile_path``, since the credential-exchange init containers/scripts that run after this
    one need to merge into that file, which a secret-backed volume mount doesn't allow. Runs first,
    so those merges land on top of the secret's own entries.

    :param pod: The Buildah build pod being constructed.
    :param authfile_path: Where to copy the secret's docker-config content to.
    """
    # buildah_image, not _credential_exchange_image(): the main container always pulls it anyway,
    # so this copy step never costs an extra image pull - unlike the credential-exchange image,
    # which a GAR-only build (no ECR/ACR init container of its own) would otherwise never need. It
    # already ships coreutils (see BuildahBackend's docstring), so `cp` is available.
    pod.append_init_container(
        config.httpdb.builder.buildah_image,
        command=["cp"],
        args=[SECRET_AUTHFILE_PATH, authfile_path],
        name=_COPY_SECRET_AUTHFILE_INIT_CONTAINER_NAME,
    )


def append_ecr_credential_exchange_init_container(
    pod,
    registry: str,
    authfile_path: str,
    dest: str | None = None,
    container_name: str = _CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
) -> None:
    """Append the init container that mints ECR credentials for ``pod``.

    Runs ``python -m mlrun mint-registry-credentials --provider ecr`` (see
    :func:`mlrun.utils.registry_auth.mint_ecr_authfile`), which uses boto3 with the pod's own AWS
    credentials - IRSA or instance role, resolved via the build pod's service account, see
    :func:`~services.api.utils.builder.base.resolve_build_pod_spec_attributes` - to mint a
    short-lived authorization token, merging it into ``authfile_path``. When ``dest`` is given
    (a push), the target ECR repository is also created (idempotent); omit it for a pull-only
    (base-image) exchange, which skips repository creation. A failed mint logs a warning and exits
    0 rather than failing the pod (ML-12988) - see :func:`soft_fail_script`.

    :param pod: The Buildah build pod being constructed.
    :param registry: The ECR registry host.
    :param authfile_path: Where to merge the docker-config-shaped authfile entry.
    :param dest: The fully resolved destination image reference (for the repository name). Omit
        for a base-image pull exchange.
    :param container_name: The init container's name - distinct names are required when both a
        push-side and a pull-side exchange run in the same pod.
    """
    mint_args = [
        "-m",
        "mlrun",
        "mint-registry-credentials",
        "--provider",
        CloudRegistryProvider.ECR.value,
        "--registry",
        registry,
    ]
    if dest:
        mint_args += ["--dest", dest]
    mint_args += ["--authfile", authfile_path]
    pod.append_init_container(
        _credential_exchange_image(),
        command=["/bin/sh", "-c"],
        args=[soft_fail_script(["python", *mint_args], "ECR")],
        name=container_name,
    )


def append_acr_credential_exchange_init_container(
    pod,
    registry: str,
    authfile_path: str,
    container_name: str = _CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
) -> None:
    """Append the init container that mints an ACR credential for ``pod``.

    Runs ``python -m mlrun mint-registry-credentials --provider acr`` (see
    :func:`mlrun.utils.registry_auth.mint_acr_authfile`), which exchanges the pod's Azure workload
    identity (federated JWT, injected by the ``azure.workload.identity/use`` label - see
    :func:`~services.api.utils.builder.base.resolve_builder_pod_labels`) for an AAD access token via
    azure-identity, then exchanges that AAD token for an ACR refresh token via ACR's
    ``/oauth2/exchange`` endpoint (no SDK covers this ACR-specific endpoint), merging the result
    into ``authfile_path``. Used for both push and base-image-pull credentials. A failed mint logs
    a warning and exits 0 rather than failing the pod (ML-12988) - see :func:`soft_fail_script`.

    :param pod: The Buildah build pod being constructed.
    :param registry: The ACR registry host.
    :param authfile_path: Where to merge the docker-config-shaped authfile entry.
    :param container_name: The init container's name - distinct names are required when both a
        push-side and a pull-side exchange run in the same pod.
    """
    mint_args = [
        "-m",
        "mlrun",
        "mint-registry-credentials",
        "--provider",
        CloudRegistryProvider.ACR.value,
        "--registry",
        registry,
        "--authfile",
        authfile_path,
    ]
    pod.append_init_container(
        _credential_exchange_image(),
        command=["/bin/sh", "-c"],
        args=[soft_fail_script(["python", *mint_args], "ACR")],
        name=container_name,
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

    Merges into ``authfile_path`` rather than overwriting it: a push-side and a pull-side script
    (different GAR hosts, ML-12961), an ACR/ECR init container, or a copied-in static secret
    (ML-12988) may already have written to it - and any other top-level docker-config keys already
    there (``credHelpers``, ``credsStore``, ...) are preserved; only this one registry's entry is
    ever added or replaced.

    :param registry: The GAR/GCR registry host.
    :param authfile_path: Where to merge the docker-config-shaped authfile entry.
    :return: The Python source to run (e.g. via ``python3 -c``).
    """
    # kept terse - this is base64-embedded into a pod env var, so every byte here is sent over the
    # wire. No `with` blocks: the process exits right after, so explicit close/flush isn't needed.
    lines = [
        "import base64, json, os, urllib.request",
        f"req = urllib.request.Request({_GCP_METADATA_TOKEN_URL!r}, headers="
        "{'Metadata-Flavor': 'Google'})",
        "token = json.load(urllib.request.urlopen(req, timeout=10))['access_token']",
        "auth = base64.b64encode(b'oauth2accesstoken:' + token.encode()).decode()",
        f"doc = json.load(open({authfile_path!r})) if os.path.exists({authfile_path!r}) else {{}}",
        f"doc.setdefault('auths', {{}})[{registry!r}] = {{'auth': auth}}",
        f"json.dump(doc, open({authfile_path!r}, 'w'))",
    ]
    return "\n".join(lines)


def _credential_exchange_image() -> str:
    image = config.httpdb.builder.registry_credential_exchange_image
    if not image:
        # default_base_image (mlrun/mlrun unless the user overrides it) ships boto3 / azure-identity
        # via the "complete" extra. Falling back to it, rather than hardcoding "mlrun/mlrun", means a
        # user-configured default is honored consistently instead of silently ignored here.
        image = mlrun.utils.enrich_image_url(config.default_base_image)
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
