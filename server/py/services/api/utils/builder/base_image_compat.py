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
"""Detect base images Buildah can't build ``FROM`` (ML-12990).

``buildah bud`` deliberately refuses to build from a base image whose manifest mixes OCI
(``application/vnd.oci.*``) and legacy-Docker (``application/vnd.docker.*``) layer media types -
a long-open, upstream-owned limitation (buildah#5272, buildah#3668, podman#19949), not a bug in
this repo's Buildah invocation. Such mixed manifests are a routine byproduct of modern
``docker buildx build`` (SBOM/provenance attestations on by default since Docker 23 / buildx
0.10) and of Kaniko, so this is checked before scheduling a Buildah pod rather than only
discovered after a failed build.

Unlike Buildah's push-side cloud-registry credential exchange (see
:mod:`services.api.utils.builder.registry_auth`), this check runs in the API server process
itself, before any pod exists - it can only use credentials already available there (a static
docker-config secret) or an anonymous pull. It can't mint ECR/ACR/GAR credentials (that exchange
needs a pod's own workload identity), so those registries are treated as inconclusive unless
anonymous access happens to work.
"""

import json
from urllib.parse import urlparse

import requests

import mlrun.errors
import mlrun.utils

import framework.utils.singletons.k8s

_REQUEST_TIMEOUT_SECONDS = 5

_OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
_OCI_INDEX = "application/vnd.oci.image.index.v1+json"
_DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
_DOCKER_MANIFEST_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
_MANIFEST_ACCEPT_HEADER = ", ".join(
    [_OCI_MANIFEST, _OCI_INDEX, _DOCKER_MANIFEST, _DOCKER_MANIFEST_LIST]
)
_INDEX_MEDIA_TYPES = (_OCI_INDEX, _DOCKER_MANIFEST_LIST)

# the platform this check evaluates when a base image resolves to a multi-arch index - mlrun's
# own build/runtime images are amd64, and the attestation/provenance layers that typically cause
# the media-type mix are list-level, not per-arch, so checking any one real platform entry is
# representative even on an arm64 cluster.
_INSPECTED_PLATFORM = {"architecture": "amd64", "os": "linux"}

_DEFAULT_REGISTRY_HOST = "registry-1.docker.io"

# loopback and the cloud-metadata endpoints every major provider exposes on the link-local
# range - unlike an arbitrary internal/private registry (a legitimate, common target this check
# must still be able to reach), there is no legitimate reason for a base image's registry to
# resolve here. base_image is caller-supplied, so this check (like the real build pod's own pull)
# is reachable by anyone with build permissions; this denylist keeps that reachability from
# turning into a credential-theft primitive against the API server's own network position.
_DISALLOWED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "169.254.169.254",  # AWS / Azure / GCP / OpenStack metadata
    "metadata.google.internal",
    "100.100.100.200",  # Alibaba Cloud metadata
}


def base_image_uses_mixed_media_types(
    base_image: str, secret_name: str | None, namespace: str | None = None
) -> bool | None:
    """Return whether ``base_image``'s manifest mixes OCI and Docker layer media types.

    Fails open by design: any network error, timeout, missing/unusable credentials, or
    unparseable response returns ``None`` ("couldn't tell") rather than raising. A build must
    never be blocked, or mis-routed to the wrong backend, because a registry was briefly
    unreachable or needs credentials this check can't obtain.

    :param base_image: The (enriched) base image reference the Dockerfile builds ``FROM``.
    :param secret_name: The docker-config secret used for the build's push auth, if any - reused
        here when it also carries an entry for the base image's registry host.
    :param namespace: The k8s namespace ``secret_name`` lives in.
    :return: ``True`` if the manifest mixes OCI/Docker media types (Buildah will reject it),
        ``False`` if it doesn't, or ``None`` if this couldn't be determined.
    """
    try:
        registry, repository, reference = _parse_image_reference(base_image)
        if _is_disallowed_host(urlparse(f"https://{registry}").hostname):
            return None
        auth_header = _resolve_static_auth_header(registry, secret_name, namespace)
        manifest = _fetch_manifest(registry, repository, reference, auth_header)
        if manifest is None:
            return None

        # a compliant registry echoes the index mediaType, but some don't - "manifests" (a
        # per-platform list) only ever appears on an index/manifest-list, never on a leaf image
        # manifest, so it's a reliable fallback signal.
        if manifest.get("mediaType") in _INDEX_MEDIA_TYPES or "manifests" in manifest:
            platform_digest = _select_platform_digest(manifest)
            if platform_digest is None:
                return None
            manifest = _fetch_manifest(
                registry, repository, platform_digest, auth_header
            )
            if manifest is None:
                return None

        return _has_mixed_media_types(manifest)
    except Exception as exc:
        mlrun.utils.logger.debug(
            "Could not inspect base image manifest for Buildah compatibility",
            base_image=base_image,
            exc=mlrun.errors.err_to_str(exc),
        )
        return None


def _parse_image_reference(base_image: str) -> tuple[str, str, str]:
    """Split ``base_image`` into ``(registry_host, repository, reference)``.

    :param base_image: A full image reference (e.g. ``registry.local:5000/proj/img:tag``, or a
        Docker-Hub shorthand like ``mlrun/mlrun:1.9`` / ``python:3.11-slim``).
    :return: The registry host, the repository path and the tag or ``<algo>:<hex>`` digest.
    """
    # '@' is reserved for the digest separator in a reference - never split the tag on it.
    path, sep, reference = base_image.partition("@")
    if not sep:
        path, sep, tag = base_image.rpartition(":")
        # no ':' at all, or the last ':' was a "host:port" separator rather than a tag (a real
        # tag never contains '/') - either way, there's no explicit tag.
        if not sep or "/" in tag:
            path, reference = base_image, "latest"
        else:
            reference = tag

    first_segment, _, rest = path.partition("/")
    if rest and (
        "." in first_segment or ":" in first_segment or first_segment == "localhost"
    ):
        return first_segment, rest, reference

    repository = path if "/" in path else f"library/{path}"
    return _DEFAULT_REGISTRY_HOST, repository, reference


def _is_disallowed_host(hostname: str | None) -> bool:
    if not hostname:
        return True
    hostname = hostname.lower()
    return hostname in _DISALLOWED_HOSTS or hostname.startswith(("169.254.", "fe80:"))


def _resolve_static_auth_header(
    registry: str, secret_name: str | None, namespace: str | None
) -> str | None:
    if not secret_name:
        return None
    try:
        secret_data = framework.utils.singletons.k8s.get_k8s_helper(
            silent=True
        ).get_secret_data(secret_name, namespace)
        dockerconfig = json.loads(secret_data.get(".dockerconfigjson", "{}"))
    except Exception:
        return None

    for host, entry in dockerconfig.get("auths", {}).items():
        # .netloc, not .hostname: .hostname silently drops the port, which would make a
        # host:port registry (a common self-hosted-registry pattern) never match its own
        # docker-config entry.
        if urlparse(f"https://{host}").netloc == registry and entry.get("auth"):
            return f"Basic {entry['auth']}"
    return None


def _fetch_manifest(
    registry: str, repository: str, reference: str, auth_header: str | None
) -> dict | None:
    url = f"https://{registry}/v2/{repository}/manifests/{reference}"
    headers = {"Accept": _MANIFEST_ACCEPT_HEADER}
    if auth_header:
        headers["Authorization"] = auth_header

    response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 401:
        token = _exchange_bearer_token(response, auth_header)
        if token is None:
            return None
        headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)

    if not response.ok:
        return None
    return response.json()


def _exchange_bearer_token(
    challenge_response: requests.Response, auth_header: str | None
) -> str | None:
    """Follow the registry's ``Www-Authenticate: Bearer`` challenge to mint a pull token."""
    challenge = challenge_response.headers.get("Www-Authenticate", "")
    if not challenge.lower().startswith("bearer "):
        return None

    params = {}
    for part in challenge[len("bearer ") :].split(","):
        key, _, value = part.strip().partition("=")
        params[key] = value.strip('"')

    realm = params.pop("realm", None)
    # the registry (attacker-controlled, in the worst case - it's whatever the caller put in
    # base_image) picks this URL; requiring https and rejecting the same disallowed hosts as the
    # initial request guards against it redirecting this request - and the static secret's
    # credentials, if any - to an unencrypted or internal-only endpoint.
    if not realm or not realm.startswith("https://"):
        return None
    if _is_disallowed_host(urlparse(realm).hostname):
        return None

    headers = {"Authorization": auth_header} if auth_header else {}
    token_response = requests.get(
        realm, params=params, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
    )
    if not token_response.ok:
        return None
    body = token_response.json()
    return body.get("token") or body.get("access_token")


def _select_platform_digest(index_manifest: dict) -> str | None:
    for entry in index_manifest.get("manifests", []):
        platform = entry.get("platform", {})
        if (
            platform.get("architecture") == _INSPECTED_PLATFORM["architecture"]
            and platform.get("os") == _INSPECTED_PLATFORM["os"]
        ):
            return entry.get("digest")
    return None


def _has_mixed_media_types(manifest: dict) -> bool:
    # config is included alongside layers, not just layers against each other: the real-world
    # repro behind this check (buildah#3668) had an OCI config with mostly-Docker layers, and
    # nalind's own read of the failure ("manifests that mark layers with types from different
    # specs") doesn't draw a config/layer distinction either. Erring towards over-flagging here
    # is the safe direction - a false positive costs an unnecessary Kaniko fallback, a false
    # negative reproduces the exact build failure this check exists to prevent.
    media_types = [manifest.get("config", {}).get("mediaType", "")]
    media_types += [layer.get("mediaType", "") for layer in manifest.get("layers", [])]
    families = {family for mt in media_types if (family := _media_type_family(mt))}
    return len(families) > 1


def _media_type_family(media_type: str) -> str | None:
    if media_type.startswith("application/vnd.oci."):
        return "oci"
    if media_type.startswith("application/vnd.docker."):
        return "docker"
    return None
