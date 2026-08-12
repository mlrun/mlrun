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

# Tests for the Buildah base-image compatibility check (ML-12990): buildah bud rejects a base
# image whose manifest mixes OCI and Docker layer media types, so this is detected before
# scheduling a Buildah pod. Fails open by design - any inconclusive case (missing/unusable
# credentials, network error, disallowed host) must return None, never raise.

import json
import unittest.mock

import pytest
import requests

import framework.utils.singletons.k8s
import services.api.utils.builder.base_image_compat as base_image_compat

_OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
_OCI_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
_DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
_DOCKER_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"


def _manifest(config_media_type, layer_media_types, media_type=None, manifests=None):
    body = {"config": {"mediaType": config_media_type}}
    if layer_media_types is not None:
        body["layers"] = [{"mediaType": mt} for mt in layer_media_types]
    if media_type:
        body["mediaType"] = media_type
    if manifests is not None:
        body["manifests"] = manifests
    return body


def _response(status_code=200, json_body=None, headers=None):
    response = unittest.mock.Mock(spec=requests.Response)
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.headers = headers or {}
    response.json.return_value = json_body or {}
    return response


# --- base_image_uses_mixed_media_types: the leaf-manifest cases -----------------------------------


def test_mixed_oci_and_docker_layers_is_detected(monkeypatch):
    manifest = _manifest(_OCI_CONFIG, [_OCI_LAYER, _DOCKER_LAYER])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _response(json_body=manifest))
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is True
    )


def test_homogeneous_oci_manifest_is_compatible(monkeypatch):
    manifest = _manifest(_OCI_CONFIG, [_OCI_LAYER, _OCI_LAYER])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _response(json_body=manifest))
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is False
    )


def test_homogeneous_docker_manifest_is_compatible(monkeypatch):
    manifest = _manifest(_DOCKER_CONFIG, [_DOCKER_LAYER, _DOCKER_LAYER])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _response(json_body=manifest))
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is False
    )


def test_config_family_mismatch_against_layers_is_detected(monkeypatch):
    # deliberately conservative: an OCI config with all-Docker layers is flagged too (see the
    # comment on _has_mixed_media_types for why erring towards over-flagging is the safe side).
    manifest = _manifest(_OCI_CONFIG, [_DOCKER_LAYER, _DOCKER_LAYER])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _response(json_body=manifest))
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is True
    )


# --- manifest-list / index resolution --------------------------------------------------------------


def test_manifest_list_resolves_amd64_linux_entry_then_checks_it(monkeypatch):
    index = _manifest(
        None,
        None,
        media_type="application/vnd.oci.image.index.v1+json",
        manifests=[
            {
                "platform": {"architecture": "arm64", "os": "linux"},
                "digest": "sha256:arm",
            },
            {
                "platform": {"architecture": "amd64", "os": "linux"},
                "digest": "sha256:amd",
            },
        ],
    )
    leaf = _manifest(_OCI_CONFIG, [_OCI_LAYER])
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _response(json_body=leaf if "sha256:amd" in url else index)

    monkeypatch.setattr(requests, "get", fake_get)
    result = base_image_compat.base_image_uses_mixed_media_types(
        "reg.example.com/repo", None
    )
    assert result is False
    assert calls[0].endswith("manifests/latest")
    assert any("sha256:amd" in c for c in calls[1:])


def test_manifest_list_without_amd64_linux_entry_is_inconclusive(monkeypatch):
    index = _manifest(
        None,
        None,
        media_type="application/vnd.docker.distribution.manifest.list.v2+json",
        manifests=[
            {
                "platform": {"architecture": "arm64", "os": "linux"},
                "digest": "sha256:arm",
            }
        ],
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: _response(json_body=index))
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is None
    )


def test_index_without_self_declared_media_type_is_still_detected_via_manifests_key(
    monkeypatch,
):
    # some registries omit the top-level "mediaType" on an index - the presence of "manifests"
    # (only ever on an index, never a leaf manifest) is the fallback signal.
    index = _manifest(
        None,
        None,
        manifests=[
            {
                "platform": {"architecture": "amd64", "os": "linux"},
                "digest": "sha256:amd",
            }
        ],
    )
    leaf = _manifest(_OCI_CONFIG, [_DOCKER_LAYER])
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **k: _response(
            json_body=index if "sha256:amd" not in url else leaf
        ),
    )
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is True
    )


# --- fails open: credentials, network, and disallowed hosts ---------------------------------------


def test_no_credentials_and_401_is_inconclusive(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _response(status_code=401, headers={})
    )
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is None
    )


def test_bearer_challenge_token_exchange_then_retry_succeeds(monkeypatch):
    manifest = _manifest(_OCI_CONFIG, [_OCI_LAYER])
    challenge = _response(
        status_code=401,
        headers={
            "Www-Authenticate": (
                'Bearer realm="https://auth.example.com/token",'
                'service="reg.example.com",scope="repository:repo:pull"'
            )
        },
    )
    token_response = _response(json_body={"token": "minted-token"})
    manifest_response = _response(json_body=manifest)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("headers", {}).get("Authorization")))
        if "auth.example.com" in url:
            return token_response
        if len(calls) == 1:
            return challenge
        return manifest_response

    monkeypatch.setattr(requests, "get", fake_get)
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is False
    )
    assert calls[-1][1] == "Bearer minted-token"


def test_bearer_challenge_with_http_realm_is_rejected(monkeypatch):
    challenge = _response(
        status_code=401,
        headers={"Www-Authenticate": 'Bearer realm="http://auth.example.com/token"'},
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: challenge)
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is None
    )


def test_bearer_challenge_with_metadata_realm_is_rejected(monkeypatch):
    challenge = _response(
        status_code=401,
        headers={"Www-Authenticate": 'Bearer realm="https://169.254.169.254/token"'},
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: challenge)
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is None
    )


@pytest.mark.parametrize(
    "base_image",
    [
        "localhost/repo",
        "127.0.0.1/repo",
        "169.254.169.254/repo",
        "metadata.google.internal/repo",
    ],
)
def test_disallowed_host_never_makes_a_request(monkeypatch, base_image):
    get = unittest.mock.Mock()
    monkeypatch.setattr(requests, "get", get)
    assert base_image_compat.base_image_uses_mixed_media_types(base_image, None) is None
    get.assert_not_called()


def test_network_error_is_inconclusive_not_raised(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        unittest.mock.Mock(side_effect=requests.ConnectionError("boom")),
    )
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is None
    )


def test_timeout_is_inconclusive_not_raised(monkeypatch):
    monkeypatch.setattr(
        requests, "get", unittest.mock.Mock(side_effect=requests.Timeout("boom"))
    )
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", None
        )
        is None
    )


# --- static docker-config secret auth --------------------------------------------------------------


def _patch_secret(monkeypatch, dockerconfig, expected_namespace=None):
    helper = unittest.mock.Mock()

    def get_secret_data(secret_name, namespace):
        if expected_namespace is not None:
            assert namespace == expected_namespace
        return {".dockerconfigjson": json.dumps(dockerconfig)}

    helper.get_secret_data = unittest.mock.Mock(side_effect=get_secret_data)
    monkeypatch.setattr(
        framework.utils.singletons.k8s, "get_k8s_helper", lambda **kwargs: helper
    )


def test_static_secret_matching_host_port_registry_is_used(monkeypatch):
    # regression for the .hostname-vs-.netloc bug: a host:port registry must still match its own
    # docker-config entry.
    _patch_secret(
        monkeypatch, {"auths": {"registry.local:5000": {"auth": "dXNlcjpwYXNz"}}}
    )
    manifest = _manifest(_OCI_CONFIG, [_OCI_LAYER])
    captured_headers = {}

    def fake_get(url, headers=None, **kwargs):
        captured_headers.update(headers or {})
        return _response(json_body=manifest)

    monkeypatch.setattr(requests, "get", fake_get)
    result = base_image_compat.base_image_uses_mixed_media_types(
        "registry.local:5000/proj/img", "my-secret"
    )
    assert result is False
    assert captured_headers["Authorization"] == "Basic dXNlcjpwYXNz"


def test_static_secret_namespace_is_threaded_through(monkeypatch):
    # regression for the namespace-defaulting bug: the caller's namespace, not the k8s helper's
    # global default, must be used to look up the secret.
    _patch_secret(
        monkeypatch,
        {"auths": {"reg.example.com": {"auth": "dXNlcjpwYXNz"}}},
        expected_namespace="some-other-namespace",
    )
    manifest = _manifest(_OCI_CONFIG, [_OCI_LAYER])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _response(json_body=manifest))
    result = base_image_compat.base_image_uses_mixed_media_types(
        "reg.example.com/repo", "my-secret", "some-other-namespace"
    )
    assert result is False


def test_static_secret_without_matching_host_falls_back_to_anonymous(monkeypatch):
    _patch_secret(
        monkeypatch, {"auths": {"unrelated-registry.example.com": {"auth": "x"}}}
    )
    manifest = _manifest(_OCI_CONFIG, [_OCI_LAYER])
    captured_headers = {}

    def fake_get(url, headers=None, **kwargs):
        captured_headers.update(headers or {})
        return _response(json_body=manifest)

    monkeypatch.setattr(requests, "get", fake_get)
    base_image_compat.base_image_uses_mixed_media_types(
        "reg.example.com/repo", "my-secret"
    )
    assert "Authorization" not in captured_headers


def test_unreadable_secret_falls_back_to_anonymous_instead_of_aborting(monkeypatch):
    helper = unittest.mock.Mock()
    helper.get_secret_data = unittest.mock.Mock(
        side_effect=RuntimeError("k8s api down")
    )
    monkeypatch.setattr(
        framework.utils.singletons.k8s, "get_k8s_helper", lambda **kwargs: helper
    )
    manifest = _manifest(_OCI_CONFIG, [_OCI_LAYER])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _response(json_body=manifest))
    # the secret lookup failed, but the anonymous attempt below it still gets a chance to run.
    assert (
        base_image_compat.base_image_uses_mixed_media_types(
            "reg.example.com/repo", "my-secret"
        )
        is False
    )


# --- image-reference parsing ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_image,expected",
    [
        (
            "registry.local:5000/proj/img:tag",
            ("registry.local:5000", "proj/img", "tag"),
        ),
        ("registry.local:5000/proj/img", ("registry.local:5000", "proj/img", "latest")),
        ("mlrun/mlrun:1.9", ("registry-1.docker.io", "mlrun/mlrun", "1.9")),
        ("python:3.11-slim", ("registry-1.docker.io", "library/python", "3.11-slim")),
        ("localhost/myimage:latest", ("localhost", "myimage", "latest")),
        ("localhost:5000/myimage", ("localhost:5000", "myimage", "latest")),
        (
            "registry.local/proj/img@sha256:" + "a" * 64,
            ("registry.local", "proj/img", "sha256:" + "a" * 64),
        ),
        (
            "registry.local/proj/img@sha512:" + "b" * 128,
            ("registry.local", "proj/img", "sha512:" + "b" * 128),
        ),
    ],
)
def test_parse_image_reference(base_image, expected):
    assert base_image_compat._parse_image_reference(base_image) == expected
