# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Wire-level contract of the token, user and version calls MLRun makes to Orca.

A local HTTP server stands in for Orca, so the tests apply to whichever HTTP client MLRun
uses to reach it. They call the callers (``crud.Secrets``, ``utils.helpers``) and assert what Orca
receives (method, path, body, whose identity) and what MLRun raises.
"""

import socket
import time
import unittest.mock

import jwt
import pytest
import pytest_httpserver
import werkzeug

import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.errors
from mlrun.utils.logger import context_id_var

import framework.utils.clients.service_account_token
import framework.utils.singletons.k8s
import services.api.crud
import services.api.utils.helpers

SA_TOKEN = "sa-token"
USER_BEARER = "Bearer user-jwt"
ADMIN_BEARER = "Bearer admin-jwt"
CTX_ID = "test-context-id"
OFFLINE_TOKEN = "offline-xyz"
UPSTREAM_MESSAGE = "upstream-message"
UPSTREAM_CTX = "upstream-ctx"

REFRESH_TOKENS_PATH = "/api/v1/authentication/refresh-access-tokens"
REFRESH_TOKEN_PATH = "/api/v1/authentication/refresh-access-token"
REVOKE_PATH = "/api/v1/authentication/revoke-offline-token"
SYSTEM_INFO_PATH = "/api/v1/info/system"

# Statuses MLRun maps to a specific error class, plus unmapped ones that fall back to the
# class each call raises by default.
MAPPED_ERRORS = [
    (400, mlrun.errors.MLRunBadRequestError),
    (401, mlrun.errors.MLRunUnauthorizedError),
    (403, mlrun.errors.MLRunAccessDeniedError),
    (404, mlrun.errors.MLRunNotFoundError),
    (409, mlrun.errors.MLRunConflictError),
    (412, mlrun.errors.MLRunPreconditionFailedError),
    (500, mlrun.errors.MLRunInternalServerError),
    (501, mlrun.errors.MLRunNotImplementedServerError),
    (503, mlrun.errors.MLRunServiceUnavailableError),
]
UNMAPPED_STATUSES = [429, 502]


@pytest.fixture(autouse=True)
def service_account_token(monkeypatch):
    monkeypatch.setattr(
        framework.utils.clients.service_account_token.Client,
        "token",
        property(lambda self: SA_TOKEN),
    )


@pytest.fixture
def orca(httpserver: pytest_httpserver.HTTPServer):
    mlrun.mlconf.iguazio_api_url = httpserver.url_for("").rstrip("/")
    yield httpserver
    httpserver.check_assertions()


@pytest.fixture
def unreachable_orca():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mlrun.mlconf.iguazio_api_url = f"http://127.0.0.1:{port}"


@pytest.fixture
def secrets_provider():
    provider = unittest.mock.create_autospec(
        mlrun.common.secrets.SecretProviderInterface, instance=True
    )
    provider.store_user_token_secret.return_value = (
        mlrun.common.schemas.SecretEventActions.created
    )
    provider.get_user_token_secret_value.return_value = OFFLINE_TOKEN
    services.api.crud.Secrets().secrets_provider = provider
    return provider


@pytest.fixture
def context_id():
    reset_token = context_id_var.set(CTX_ID)
    yield CTX_ID
    context_id_var.reset(reset_token)


@pytest.fixture
def user_tokens(monkeypatch) -> list[tuple[str, str]]:
    """
    The user's stored tokens as (name, value) pairs, in listing order. Replaces the K8s reads
    only, so the igz.yml document is still built by the real helper.
    """
    tokens: list[tuple[str, str]] = []

    def get_user_token_secret_value(self, user_id, token_name, namespace=None):
        for name, value in tokens:
            if name == token_name:
                return value
        raise mlrun.errors.MLRunNotFoundError(f"Token '{token_name}' not found")

    def list_user_token_secret_values(self, user_id, namespace=None):
        return [
            mlrun.common.schemas.SecretToken(name=name, token=value)
            for name, value in tokens
        ]

    monkeypatch.setattr(
        framework.utils.singletons.k8s.K8sHelper,
        "get_user_token_secret_value",
        get_user_token_secret_value,
    )
    monkeypatch.setattr(
        framework.utils.singletons.k8s.K8sHelper,
        "list_user_token_secret_values",
        list_user_token_secret_values,
    )
    return tokens


# --- Batch refresh: POST refresh-access-tokens, service-account identity ---


def test_store_secret_tokens_refreshes_via_service_account(orca, secrets_provider):
    first, second = _offline_token("jane-id"), _offline_token("jane-id")
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_json(
        _refresh_tokens_response(first, second)
    )

    response = services.api.crud.Secrets().store_secret_tokens(
        secret_tokens=[
            mlrun.common.schemas.SecretToken(name="first", token=first),
            mlrun.common.schemas.SecretToken(name="second", token=second),
        ],
        auth_info=_user_auth_info(),
    )

    request = _only_request(orca, "POST", REFRESH_TOKENS_PATH)
    _assert_service_account_identity(request)
    body = _json_body(request)
    assert set(body) <= {"refreshTokens", "mode"}
    assert body["refreshTokens"] == [first, second]
    # No mode is chosen, so Orca applies its default; 0 is that default's wire value
    assert body.get("mode", 0) == 0
    assert secrets_provider.store_user_token_secret.call_count == 2
    assert response.created_tokens == ["first", "second"]


def test_store_secret_tokens_multi_status_stores_every_token(orca, secrets_provider):
    valid, rejected = _offline_token("jane-id"), _offline_token("jane-id")
    multi_status = _refresh_tokens_response(valid, rejected)
    multi_status["items"][1] = {
        "status": {"statusCode": 401, "errorMessage": "Token is not active"}
    }
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_json(
        multi_status, status=207
    )

    response = services.api.crud.Secrets().store_secret_tokens(
        secret_tokens=[
            mlrun.common.schemas.SecretToken(name="valid", token=valid),
            mlrun.common.schemas.SecretToken(name="rejected", token=rejected),
        ],
        auth_info=_user_auth_info(),
    )

    # Per-token results inside a 207 are not inspected, so a token Orca rejected is still
    # stored. Known and kept on purpose; checking per-token status is a separate change.
    assert secrets_provider.store_user_token_secret.call_count == 2
    assert response.created_tokens == ["valid", "rejected"]


@pytest.mark.parametrize(
    "status_code, expected_error",
    MAPPED_ERRORS
    + [(status, mlrun.errors.MLRunUnauthorizedError) for status in UNMAPPED_STATUSES],
)
def test_store_secret_tokens_refresh_failure(
    orca, secrets_provider, status_code, expected_error
):
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_json(
        _error_body(status_code), status=status_code
    )

    with pytest.raises(expected_error) as exc_info:
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens=[
                mlrun.common.schemas.SecretToken(
                    name="token", token=_offline_token("jane-id")
                )
            ],
            auth_info=_user_auth_info(),
        )

    # pytest.raises also accepts subclasses, and some mapped classes have them
    # (e.g. MLRunNotFoundError), so pin the exact class.
    assert exc_info.type is expected_error
    _assert_upstream_details(exc_info.value)
    _only_request(orca, "POST", REFRESH_TOKENS_PATH)
    secrets_provider.store_user_token_secret.assert_not_called()


def test_store_secret_tokens_refresh_non_json_error(orca, secrets_provider):
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_data(
        "<html>bad gateway</html>", status=500, content_type="text/html"
    )

    with pytest.raises(mlrun.errors.MLRunInternalServerError):
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens=[
                mlrun.common.schemas.SecretToken(
                    name="token", token=_offline_token("jane-id")
                )
            ],
            auth_info=_user_auth_info(),
        )

    _only_request(orca, "POST", REFRESH_TOKENS_PATH)
    secrets_provider.store_user_token_secret.assert_not_called()


def test_store_secret_tokens_refresh_unreachable(unreachable_orca, secrets_provider):
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens=[
                mlrun.common.schemas.SecretToken(
                    name="token", token=_offline_token("jane-id")
                )
            ],
            auth_info=_user_auth_info(),
        )

    secrets_provider.store_user_token_secret.assert_not_called()


def test_store_secret_tokens_service_account_caller_skips_refresh(
    orca, secrets_provider
):
    services.api.crud.Secrets().store_secret_tokens(
        secret_tokens=[
            mlrun.common.schemas.SecretToken(
                name="token", token=_offline_token("jane-id")
            )
        ],
        auth_info=mlrun.common.schemas.AuthInfo(
            username="orca-sa",
            user_id="orca-sa-id",
            kind=mlrun.common.schemas.AuthInfoKind.service_account,
        ),
        username="jane",
    )

    assert orca.log == []
    secrets_provider.store_user_token_secret.assert_called_once()


def test_store_secret_tokens_refresh_propagates_context_id(
    orca, secrets_provider, context_id
):
    token = _offline_token("jane-id")
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_json(
        _refresh_tokens_response(token)
    )

    services.api.crud.Secrets().store_secret_tokens(
        secret_tokens=[mlrun.common.schemas.SecretToken(name="token", token=token)],
        auth_info=_user_auth_info(),
    )

    request = _only_request(orca, "POST", REFRESH_TOKENS_PATH)
    assert request.headers.get(mlrun.common.schemas.HeaderNames.igz_ctx) == context_id


# --- Revoke: POST revoke-offline-token, forwarded user identity ---


@pytest.fixture
def revocation_required():
    # Older than the first platform version that revokes the session itself
    mlrun.mlconf.oris_version = "1.0.0"


def test_delete_secret_token_revokes_with_forwarded_identity(
    orca, secrets_provider, revocation_required
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    response = services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    request = _only_request(orca, "POST", REVOKE_PATH)
    _assert_forwarded_identity(request, USER_BEARER)
    assert _json_body(request) == {"token": OFFLINE_TOKEN}
    secrets_provider.delete_user_token_secret.assert_called_once_with(
        user_id="jane-id", token_name="my-token"
    )
    assert response.deleted is True
    assert response.username == "jane"


@pytest.mark.parametrize(
    "status_code, expected_error",
    [
        (401, mlrun.errors.MLRunUnauthorizedError),
        (403, mlrun.errors.MLRunAccessDeniedError),
        (500, mlrun.errors.MLRunInternalServerError),
        (503, mlrun.errors.MLRunServiceUnavailableError),
    ]
    + [(status, mlrun.errors.MLRunUnauthorizedError) for status in UNMAPPED_STATUSES],
)
def test_delete_secret_token_revoke_failure_keeps_secret(
    orca, secrets_provider, revocation_required, status_code, expected_error
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json(
        _error_body(status_code), status=status_code
    )

    with pytest.raises(expected_error) as exc_info:
        services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_user_auth_info()
        )

    # pytest.raises also accepts subclasses, and some mapped classes have them
    # (e.g. MLRunNotFoundError), so pin the exact class.
    assert exc_info.type is expected_error
    _assert_upstream_details(exc_info.value)
    # Revoking is not idempotent, so a failed attempt is never retried
    _only_request(orca, "POST", REVOKE_PATH)
    secrets_provider.delete_user_token_secret.assert_not_called()


def test_delete_secret_token_revoke_not_found_reports_not_deleted(
    orca, secrets_provider, revocation_required
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json(
        _error_body(404), status=404
    )

    response = services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    assert response.deleted is False
    _only_request(orca, "POST", REVOKE_PATH)
    secrets_provider.delete_user_token_secret.assert_not_called()


def test_delete_secret_token_revoke_unreachable(
    unreachable_orca, secrets_provider, revocation_required
):
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_user_auth_info()
        )

    secrets_provider.delete_user_token_secret.assert_not_called()


def test_delete_secret_token_revoke_propagates_context_id(
    orca, secrets_provider, revocation_required, context_id
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    request = _only_request(orca, "POST", REVOKE_PATH)
    assert request.headers.get(mlrun.common.schemas.HeaderNames.igz_ctx) == context_id


# --- User lookup: GET profile/users/{username}, forwarded user identity ---


@pytest.fixture
def revocation_owned_by_platform():
    mlrun.mlconf.oris_version = "1.2.0"


def test_delete_secret_token_for_other_user_resolves_user_id(
    orca, secrets_provider, revocation_owned_by_platform
):
    orca.expect_request("/api/v1/profile/users/jane", method="GET").respond_with_json(
        _user_response(user_id="jane-id", username="jane")
    )

    response = services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_admin_auth_info()
    )

    request = _only_request(orca, "GET", "/api/v1/profile/users/jane")
    _assert_forwarded_identity(request, ADMIN_BEARER)
    assert request.query_string == b""
    assert request.get_data() == b""
    secrets_provider.delete_user_token_secret.assert_called_once_with(
        user_id="jane-id", token_name="my-token"
    )
    assert response.deleted is True
    assert response.username == "jane"


def test_delete_secret_token_for_self_case_insensitive_skips_user_lookup(
    orca, secrets_provider, revocation_owned_by_platform
):
    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="ADMIN", auth_info=_admin_auth_info()
    )

    assert orca.log == []
    secrets_provider.delete_user_token_secret.assert_called_once_with(
        user_id="admin-id", token_name="my-token"
    )


@pytest.mark.parametrize(
    "status_code, expected_error",
    [
        (401, mlrun.errors.MLRunUnauthorizedError),
        (403, mlrun.errors.MLRunAccessDeniedError),
        (404, mlrun.errors.MLRunNotFoundError),
        (500, mlrun.errors.MLRunInternalServerError),
    ]
    + [(status, mlrun.errors.MLRunUnauthorizedError) for status in UNMAPPED_STATUSES],
)
def test_delete_secret_token_user_lookup_failure(
    orca, secrets_provider, revocation_owned_by_platform, status_code, expected_error
):
    orca.expect_request("/api/v1/profile/users/jane", method="GET").respond_with_json(
        _error_body(status_code), status=status_code
    )

    with pytest.raises(expected_error) as exc_info:
        services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_admin_auth_info()
        )

    # pytest.raises also accepts subclasses, and some mapped classes have them
    # (e.g. MLRunNotFoundError), so pin the exact class.
    assert exc_info.type is expected_error
    _assert_upstream_details(exc_info.value)
    _only_request(orca, "GET", "/api/v1/profile/users/jane")
    secrets_provider.delete_user_token_secret.assert_not_called()


def test_delete_secret_token_user_lookup_unreachable(
    unreachable_orca, secrets_provider, revocation_owned_by_platform
):
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_admin_auth_info()
        )

    secrets_provider.delete_user_token_secret.assert_not_called()


def test_user_lookup_propagates_context_id(
    orca, secrets_provider, revocation_owned_by_platform, context_id
):
    orca.expect_request("/api/v1/profile/users/jane", method="GET").respond_with_json(
        _user_response(user_id="jane-id", username="jane")
    )

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_admin_auth_info()
    )

    request = _only_request(orca, "GET", "/api/v1/profile/users/jane")
    assert request.headers.get(mlrun.common.schemas.HeaderNames.igz_ctx) == context_id


# --- Platform version: GET info/system, service-account identity ---


@pytest.fixture
def unknown_platform_version():
    mlrun.mlconf.oris_version = ""


def test_revocation_gate_fetches_version_with_service_account(
    orca, secrets_provider, unknown_platform_version
):
    orca.expect_request(SYSTEM_INFO_PATH, method="GET").respond_with_json(
        _system_info_response("1.0.0")
    )
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    assert [(request.method, request.path) for request, _ in orca.log] == [
        ("GET", SYSTEM_INFO_PATH),
        ("POST", REVOKE_PATH),
    ]
    version_request, revoke_request = (request for request, _ in orca.log)
    _assert_service_account_identity(version_request)
    assert version_request.query_string == b""
    assert version_request.get_data() == b""
    _assert_forwarded_identity(revoke_request, USER_BEARER)
    assert mlrun.mlconf.oris_version == "1.0.0"


def test_revocation_gate_caches_version(
    orca, secrets_provider, unknown_platform_version
):
    orca.expect_request(SYSTEM_INFO_PATH, method="GET").respond_with_json(
        _system_info_response("1.0.0")
    )
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    for _ in range(2):
        services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_user_auth_info()
        )

    assert len(_requests(orca, "GET", SYSTEM_INFO_PATH)) == 1
    assert len(_requests(orca, "POST", REVOKE_PATH)) == 2


@pytest.mark.parametrize(
    "platform_version", ["1.2.0", "1.2.0-pr-921.1.20260818115106", "999.0.0"]
)
def test_revocation_gate_recent_version_skips_revoke(
    orca, secrets_provider, unknown_platform_version, platform_version
):
    orca.expect_request(SYSTEM_INFO_PATH, method="GET").respond_with_json(
        _system_info_response(platform_version)
    )

    response = services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    _only_request(orca, "GET", SYSTEM_INFO_PATH)
    secrets_provider.delete_user_token_secret.assert_called_once()
    assert response.deleted is True


@pytest.mark.parametrize("status_code", [401, 404, 500])
def test_revocation_gate_version_failure_skips_revoke_and_retries(
    orca, secrets_provider, unknown_platform_version, status_code
):
    orca.expect_request(SYSTEM_INFO_PATH, method="GET").respond_with_json(
        _error_body(status_code), status=status_code
    )

    for _ in range(2):
        response = services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_user_auth_info()
        )
        assert response.deleted is True

    # Failures are not cached, so each deletion asks again; neither revokes
    assert [(request.method, request.path) for request, _ in orca.log] == [
        ("GET", SYSTEM_INFO_PATH),
        ("GET", SYSTEM_INFO_PATH),
    ]
    assert mlrun.mlconf.oris_version == ""
    assert secrets_provider.delete_user_token_secret.call_count == 2


def test_revocation_gate_version_unreachable_skips_revoke(
    unreachable_orca, secrets_provider, unknown_platform_version
):
    response = services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    assert response.deleted is True
    assert mlrun.mlconf.oris_version == ""
    secrets_provider.get_user_token_secret_value.assert_not_called()
    secrets_provider.delete_user_token_secret.assert_called_once()


def test_revocation_gate_preset_version_skips_fetch(
    orca, secrets_provider, revocation_required
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    _only_request(orca, "POST", REVOKE_PATH)


def test_revocation_gate_propagates_context_id(
    orca, secrets_provider, unknown_platform_version, context_id
):
    orca.expect_request(SYSTEM_INFO_PATH, method="GET").respond_with_json(
        _system_info_response("1.2.0")
    )

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    request = _only_request(orca, "GET", SYSTEM_INFO_PATH)
    assert request.headers.get(mlrun.common.schemas.HeaderNames.igz_ctx) == context_id


# --- Token-file resolution: POST refresh-access-token, no identity ---


@pytest.mark.parametrize(
    "stored, provided_token_name, expected_name, expected_value",
    [
        ([("my-token", "T-MY"), ("other", "T-O")], "my-token", "my-token", "T-MY"),
        ([("other", "T-O"), ("default", "T-D")], None, "default", "T-D"),
        ([("first", "T-1"), ("second", "T-2")], None, "first", "T-1"),
    ],
    ids=["provided-name", "prefers-default", "falls-back-to-first"],
)
def test_resolve_auth_token_name_validates_selected_token(
    orca, user_tokens, stored, provided_token_name, expected_name, expected_value
):
    user_tokens.extend(stored)
    orca.expect_request(REFRESH_TOKEN_PATH, method="POST").respond_with_json(
        _refresh_token_response()
    )

    resolved_name = services.api.utils.helpers.resolve_auth_token_name(
        user_id="jane-id", provided_token_name=provided_token_name
    )

    assert resolved_name == expected_name
    request = _only_request(orca, "POST", REFRESH_TOKEN_PATH)
    assert _json_body(request) == {"refreshToken": expected_value}
    assert "Authorization" not in request.headers
    assert (
        mlrun.common.schemas.HeaderNames.igz_authenticator_kind not in request.headers
    )


@pytest.mark.parametrize("provided_token_name", [None, "first"])
def test_resolve_auth_token_name_rejected_token(orca, user_tokens, provided_token_name):
    user_tokens.extend([("first", "T-1"), ("second", "T-2")])
    orca.expect_request(REFRESH_TOKEN_PATH, method="POST").respond_with_json(
        _error_body(401), status=401
    )

    with pytest.raises(
        mlrun.errors.MLRunNotFoundError, match="^No valid tokens found for user$"
    ):
        services.api.utils.helpers.resolve_auth_token_name(
            user_id="jane-id", provided_token_name=provided_token_name
        )

    # Only the selected token is validated; the next one is never tried
    _only_request(orca, "POST", REFRESH_TOKEN_PATH)


def test_resolve_auth_token_name_duplicate_defaults(orca, user_tokens):
    user_tokens.extend([("default", "T-A"), ("default", "T-B")])

    with pytest.raises(
        mlrun.errors.MLRunNotFoundError, match="^No valid tokens found for user$"
    ):
        services.api.utils.helpers.resolve_auth_token_name(
            user_id="jane-id", provided_token_name=None
        )

    assert orca.log == []


def test_resolve_auth_token_name_provided_token_empty(orca, user_tokens):
    user_tokens.append(("my-token", ""))

    with pytest.raises(
        mlrun.errors.MLRunNotFoundError,
        match="^Token 'my-token' not found or invalid for user$",
    ):
        services.api.utils.helpers.resolve_auth_token_name(
            user_id="jane-id", provided_token_name="my-token"
        )

    assert orca.log == []


@pytest.mark.parametrize("provided_token_name", [None, "first"])
@pytest.mark.parametrize("status_code", [500, 503])
def test_resolve_auth_token_name_platform_error_is_not_found(
    orca, user_tokens, provided_token_name, status_code
):
    user_tokens.append(("first", "T-1"))
    orca.expect_request(REFRESH_TOKEN_PATH, method="POST").respond_with_json(
        _error_body(status_code), status=status_code
    )

    with pytest.raises(
        mlrun.errors.MLRunNotFoundError, match="^No valid tokens found for user$"
    ):
        services.api.utils.helpers.resolve_auth_token_name(
            user_id="jane-id", provided_token_name=provided_token_name
        )


@pytest.mark.parametrize("provided_token_name", [None, "first"])
def test_resolve_auth_token_name_platform_unreachable_is_not_found(
    unreachable_orca, user_tokens, provided_token_name
):
    user_tokens.append(("first", "T-1"))

    with pytest.raises(
        mlrun.errors.MLRunNotFoundError, match="^No valid tokens found for user$"
    ):
        services.api.utils.helpers.resolve_auth_token_name(
            user_id="jane-id", provided_token_name=provided_token_name
        )


@pytest.mark.parametrize("provided_token_name", [None, "missing"])
def test_resolve_auth_token_name_no_stored_token(
    orca, user_tokens, provided_token_name
):
    with pytest.raises(mlrun.errors.MLRunBadRequestError):
        services.api.utils.helpers.resolve_auth_token_name(
            user_id="jane-id", provided_token_name=provided_token_name
        )

    assert orca.log == []


def _user_auth_info() -> mlrun.common.schemas.AuthInfo:
    return mlrun.common.schemas.AuthInfo(
        username="jane",
        user_id="jane-id",
        request_headers={"authorization": USER_BEARER},
    )


def _admin_auth_info() -> mlrun.common.schemas.AuthInfo:
    return mlrun.common.schemas.AuthInfo(
        username="admin",
        user_id="admin-id",
        request_headers={"authorization": ADMIN_BEARER},
    )


def _requests(
    server: pytest_httpserver.HTTPServer, method: str, path: str
) -> list[werkzeug.Request]:
    return [
        request
        for request, _ in server.log
        if request.method == method and request.path == path
    ]


def _only_request(
    server: pytest_httpserver.HTTPServer, method: str, path: str
) -> werkzeug.Request:
    all_requests = [(request.method, request.path) for request, _ in server.log]
    # Compare against every received request, not just the count, so a failure shows
    # which extra or unexpected requests Orca got.
    assert all_requests == [(method, path)]
    return server.log[0][0]


def _json_body(request: werkzeug.Request) -> dict:
    assert request.mimetype == "application/json"
    return request.get_json()


def _assert_service_account_identity(request: werkzeug.Request) -> None:
    assert request.headers.get("Authorization") == f"Bearer {SA_TOKEN}"
    assert (
        request.headers.get(mlrun.common.schemas.HeaderNames.igz_authenticator_kind)
        == "sa"
    )
    _assert_absent_from_headers(request, USER_BEARER, ADMIN_BEARER)


def _assert_forwarded_identity(request: werkzeug.Request, bearer: str) -> None:
    assert request.headers.get("Authorization") == bearer
    assert (
        mlrun.common.schemas.HeaderNames.igz_authenticator_kind not in request.headers
    )
    _assert_absent_from_headers(request, SA_TOKEN)


def _assert_absent_from_headers(request: werkzeug.Request, *secrets: str) -> None:
    for name, value in request.headers.items():
        for secret in secrets:
            assert secret not in value, f"{secret!r} leaked in header {name!r}"


def _assert_upstream_details(exc: Exception) -> None:
    assert UPSTREAM_MESSAGE in str(exc)
    assert UPSTREAM_CTX in str(exc)


def _error_body(status_code: int) -> dict:
    return {
        "status": {
            "statusCode": status_code,
            "errorMessage": UPSTREAM_MESSAGE,
            "ctx": UPSTREAM_CTX,
        }
    }


def _jwt(payload: dict) -> str:
    return jwt.encode(payload, key="dummy", algorithm="HS256")


def _offline_token(subject: str) -> str:
    now = int(time.time())
    return _jwt({"sub": subject, "exp": now + 3600, "iat": now})


def _access_token() -> str:
    now = int(time.time())
    return _jwt({"exp": now + 300, "iat": now})


def _token_metadata() -> dict:
    return {
        "sessionId": "session-id",
        "tokenType": "Bearer",
        "issuedTokenType": "urn:ietf:params:oauth:token-type:refresh_token",
        "scope": "openid offline_access",
    }


def _refresh_token_response() -> dict:
    return {
        "metadata": _token_metadata(),
        "spec": {
            "accessToken": _access_token(),
            "idToken": _access_token(),
            "refreshToken": "rotated",
        },
        "status": {"expiresIn": 300},
    }


def _refresh_tokens_response(*refresh_tokens: str) -> dict:
    return {
        "items": [
            {
                "metadata": _token_metadata(),
                "spec": {
                    "accessToken": _access_token(),
                    "idToken": _access_token(),
                    "refreshToken": refresh_token,
                },
                "status": {
                    "expiresIn": 300,
                    "refreshExpiresIn": 0,
                    "notBeforePolicy": 0,
                    "statusCode": 200,
                },
            }
            for refresh_token in refresh_tokens
        ],
        "status": {"total": len(refresh_tokens), "hasMore": False},
    }


def _user_response(user_id: str, username: str) -> dict:
    return {
        "metadata": {"id": user_id, "username": username, "resourceType": "user"},
        "spec": {},
        "status": {"ctx": "ctx", "statusCode": 200},
    }


def _system_info_response(platform_version: str) -> dict:
    return {
        "metadata": {
            "domain": "example.com",
            "namespace": "default-tenant",
            "systemId": "system-id",
            "version": {
                "orisVersion": platform_version,
                "ecliptosVersion": "0.3.0",
                "buildDate": "2026-10-01",
                "branch": "main",
                "commit": "abc123",
            },
            "keycloak": {
                "hostAddress": "https://keycloak.example.com",
                "realm": "platform",
            },
            "registry": {
                "user": "registry.example.com/user",
                "system": "registry.example.com/system",
            },
        },
        "spec": {},
        "status": {"services": {}, "upgradeStatus": "idle"},
    }
