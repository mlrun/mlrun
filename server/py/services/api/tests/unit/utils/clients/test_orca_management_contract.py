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
Wire-level contract of the token, user, version and project-policy calls MLRun makes to Orca.

A local HTTP server stands in for Orca, so the tests apply to whichever HTTP client MLRun
uses to reach it. Most call the callers (``crud.Secrets``, ``utils.helpers``); token-file shapes the
helper never builds, calls without a caller's request, and the igz follower's project-policy calls go
to the client directly. Each asserts what Orca receives (method, path, body, whose identity) and what
MLRun raises.
"""

import asyncio
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

import framework.utils.clients.iguazio.v4
import framework.utils.clients.service_account_token
import framework.utils.singletons.k8s
import services.api.crud
import services.api.utils.helpers

SA_TOKEN = "sa-token"
USER_BEARER = "Bearer user-jwt"
ADMIN_BEARER = "Bearer admin-jwt"
COOKIE_SESSION = "cookie-session"
CTX_ID = "test-context-id"
OFFLINE_TOKEN = "offline-xyz"
UPSTREAM_MESSAGE = "upstream-message"
UPSTREAM_CTX = "upstream-ctx"

REFRESH_TOKENS_PATH = "/api/v1/authentication/refresh-access-tokens"
REFRESH_TOKEN_PATH = "/api/v1/authentication/refresh-access-token"
REVOKE_PATH = "/api/v1/authentication/revoke-offline-token"
SYSTEM_INFO_PATH = "/api/v1/info/system"

PROJECT = "my-project"
PROJECT_OWNER = "new-owner"
PROJECT_POLICIES_PATH = "/api/v1/authorization/projects/policies"
PROJECT_OWNER_PATH = f"/api/v1/authorization/projects/{PROJECT}/owner"
PROJECT_POLICIES_DELETE_PATH = f"/api/v1/authorization/projects/{PROJECT}/policies"

# The igz follower's calls that relay the caller's identity:
# (operation, method, path, body, success status, failure message)
FORWARDED_POLICY_CALLS = [
    pytest.param(
        "create",
        "POST",
        PROJECT_POLICIES_PATH,
        {"project": PROJECT},
        201,
        "Failed to create default project policies in Iguazio",
        id="create",
    ),
    pytest.param(
        "store",
        "POST",
        PROJECT_POLICIES_PATH,
        {"project": PROJECT},
        201,
        "Failed to store project policies in Iguazio",
        id="store",
    ),
    pytest.param(
        "patch",
        "PUT",
        PROJECT_OWNER_PATH,
        {"owner": PROJECT_OWNER},
        200,
        "Failed to update project owner in Iguazio",
        id="patch",
    ),
]

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


def test_store_secret_tokens_any_success_status_stores_every_token(
    orca, secrets_provider
):
    first, second = _offline_token("jane-id"), _offline_token("jane-id")
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_json(
        {}, status=202
    )

    response = services.api.crud.Secrets().store_secret_tokens(
        secret_tokens=[
            mlrun.common.schemas.SecretToken(name="first", token=first),
            mlrun.common.schemas.SecretToken(name="second", token=second),
        ],
        auth_info=_user_auth_info(),
    )

    assert secrets_provider.store_user_token_secret.call_count == 2
    assert response.created_tokens == ["first", "second"]


def test_response_cookies_are_not_sent_on_later_calls(
    orca, secrets_provider, revocation_required
):
    stored_token = _offline_token("jane-id")
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_json(
        _refresh_tokens_response(stored_token),
        headers={"Set-Cookie": "session=service-account-session; Path=/"},
    )
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    services.api.crud.Secrets().store_secret_tokens(
        secret_tokens=[
            mlrun.common.schemas.SecretToken(name="first", token=stored_token)
        ],
        auth_info=_user_auth_info(),
    )
    services.api.crud.Secrets().delete_secret_token(
        token_name="first", username="jane", auth_info=_user_auth_info()
    )

    (revoke_request,) = _requests(orca, "POST", REVOKE_PATH)
    assert "Cookie" not in revoke_request.headers
    _assert_forwarded_identity(revoke_request, USER_BEARER)


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


def test_delete_secret_token_revoke_any_success_status_deletes(
    orca, secrets_provider, revocation_required
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_data("", status=202)

    response = services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    assert response.deleted is True
    secrets_provider.delete_user_token_secret.assert_called_once()


def test_delete_secret_token_revoke_with_cookie_only_forwards_only_the_cookie(
    orca, secrets_provider, revocation_required
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_cookie_auth_info()
    )

    request = _only_request(orca, "POST", REVOKE_PATH)
    _assert_cookie_only_identity(request)


def test_revoke_without_request_headers_sends_no_identity(orca):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json(
        _error_body(401), status=401
    )

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        framework.utils.clients.iguazio.v4.Client().revoke_offline_token(
            OFFLINE_TOKEN, request_headers=None
        )

    request = _only_request(orca, "POST", REVOKE_PATH)
    assert "Authorization" not in request.headers
    assert (
        mlrun.common.schemas.HeaderNames.igz_authenticator_kind not in request.headers
    )
    _assert_absent_from_headers(request, SA_TOKEN)


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


def test_user_lookup_with_cookie_only_forwards_only_the_cookie(
    orca, secrets_provider, revocation_owned_by_platform
):
    orca.expect_request("/api/v1/profile/users/jane", method="GET").respond_with_json(
        _user_response(user_id="jane-id", username="jane")
    )
    auth_info = _cookie_auth_info()
    auth_info.username = "admin"

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=auth_info
    )

    request = _only_request(orca, "GET", "/api/v1/profile/users/jane")
    _assert_cookie_only_identity(request)


def test_user_lookup_reads_only_the_user_id(
    orca, secrets_provider, revocation_owned_by_platform
):
    orca.expect_request("/api/v1/profile/users/jane", method="GET").respond_with_json(
        {"metadata": {"id": "jane-id"}}
    )

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_admin_auth_info()
    )

    secrets_provider.delete_user_token_secret.assert_called_once_with(
        user_id="jane-id", token_name="my-token"
    )


@pytest.mark.parametrize(
    "response_body",
    ["{}", '{"metadata": {}}', '{"metadata": {"id": ""}}', "not-json"],
    ids=["empty", "no-id", "empty-id", "not-json"],
)
def test_user_lookup_without_user_id_fails(
    orca, secrets_provider, revocation_owned_by_platform, response_body
):
    orca.expect_request("/api/v1/profile/users/jane", method="GET").respond_with_data(
        response_body, content_type="application/json"
    )

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc_info:
        services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_admin_auth_info()
        )

    assert exc_info.type is mlrun.errors.MLRunUnauthorizedError
    secrets_provider.delete_user_token_secret.assert_not_called()


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


def test_revocation_gate_reads_only_the_platform_version(
    orca, secrets_provider, unknown_platform_version
):
    orca.expect_request(SYSTEM_INFO_PATH, method="GET").respond_with_json(
        {"metadata": {"version": {"orisVersion": "1.0.0"}}}
    )
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})

    services.api.crud.Secrets().delete_secret_token(
        token_name="my-token", username="jane", auth_info=_user_auth_info()
    )

    assert len(_requests(orca, "POST", REVOKE_PATH)) == 1
    assert mlrun.mlconf.oris_version == "1.0.0"


@pytest.mark.parametrize(
    "response_body",
    [
        {},
        {"metadata": {"version": {}}},
        {"metadata": {"version": {"orisVersion": ""}}},
    ],
    ids=["empty", "no-version", "empty-version"],
)
def test_revocation_gate_missing_platform_version_is_not_cached(
    orca, secrets_provider, unknown_platform_version, response_body
):
    orca.expect_request(SYSTEM_INFO_PATH, method="GET").respond_with_json(response_body)

    for _ in range(2):
        response = services.api.crud.Secrets().delete_secret_token(
            token_name="my-token", username="jane", auth_info=_user_auth_info()
        )
        assert response.deleted is True

    assert [(request.method, request.path) for request, _ in orca.log] == [
        ("GET", SYSTEM_INFO_PATH),
        ("GET", SYSTEM_INFO_PATH),
    ]
    assert mlrun.mlconf.oris_version == ""


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


def test_resolve_auth_token_name_any_success_response_is_valid(orca, user_tokens):
    user_tokens.append(("first", "T-1"))
    orca.expect_request(REFRESH_TOKEN_PATH, method="POST").respond_with_json({})

    assert (
        services.api.utils.helpers.resolve_auth_token_name(
            user_id="jane-id", provided_token_name=None
        )
        == "first"
    )


def test_resolve_auth_token_name_propagates_context_id(orca, user_tokens, context_id):
    user_tokens.append(("first", "T-1"))
    orca.expect_request(REFRESH_TOKEN_PATH, method="POST").respond_with_json(
        _refresh_token_response()
    )

    services.api.utils.helpers.resolve_auth_token_name(
        user_id="jane-id", provided_token_name=None
    )

    request = _only_request(orca, "POST", REFRESH_TOKEN_PATH)
    assert request.headers.get(mlrun.common.schemas.HeaderNames.igz_ctx) == context_id
    assert "Authorization" not in request.headers


@pytest.mark.parametrize(
    "token_file_content, token_name, expected_message",
    [
        ("{}", None, "^No valid tokens found for user$"),
        ("{}", "x", "^Token 'x' not found or invalid for user$"),
        ("secretTokens:", None, "^No valid tokens found for user$"),
        ("secretTokens: []", "x", "^Token 'x' not found or invalid for user$"),
        (
            "secretTokens: [{name: other, token: T-O}]",
            "x",
            "^Token 'x' not found or invalid for user$",
        ),
        (
            "secretTokens: [{name: first, token: ''}, {name: second, token: T-2}]",
            None,
            "^No valid tokens found for user$",
        ),
        ("secretTokens: [{token: T-1}]", None, "^No valid tokens found for user$"),
    ],
    ids=[
        "no-tokens-key",
        "no-tokens-key-named",
        "null-tokens",
        "empty-tokens-named",
        "named-token-absent",
        "first-token-empty",
        "first-token-unnamed",
    ],
)
def test_resolve_token_from_token_file_without_usable_token(
    orca, token_file_content, token_name, expected_message
):
    with pytest.raises(mlrun.errors.MLRunNotFoundError, match=expected_message):
        framework.utils.clients.iguazio.v4.Client().resolve_token_from_igz_yml(
            token_file_content, "jane-id", token_name
        )

    assert orca.log == []


# --- Identity isolation across concurrent calls ---


async def test_concurrent_calls_carry_only_their_own_identity(
    orca, revocation_required
):
    orca.expect_request(REVOKE_PATH, method="POST").respond_with_json({})
    orca.expect_request(REFRESH_TOKENS_PATH, method="POST").respond_with_json(
        _refresh_tokens_response(OFFLINE_TOKEN)
    )
    orca.expect_request("/api/v1/profile/users/jane", method="GET").respond_with_json(
        _user_response(user_id="jane-id", username="jane")
    )
    client = framework.utils.clients.iguazio.v4.AsyncClient()
    iterations = 5

    for _ in range(iterations):
        await asyncio.gather(
            client.revoke_offline_token("user-revoked", {"authorization": USER_BEARER}),
            client.revoke_offline_token(
                "cookie-revoked", _cookie_auth_info().request_headers
            ),
            client.get_user_id_by_username("jane", _admin_auth_info()),
            client.refresh_access_tokens(
                [mlrun.common.schemas.SecretToken(name="t", token=OFFLINE_TOKEN)]
            ),
        )

    revoke_requests = _requests(orca, "POST", REVOKE_PATH)
    assert len(revoke_requests) == 2 * iterations
    for request in revoke_requests:
        if _json_body(request)["token"] == "user-revoked":
            _assert_forwarded_identity(request, USER_BEARER)
            _assert_absent_from_headers(request, ADMIN_BEARER)
        else:
            _assert_cookie_only_identity(request)
    lookup_requests = _requests(orca, "GET", "/api/v1/profile/users/jane")
    assert len(lookup_requests) == iterations
    for request in lookup_requests:
        _assert_forwarded_identity(request, ADMIN_BEARER)
        _assert_absent_from_headers(request, USER_BEARER)
    refresh_requests = _requests(orca, "POST", REFRESH_TOKENS_PATH)
    assert len(refresh_requests) == iterations
    for request in refresh_requests:
        _assert_service_account_identity(request)


# --- Project policies (igz follower): create/store/patch as the caller, delete as the service account ---


@pytest.mark.parametrize(
    "operation, method, path, body, success_status, failure_message",
    FORWARDED_POLICY_CALLS,
)
def test_project_policies_forward_caller_identity(
    orca, operation, method, path, body, success_status, failure_message
):
    orca.expect_request(path, method=method).respond_with_json(
        {}, status=success_status
    )

    _call_project_policies(operation, _user_auth_info())

    request = _only_request(orca, method, path)
    _assert_forwarded_identity(request, USER_BEARER)
    assert _json_body(request) == body


@pytest.mark.parametrize(
    "operation, method, path, body, success_status, failure_message",
    FORWARDED_POLICY_CALLS,
)
@pytest.mark.parametrize(
    "status_code, expected_error",
    MAPPED_ERRORS
    + [(status, mlrun.errors.MLRunInternalServerError) for status in UNMAPPED_STATUSES],
)
def test_project_policies_failure(
    orca,
    operation,
    method,
    path,
    body,
    success_status,
    failure_message,
    status_code,
    expected_error,
):
    if operation == "store" and status_code == 409:
        pytest.skip("A conflict on store means the policies already exist")
    orca.expect_request(path, method=method).respond_with_json(
        _error_body(status_code), status=status_code
    )

    with pytest.raises(expected_error) as exc_info:
        _call_project_policies(operation, _user_auth_info())

    assert exc_info.type is expected_error
    assert failure_message in str(exc_info.value)
    _assert_upstream_details(exc_info.value)
    _only_request(orca, method, path)


@pytest.mark.parametrize(
    "operation, method, path, body, success_status, failure_message",
    FORWARDED_POLICY_CALLS,
)
def test_project_policies_with_cookie_only_forward_only_the_cookie(
    orca, operation, method, path, body, success_status, failure_message
):
    orca.expect_request(path, method=method).respond_with_json(
        {}, status=success_status
    )

    _call_project_policies(operation, _cookie_auth_info())

    request = _only_request(orca, method, path)
    _assert_cookie_only_identity(request)


@pytest.mark.parametrize(
    "operation, method, path, body, success_status, failure_message",
    FORWARDED_POLICY_CALLS,
)
def test_project_policies_without_request_headers_send_no_identity(
    orca, operation, method, path, body, success_status, failure_message
):
    orca.expect_request(path, method=method).respond_with_json(
        _error_body(401), status=401
    )

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        _call_project_policies(operation)

    request = _only_request(orca, method, path)
    assert "Authorization" not in request.headers
    assert (
        mlrun.common.schemas.HeaderNames.igz_authenticator_kind not in request.headers
    )
    _assert_absent_from_headers(request, SA_TOKEN)


@pytest.mark.parametrize(
    "operation, method, path, success_status",
    [
        ("create", "POST", PROJECT_POLICIES_PATH, 200),
        ("store", "POST", PROJECT_POLICIES_PATH, 202),
        ("patch", "PUT", PROJECT_OWNER_PATH, 204),
        ("delete", "DELETE", PROJECT_POLICIES_DELETE_PATH, 204),
    ],
)
def test_project_policies_any_success_status_succeeds(
    orca, operation, method, path, success_status
):
    orca.expect_request(path, method=method).respond_with_data(
        "", status=success_status
    )

    _call_project_policies(operation, _user_auth_info())

    _only_request(orca, method, path)


def test_store_project_with_existing_policies_succeeds(orca):
    orca.expect_request(PROJECT_POLICIES_PATH, method="POST").respond_with_json(
        _error_body(409), status=409
    )

    _call_project_policies("store", _user_auth_info())

    _only_request(orca, "POST", PROJECT_POLICIES_PATH)


def test_patch_project_without_owner_sends_no_request(orca):
    framework.utils.clients.iguazio.v4.FollowerClient().patch_project(
        None,
        PROJECT,
        {"metadata": {"name": PROJECT}, "spec": {}},
        auth_info=_user_auth_info(),
    )

    assert orca.log == []


@pytest.mark.parametrize(
    "deletion_strategy",
    [
        mlrun.common.schemas.DeletionStrategy.restricted,
        mlrun.common.schemas.DeletionStrategy.cascading,
    ],
)
def test_delete_project_policies_uses_service_account(orca, deletion_strategy):
    orca.expect_request(
        PROJECT_POLICIES_DELETE_PATH, method="DELETE"
    ).respond_with_json({})

    framework.utils.clients.iguazio.v4.FollowerClient().delete_project(
        None, PROJECT, deletion_strategy=deletion_strategy, auth_info=_user_auth_info()
    )

    request = _only_request(orca, "DELETE", PROJECT_POLICIES_DELETE_PATH)
    _assert_service_account_identity(request)
    assert request.get_data() == b""


def test_delete_project_check_strategy_sends_no_request(orca):
    framework.utils.clients.iguazio.v4.FollowerClient().delete_project(
        None,
        PROJECT,
        deletion_strategy=mlrun.common.schemas.DeletionStrategy.check,
        auth_info=_user_auth_info(),
    )

    assert orca.log == []


@pytest.mark.parametrize(
    "status_code, expected_error",
    MAPPED_ERRORS
    + [(status, mlrun.errors.MLRunInternalServerError) for status in UNMAPPED_STATUSES],
)
def test_delete_project_policies_failure(orca, status_code, expected_error):
    orca.expect_request(
        PROJECT_POLICIES_DELETE_PATH, method="DELETE"
    ).respond_with_json(_error_body(status_code), status=status_code)

    with pytest.raises(expected_error) as exc_info:
        _call_project_policies("delete")

    assert exc_info.type is expected_error
    assert "Failed to delete project policies in Iguazio" in str(exc_info.value)
    _assert_upstream_details(exc_info.value)
    _only_request(orca, "DELETE", PROJECT_POLICIES_DELETE_PATH)


@pytest.mark.parametrize(
    "operation, method, path, success_status",
    [
        ("create", "POST", PROJECT_POLICIES_PATH, 201),
        ("store", "POST", PROJECT_POLICIES_PATH, 201),
        ("patch", "PUT", PROJECT_OWNER_PATH, 200),
        ("delete", "DELETE", PROJECT_POLICIES_DELETE_PATH, 200),
    ],
)
def test_project_policies_propagate_context_id(
    orca, context_id, operation, method, path, success_status
):
    orca.expect_request(path, method=method).respond_with_json(
        {}, status=success_status
    )

    _call_project_policies(operation, _user_auth_info())

    request = _only_request(orca, method, path)
    assert request.headers.get(mlrun.common.schemas.HeaderNames.igz_ctx) == context_id


@pytest.mark.parametrize("operation", ["create", "store", "patch", "delete"])
def test_project_policies_unreachable(unreachable_orca, operation):
    with pytest.raises(mlrun.errors.MLRunInternalServerError) as exc_info:
        _call_project_policies(operation, _user_auth_info())

    assert exc_info.type is mlrun.errors.MLRunInternalServerError


def _call_project_policies(
    operation: str, auth_info: mlrun.common.schemas.AuthInfo | None = None
) -> None:
    follower = framework.utils.clients.iguazio.v4.FollowerClient()
    auth_info = auth_info or mlrun.common.schemas.AuthInfo()
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=PROJECT),
        spec=mlrun.common.schemas.ProjectSpec(owner=PROJECT_OWNER),
    )
    if operation == "create":
        follower.create_project(None, project, auth_info=auth_info)
    elif operation == "store":
        follower.store_project(None, PROJECT, project, auth_info=auth_info)
    elif operation == "patch":
        follower.patch_project(
            None,
            PROJECT,
            {"metadata": {"name": PROJECT}, "spec": {"owner": PROJECT_OWNER}},
            auth_info=auth_info,
        )
    else:
        follower.delete_project(None, PROJECT, auth_info=auth_info)


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


def _cookie_auth_info() -> mlrun.common.schemas.AuthInfo:
    return mlrun.common.schemas.AuthInfo(
        username="jane",
        user_id="jane-id",
        request_headers={"cookie": f"_oauth2_proxy={COOKIE_SESSION}"},
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


def _assert_cookie_only_identity(request: werkzeug.Request) -> None:
    # A caller authenticated by cookie alone reaches Orca with that cookie and nothing added
    assert request.headers.get("Cookie") == f"_oauth2_proxy={COOKIE_SESSION}"
    assert "Authorization" not in request.headers
    assert (
        mlrun.common.schemas.HeaderNames.igz_authenticator_kind not in request.headers
    )
    _assert_absent_from_headers(request, SA_TOKEN, USER_BEARER, ADMIN_BEARER)


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
