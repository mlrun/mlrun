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

import unittest.mock

import pytest

import mlrun
import mlrun.common.schemas as schemas
import mlrun.errors

import framework.api.deps as deps
import framework.utils.auth.verifier


@pytest.fixture
def leader_identity(monkeypatch: pytest.MonkeyPatch) -> str:
    identity = "system:serviceaccount:orca:project-leader"
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.projects, "follower_leader_identity", identity
    )
    return identity


def _mock_authenticate_request(
    monkeypatch: pytest.MonkeyPatch, auth_info: schemas.AuthInfo
) -> None:
    async def _fake_authenticate_request(self, request):
        return auth_info

    monkeypatch.setattr(
        framework.utils.auth.verifier.AuthVerifier,
        "authenticate_request",
        _fake_authenticate_request,
    )


@pytest.mark.asyncio
async def test_authenticate_leader_request_accepts_matching_service_account(
    leader_identity: str,
    monkeypatch: pytest.MonkeyPatch,
):
    auth_info = schemas.AuthInfo(
        username=leader_identity, kind=schemas.AuthInfoKind.service_account
    )
    _mock_authenticate_request(monkeypatch, auth_info)

    result = await deps.authenticate_leader_request(unittest.mock.MagicMock())

    assert result is auth_info


@pytest.mark.asyncio
async def test_authenticate_leader_request_rejects_user_caller(
    leader_identity: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """A regular user token — even one that happens to share the configured leader
    identity string as a username — must never be accepted: only a service account
    may call this surface."""
    auth_info = schemas.AuthInfo(
        username=leader_identity, kind=schemas.AuthInfoKind.user
    )
    _mock_authenticate_request(monkeypatch, auth_info)

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        await deps.authenticate_leader_request(unittest.mock.MagicMock())


@pytest.mark.asyncio
async def test_authenticate_leader_request_rejects_wrong_service_account_identity(
    leader_identity: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """A service account that isn't the configured leader must be rejected — this is
    what stops any other SA (or a follower's own SA) from driving this surface."""
    auth_info = schemas.AuthInfo(
        username="system:serviceaccount:some-other-namespace:some-other-sa",
        kind=schemas.AuthInfoKind.service_account,
    )
    _mock_authenticate_request(monkeypatch, auth_info)

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        await deps.authenticate_leader_request(unittest.mock.MagicMock())
