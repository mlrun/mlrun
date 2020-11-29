import os
import deepdiff
from http import HTTPStatus
from unittest.mock import Mock
import mlrun

import requests

from mlrun.platforms import add_or_refresh_credentials


def test_add_or_refresh_credentials_iguazio_2_8_success(monkeypatch):
    username = "username"
    password = "password"
    control_session = "control_session"
    api_url = "https://dashboard.default-tenant.app.hedingber-28-1.iguazio-cd2.com"
    env = os.environ
    env["V3IO_USERNAME"] = username
    env["V3IO_PASSWORD"] = password

    def mock_get(*args, **kwargs):
        not_found_response_mock = Mock()
        not_found_response_mock.ok = False
        not_found_response_mock.status_code = HTTPStatus.NOT_FOUND.value
        return not_found_response_mock

    def mock_session(*args, **kwargs):
        session_mock = Mock()

        def _mock_successful_session_creation(*args, **kwargs):
            assert session_mock.auth == (username, password)
            successful_response_mock = Mock()
            successful_response_mock.ok = True
            successful_response_mock.json.return_value = {
                "data": {"id": control_session}
            }
            return successful_response_mock

        session_mock.post = _mock_successful_session_creation
        return session_mock

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "Session", mock_session)

    result_username, result_control_session, _ = add_or_refresh_credentials(api_url)
    assert username == result_username
    assert control_session == result_control_session


def test_add_or_refresh_credentials_iguazio_2_10_success(monkeypatch):
    username = "username"
    access_key = "access_key"
    api_url = "https://dashboard.default-tenant.app.hedingber-210-1.iguazio-cd2.com"
    env = os.environ
    env["V3IO_USERNAME"] = username
    env["V3IO_ACCESS_KEY"] = access_key

    def mock_get(*args, **kwargs):
        ok_response_mock = Mock()
        ok_response_mock.ok = True
        return ok_response_mock

    monkeypatch.setattr(requests, "get", mock_get)

    result_username, result_access_key, _ = add_or_refresh_credentials(api_url)
    assert username == result_username
    assert access_key == result_access_key


def test_mount_v3io():
    username = "username"
    access_key = "access-key"
    env = os.environ
    env["V3IO_USERNAME"] = username
    env["V3IO_ACCESS_KEY"] = access_key
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(mlrun.mount_v3io())
    expected_volume = {
        "flexVolume": {
            "driver": "v3io/fuse",
            "options": {
                "accessKey": access_key,
                "container": "users",
                "subPath": f"/{username}",
            },
        },
        "name": "v3io",
    }
    expected_volume_mount = {"mountPath": "/User", "name": "v3io", "subPath": ""}
    assert (
        deepdiff.DeepDiff([expected_volume], function.spec.volumes, ignore_order=True,)
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            [expected_volume_mount], function.spec.volume_mounts, ignore_order=True,
        )
        == {}
    )


def test_mount_v3io_extended():
    username = "username"
    access_key = "access-key"
    env = os.environ
    env["V3IO_USERNAME"] = username
    env["V3IO_ACCESS_KEY"] = access_key
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(mlrun.mount_v3io_extended())
    expected_volume = {
        "flexVolume": {"driver": "v3io/fuse", "options": {"accessKey": access_key,},},
        "name": "v3io",
    }
    expected_volume_mounts = [
        {"mountPath": "/User", "name": "v3io", "subPath": f"users/{username}"},
        {"mountPath": "/v3io", "name": "v3io", "subPath": ""},
    ]
    assert (
        deepdiff.DeepDiff([expected_volume], function.spec.volumes, ignore_order=True,)
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            expected_volume_mounts, function.spec.volume_mounts, ignore_order=True,
        )
        == {}
    )
