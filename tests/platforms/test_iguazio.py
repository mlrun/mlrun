# Copyright 2018 Iguazio
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
#
import os
from http import HTTPStatus
from unittest.mock import Mock

import deepdiff
import pytest
import requests

import mlrun
import mlrun.errors
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


def test_add_or_refresh_credentials_kubernetes_svc_url_success(monkeypatch):
    access_key = "access_key"
    api_url = "http://mlrun-api:8080"
    env = os.environ
    env["V3IO_ACCESS_KEY"] = access_key

    _, _, result_access_key = add_or_refresh_credentials(api_url)
    assert access_key == result_access_key


def test_mount_v3io_legacy():
    username = "username"
    access_key = "access-key"
    os.environ["V3IO_USERNAME"] = username
    os.environ["V3IO_ACCESS_KEY"] = access_key
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(mlrun.mount_v3io_legacy())
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
        deepdiff.DeepDiff(
            [expected_volume],
            function.spec.volumes,
            ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            [expected_volume_mount],
            function.spec.volume_mounts,
            ignore_order=True,
        )
        == {}
    )


def test_mount_v3io():
    cases = [
        {
            "mount_path": "/custom-mount-path",
            "volume_mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path")
            ],
            "expect_failure": True,
        },
        {
            "mount_path": "/custom-mount-path",
            "volume_mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path")
            ],
            "remote": "~/custom-remote",
            "expect_failure": True,
        },
        {
            "mount_path": "/custom-mount-path",
            "remote": "~/custom-remote",
            "set_user": True,
        },
        {"mount_path": "/custom-mount-path", "set_user": True},
        {"remote": "~/custom-remote", "set_user": True},
        {
            "remote": "~/custom-remote",
            "volume_mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path")
            ],
            "set_user": True,
        },
        {
            "volume_mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path")
            ],
            "set_user": True,
        },
        {"set_user": True},
        {"set_user": True, "user": "some_other_user"},
    ]
    for case in cases:
        username = "username"
        v3io_api_path = "v3io_api"
        tested_function = mlrun.new_function(
            "tested-function-name",
            "function-project",
            kind=mlrun.runtimes.RuntimeKinds.job,
        )
        expectation_function = mlrun.new_function(
            "expectation-function-name",
            "function-project",
            kind=mlrun.runtimes.RuntimeKinds.job,
        )
        if case.get("set_user"):
            os.environ["V3IO_USERNAME"] = username
            os.environ["V3IO_ACCESS_KEY"] = "access-key"
            os.environ["V3IO_API"] = v3io_api_path
        else:
            os.environ.pop("V3IO_USERNAME", None)
            os.environ.pop("V3IO_ACCESS_KEY", None)
            os.environ.pop("V3IO_API", None)
        mount_v3io_kwargs = {
            "remote": case.get("remote"),
            "mount_path": case.get("mount_path"),
            "volume_mounts": case.get("volume_mounts"),
            "user": case.get("user"),
        }
        mount_v3io_kwargs = {k: v for k, v in mount_v3io_kwargs.items() if v}
        if case.get("expect_failure"):
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                tested_function.apply(mlrun.mount_v3io(**mount_v3io_kwargs))
        else:
            tested_function.apply(mlrun.mount_v3io(**mount_v3io_kwargs))
            if not case.get("volume_mounts") and case.get("remote"):
                expectation_modifier = mlrun.mount_v3io_legacy
                expectation_modifier_kwargs = {
                    "remote": case.get("remote"),
                    "mount_path": case.get("mount_path"),
                    "user": case.get("user"),
                }
            else:
                expectation_modifier = mlrun.mount_v3io_extended
                expectation_modifier_kwargs = {
                    "remote": case.get("remote"),
                    "mounts": case.get("volume_mounts"),
                    "user": case.get("user"),
                }
            expectation_modifier_kwargs = {
                k: v for k, v in expectation_modifier_kwargs.items() if v
            }
            if list(mount_v3io_kwargs.keys()) == ["mount_path"]:
                expectation_modifier_kwargs["mounts"] = [
                    mlrun.VolumeMount(path="/v3io", sub_path=""),
                    mlrun.VolumeMount(
                        path=mount_v3io_kwargs["mount_path"],
                        sub_path="users/" + username,
                    ),
                ]
            expectation_function.apply(
                expectation_modifier(**expectation_modifier_kwargs)
            )

            # Verify that env variables are set and overridden correctly
            env_variables = {
                env_variable["name"]: env_variable["value"]
                for env_variable in tested_function.spec.env
            }
            expected_user = username
            if case.get("user"):
                expected_user = case["user"]
            assert env_variables["V3IO_USERNAME"] == expected_user
            expected_api_path = mlrun.config.config.v3io_api
            if case.get("set_user"):
                expected_api_path = v3io_api_path
            assert env_variables["V3IO_API"] == expected_api_path

            assert (
                deepdiff.DeepDiff(
                    expectation_function.spec.volumes,
                    tested_function.spec.volumes,
                    ignore_order=True,
                )
                == {}
            )
            assert (
                deepdiff.DeepDiff(
                    expectation_function.spec.volume_mounts,
                    tested_function.spec.volume_mounts,
                    ignore_order=True,
                )
                == {}
            )


def test_mount_v3io_multiple_user():
    username_1 = "first-username"
    username_2 = "second-username"
    access_key_1 = "access_key_1"
    access_key_2 = "access_key_2"
    v3io_api_path = "v3io_api"
    function = mlrun.new_function(
        "function-name",
        "function-project",
        kind=mlrun.runtimes.RuntimeKinds.job,
    )
    os.environ["V3IO_API"] = v3io_api_path

    os.environ["V3IO_USERNAME"] = username_1
    os.environ["V3IO_ACCESS_KEY"] = access_key_1
    function.apply(mlrun.mount_v3io())
    os.environ["V3IO_USERNAME"] = username_2
    os.environ["V3IO_ACCESS_KEY"] = access_key_2
    function.apply(mlrun.mount_v3io())

    user_volume_mounts = list(
        filter(
            lambda volume_mount: volume_mount["mountPath"] == "/User",
            function.spec.volume_mounts,
        )
    )
    assert len(user_volume_mounts) == 1
    assert user_volume_mounts[0]["subPath"] == f"users/{username_2}"
    assert (
        function.spec.volumes[0]["flexVolume"]["options"]["accessKey"] == access_key_2
    )


def test_mount_v3io_extended():
    username = "username"
    access_key = "access-key"
    cases = [
        {
            "set_user": True,
            "expected_volume": {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {"accessKey": access_key},
                },
                "name": "v3io",
            },
            "expected_volume_mounts": [
                {"mountPath": "/User", "name": "v3io", "subPath": f"users/{username}"},
                {"mountPath": "/v3io", "name": "v3io", "subPath": ""},
            ],
        },
        {"remote": "~/custom-remote", "expect_failure": True},
        {
            "mounts": [mlrun.VolumeMount("/volume-mount-path", "volume-sub-path")],
            "remote": "~/custom-remote",
            "expect_failure": True,
        },
        {
            "mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path"),
                mlrun.VolumeMount("/volume-mount-path-2", "volume-sub-path-2"),
            ],
            "remote": "~/custom-remote",
            "set_user": True,
            "expected_volume": {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {
                        "accessKey": access_key,
                        "container": "users",
                        "subPath": f"/{username}/custom-remote",
                    },
                },
                "name": "v3io",
            },
            "expected_volume_mounts": [
                {
                    "mountPath": "/volume-mount-path",
                    "name": "v3io",
                    "subPath": "volume-sub-path",
                },
                {
                    "mountPath": "/volume-mount-path-2",
                    "name": "v3io",
                    "subPath": "volume-sub-path-2",
                },
            ],
        },
        {
            "mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path"),
                mlrun.VolumeMount("/volume-mount-path-2", "volume-sub-path-2"),
            ],
            "set_user": True,
            "expected_volume": {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {"accessKey": access_key},
                },
                "name": "v3io",
            },
            "expected_volume_mounts": [
                {
                    "mountPath": "/volume-mount-path",
                    "name": "v3io",
                    "subPath": "volume-sub-path",
                },
                {
                    "mountPath": "/volume-mount-path-2",
                    "name": "v3io",
                    "subPath": "volume-sub-path-2",
                },
            ],
        },
    ]
    for case in cases:
        if case.get("set_user"):
            os.environ["V3IO_USERNAME"] = username
            os.environ["V3IO_ACCESS_KEY"] = access_key
        else:
            os.environ.pop("V3IO_USERNAME", None)
            os.environ.pop("V3IO_ACCESS_KEY", None)

        function = mlrun.new_function(
            "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
        )
        mount_v3io_extended_kwargs = {
            "remote": case.get("remote"),
            "mounts": case.get("mounts"),
        }
        mount_v3io_extended_kwargs = {
            k: v for k, v in mount_v3io_extended_kwargs.items() if v
        }

        if case.get("expect_failure"):
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                function.apply(mlrun.mount_v3io_extended(**mount_v3io_extended_kwargs))
        else:
            function.apply(mlrun.mount_v3io_extended(**mount_v3io_extended_kwargs))

            assert (
                deepdiff.DeepDiff(
                    [case.get("expected_volume")],
                    function.spec.volumes,
                    ignore_order=True,
                )
                == {}
            )
            assert (
                deepdiff.DeepDiff(
                    case.get("expected_volume_mounts"),
                    function.spec.volume_mounts,
                    ignore_order=True,
                )
                == {}
            )


def test_is_iguazio_session_cookie():
    assert (
        mlrun.platforms.is_iguazio_session_cookie(
            "j%3A%7B%22sid%22%3A%20%22946b0749-5c40-4837-a4ac-341d295bfaf7%22%7D"
        )
        is True
    )
    assert mlrun.platforms.is_iguazio_session_cookie("dummy") is False
