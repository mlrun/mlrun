# Copyright 2023 Iguazio
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

import deepdiff
import mlrun_pipelines
import pytest

import mlrun


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
    function.apply(mlrun_pipelines.iguazio.mount_v3io())
    os.environ["V3IO_USERNAME"] = username_2
    os.environ["V3IO_ACCESS_KEY"] = access_key_2
    function.apply(mlrun_pipelines.iguazio.mount_v3io())

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


def test_mount_v3io():
    username = "username"
    access_key = "access-key"
    cases = [
        {
            "set_user": True,
            "expected_volume": {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {
                        "accessKey": access_key,
                        "dirsToCreate": f'[{{"name": "users//{username}", "permissions": 488}}]',
                    },
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
            "volume_mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path")
            ],
            "remote": "~/custom-remote",
            "expect_failure": True,
        },
        {
            "volume_mounts": [
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
                        "dirsToCreate": f'[{{"name": "users//{username}", "permissions": 488}}]',
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
            "volume_mounts": [
                mlrun.VolumeMount("/volume-mount-path", "volume-sub-path"),
                mlrun.VolumeMount("/volume-mount-path-2", "volume-sub-path-2"),
            ],
            "set_user": True,
            "expected_volume": {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {
                        "accessKey": access_key,
                        "dirsToCreate": f'[{{"name": "users//{username}", "permissions": 488}}]',
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
        mount_v3io_kwargs = {
            "remote": case.get("remote"),
            "volume_mounts": case.get("volume_mounts"),
        }
        mount_v3io_kwargs = {k: v for k, v in mount_v3io_kwargs.items() if v}

        if case.get("expect_failure"):
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                function.apply(mlrun_pipelines.iguazio.mount_v3io(**mount_v3io_kwargs))
        else:
            function.apply(mlrun_pipelines.iguazio.mount_v3io(**mount_v3io_kwargs))

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
