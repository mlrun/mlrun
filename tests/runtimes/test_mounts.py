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
import os

import deepdiff
import pytest

import mlrun
import mlrun.platforms
import mlrun.runtimes.mounts


def test_mount_configmap():
    expected_volume = {"configMap": {"name": "my-config-map"}, "name": "my-volume"}
    expected_volume_mount = {"mountPath": "/myConfMapPath", "name": "my-volume"}

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(
        mlrun.runtimes.mounts.mount_configmap(
            configmap_name="my-config-map",
            mount_path="/myConfMapPath",
            volume_name="my-volume",
        )
    )

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


def test_mount_hostpath():
    expected_volume = {"hostPath": {"path": "/tmp", "type": ""}, "name": "my-volume"}
    expected_volume_mount = {"mountPath": "/myHostPath", "name": "my-volume"}

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(
        mlrun.runtimes.mounts.mount_hostpath(
            host_path="/tmp", mount_path="/myHostPath", volume_name="my-volume"
        )
    )

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


def test_mount_s3():
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            aws_access_key="xx", aws_secret_key="yy", endpoint_url="a.b"
        )
    )
    env_dict = {var["name"]: var["value"] for var in function.spec.env}
    assert env_dict == {
        "AWS_ENDPOINT_URL_S3": "a.b",
        "AWS_ACCESS_KEY_ID": "xx",
        "AWS_SECRET_ACCESS_KEY": "yy",
    }

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(mlrun.runtimes.mounts.mount_s3(secret_name="s", endpoint_url="a.b"))
    env_dict = {
        var["name"]: var.get("value", var.get("valueFrom")) for var in function.spec.env
    }
    assert env_dict == {
        "AWS_ENDPOINT_URL_S3": "a.b",
        "AWS_ACCESS_KEY_ID": {
            "secretKeyRef": {"key": "AWS_ACCESS_KEY_ID", "name": "s"}
        },
        "AWS_SECRET_ACCESS_KEY": {
            "secretKeyRef": {"key": "AWS_SECRET_ACCESS_KEY", "name": "s"}
        },
    }


# TODO: Remove this in 1.12.0
def test_mount_s3_backward_compatibility():
    """Test backward compatibility for S3_ENDPOINT_URL environment variable"""
    import os
    import warnings

    # Set up deprecated environment variable
    os.environ["S3_ENDPOINT_URL"] = "s3.deprecated.com"

    # Ensure AWS_ENDPOINT_URL_S3 is not set so we test the fallback
    os.environ.pop("AWS_ENDPOINT_URL_S3", None)

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )

    # Capture deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Use credentials so that the mount function actually sets environment variables
        function.apply(
            mlrun.runtimes.mounts.mount_s3(
                aws_access_key="test-key", aws_secret_key="test-secret"
            )
        )

        # Check that deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "S3_ENDPOINT_URL is deprecated" in str(w[0].message)

    env_dict = {var["name"]: var["value"] for var in function.spec.env}
    assert env_dict == {
        "AWS_ENDPOINT_URL_S3": "s3.deprecated.com",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
    }

    # Clean up
    os.environ.pop("S3_ENDPOINT_URL", None)


def test_set_env_variables():
    env_variables = {
        "some_env_1": "some-value",
        "SOMETHING": "ELSE",
        "and_another": "like_this",
    }

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    assert function.spec.env == []

    # Using a dictionary
    function.apply(mlrun.runtimes.mounts.set_env_variables(env_variables))
    env_dict = {var["name"]: var.get("value") for var in function.spec.env}

    assert env_dict == env_variables

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    assert function.spec.env == []

    # And using key=value parameters
    function.apply(mlrun.runtimes.mounts.set_env_variables(**env_variables))
    env_dict = {var["name"]: var.get("value") for var in function.spec.env}

    assert env_dict == env_variables


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
                mlrun.runtimes.mounts.VolumeMount(
                    "/volume-mount-path", "volume-sub-path"
                )
            ],
            "remote": "~/custom-remote",
            "expect_failure": True,
        },
        {
            "volume_mounts": [
                mlrun.runtimes.mounts.VolumeMount(
                    "/volume-mount-path", "volume-sub-path"
                ),
                mlrun.runtimes.mounts.VolumeMount(
                    "/volume-mount-path-2", "volume-sub-path-2"
                ),
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
                mlrun.runtimes.mounts.VolumeMount(
                    "/volume-mount-path", "volume-sub-path"
                ),
                mlrun.runtimes.mounts.VolumeMount(
                    "/volume-mount-path-2", "volume-sub-path-2"
                ),
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
                function.apply(mlrun.mount_v3io(**mount_v3io_kwargs))
        else:
            function.apply(mlrun.mount_v3io(**mount_v3io_kwargs))

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


# TODO: Remove in 1.11.0
@pytest.mark.parametrize(
    "mount, args, kwargs",
    [
        (mlrun.platforms.VolumeMount, ("", ""), {}),
        (mlrun.platforms.auto_mount, (), {"pvc_name": "a", "volume_mount_path": "b"}),
        (
            mlrun.platforms.mount_configmap,
            (),
            {"configmap_name": "a", "mount_path": "b"},
        ),
        (mlrun.platforms.mount_hostpath, (), {"host_path": "a", "mount_path": "b"}),
        (mlrun.platforms.mount_pvc, (), {"pvc_name": "a"}),
        (mlrun.platforms.mount_s3, (), {}),
        (mlrun.platforms.mount_secret, (), {"secret_name": "b", "mount_path": "c"}),
        (mlrun.platforms.mount_v3io, (), {"access_key": "bb", "user": "cc"}),
        (mlrun.platforms.set_env_variables, (), {}),
        (mlrun.platforms.v3io_cred, (), {}),
    ],
)
def test_mount_import_backwards_compatibility(mount, args, kwargs):
    """Test that the deprecated mlrun.platforms.mount_* functions import the new mlrun.runtimes.mounts.* functions."""
    assert isinstance(mount, mlrun.platforms._DeprecationHelper)
    assert type(mount(*args, **kwargs)) is type(
        getattr(mlrun.runtimes.mounts, mount._new_target)(*args, **kwargs)
    )


def _auth_prefix() -> str:
    # Matches how the code builds the pattern: format(hashed_access_key="")
    return mlrun.mlconf.secret_stores.kubernetes.auth_secret_name.format(
        hashed_access_key=""
    )


def test_mount_secret_blocks_auth_secret_name():
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    forbidden = _auth_prefix() + "anything"

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        function.apply(
            mlrun.runtimes.mounts.mount_secret(
                secret_name=forbidden,
                mount_path="/mnt/secret",
                volume_name="my-secret-vol",
            )
        )
    assert "Forbidden secret" in str(exc.value)
    assert forbidden in str(exc.value)


def test_mount_secret_allows_regular_secret_and_sets_volume():
    expected_volume = {
        "secret": {
            "secretName": "my-secret",
            "items": [{"key": "k", "path": "p"}],
        },
        "name": "my-volume",
    }
    expected_volume_mount = {"mountPath": "/mnt/secret", "name": "my-volume"}

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )

    function.apply(
        mlrun.runtimes.mounts.mount_secret(
            secret_name="my-secret",
            mount_path="/mnt/secret",
            volume_name="my-volume",
            items=[{"key": "k", "path": "p"}],
        )
    )

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
