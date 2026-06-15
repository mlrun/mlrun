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
import mlrun.errors
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


def test_auto_mount_s3():
    """Test that auto_mount() returns s3 mount modifier when auto_mount_type is 's3'."""
    mlrun.mlconf.storage.auto_mount_type = "s3"
    mlrun.mlconf.storage.auto_mount_params = (
        "endpoint_url=http://seaweedfs:8333,secret_name=minio-credentials"
    )

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(mlrun.runtimes.mounts.auto_mount())

    env_dict = {
        var["name"]: var.get("value", var.get("valueFrom")) for var in function.spec.env
    }
    assert env_dict == {
        "AWS_ENDPOINT_URL_S3": "http://seaweedfs:8333",
        "AWS_ACCESS_KEY_ID": {
            "secretKeyRef": {"key": "AWS_ACCESS_KEY_ID", "name": "minio-credentials"}
        },
        "AWS_SECRET_ACCESS_KEY": {
            "secretKeyRef": {
                "key": "AWS_SECRET_ACCESS_KEY",
                "name": "minio-credentials",
            }
        },
    }


@pytest.mark.parametrize("with_cleartext", [True, False])
@pytest.mark.parametrize("with_keys", [True, False])
def test_auto_mount_secret_env(with_keys, with_cleartext):
    """Test secret_env mount modifier: all 4 combos of with_keys x with_cleartext."""
    secret_name = "s3-credentials"
    keys = ["KEY_A", "KEY_B"]
    cleartext = {"PLAIN_VAR": "plainval"}

    mlrun.mlconf.storage.auto_mount_type = "secret_env"
    params = f"secret_name={secret_name}"
    if with_keys:
        params += f",keys={';'.join(keys)}"
    if with_cleartext:
        params += ",cleartext_env=" + ";".join(f"{k}:{v}" for k, v in cleartext.items())
    mlrun.mlconf.storage.auto_mount_params = params

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(mlrun.runtimes.mounts.auto_mount())

    if with_keys:
        # Each key should be a secretKeyRef env var
        secret_env_names = []
        for item in function.spec.env:
            if hasattr(item, "value_from") and item.value_from is not None:
                if (
                    hasattr(item.value_from, "secret_key_ref")
                    and item.value_from.secret_key_ref is not None
                ):
                    secret_env_names.append(item.name)
            elif isinstance(item, dict) and "valueFrom" in item:
                if "secretKeyRef" in item["valueFrom"]:
                    secret_env_names.append(item["name"])
        for key in keys:
            assert key in secret_env_names, (
                f"Expected {key} as secretKeyRef env var, got: {secret_env_names}"
            )
    else:
        # Whole secret via envFrom
        env_from = function.spec.env_from
        assert len(env_from) == 1
        assert env_from[0].config_map_ref is None
        assert env_from[0].secret_ref.name == secret_name

    if with_cleartext:
        plain_env = {}
        for item in function.spec.env:
            if hasattr(item, "value") and item.value is not None:
                plain_env[item.name] = item.value
            elif isinstance(item, dict) and "value" in item:
                plain_env[item["name"]] = item["value"]
        for k, v in cleartext.items():
            assert plain_env.get(k) == v, (
                f"Expected cleartext env var {k!r}={v!r}, got: {plain_env}"
            )
    else:
        # No plain env vars from cleartext
        plain_env = {}
        for item in function.spec.env:
            if hasattr(item, "value") and item.value is not None:
                plain_env[item.name] = item.value
            elif isinstance(item, dict) and "value" in item and "valueFrom" not in item:
                plain_env[item["name"]] = item["value"]
        cleartext_keys = set(cleartext.keys())
        overlap = cleartext_keys & set(plain_env.keys())
        assert not overlap, f"Expected no cleartext env vars but found: {overlap}"


def test_auto_mount_secret_env_cleartext_only_no_secret():
    """secret_env auto-mount with only cleartext_env (no secret_name) injects a plain env var.

    This is the Azure workload-identity path (ML-12692): the federated token arrives via the
    WI webhook, so only the storage account name needs to be set, with no secret mounted.
    """
    mlrun.mlconf.storage.auto_mount_type = "secret_env"
    mlrun.mlconf.storage.auto_mount_params = (
        "cleartext_env=AZURE_STORAGE_ACCOUNT:teststorage"
    )

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(mlrun.runtimes.mounts.auto_mount())

    # No secret is mounted (neither envFrom nor secretKeyRef env vars).
    assert len(function.spec.env_from) == 0
    plain_env = {}
    for item in function.spec.env:
        if isinstance(item, dict):
            assert "valueFrom" not in item, f"unexpected secret-backed env: {item}"
            plain_env[item["name"]] = item.get("value")
        else:
            assert getattr(item, "value_from", None) is None, (
                f"unexpected secret-backed env: {item}"
            )
            plain_env[item.name] = item.value
    assert plain_env.get("AZURE_STORAGE_ACCOUNT") == "teststorage"


def test_set_env_vars_from_secret_requires_secret_or_cleartext():
    """Calling without a secret_name and without cleartext_env is a usage error."""
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="secret_name or cleartext_env"
    ):
        mlrun.runtimes.mounts.set_env_vars_from_secret()


def test_set_env_vars_from_secret_keys_without_secret_name_raises():
    """Keys make no sense without a secret to read them from."""
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="secret_name when keys"
    ):
        mlrun.runtimes.mounts.set_env_vars_from_secret(keys="KEY_A;KEY_B")


@pytest.mark.parametrize(
    "cleartext_env_param,expected_error",
    [
        ("NOCORON", True),  # missing ':' — must raise
        ("K:V;NOCORON", True),  # second token missing ':'
        ("K:V;M:W", False),  # valid
        ("K:V", False),  # valid single
    ],
)
def test_set_env_vars_from_secret_cleartext_env_string_validation(
    cleartext_env_param, expected_error
):
    """Malformed cleartext_env string (missing ':') must raise MLRunInvalidArgumentError."""
    import mlrun.errors

    if expected_error:
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="cleartext_env"
        ):
            mlrun.runtimes.mounts.set_env_vars_from_secret(
                "my-secret", cleartext_env=cleartext_env_param
            )
    else:
        # Should not raise
        mlrun.runtimes.mounts.set_env_vars_from_secret(
            "my-secret", cleartext_env=cleartext_env_param
        )


def test_auto_mount_s3_takes_precedence_over_pvc_env():
    """Test that auto_mount_type=s3 takes precedence over MLRUN_PVC_MOUNT env var.

    When the server sets auto_mount_type=s3, it should be honoured even if
    MLRUN_PVC_MOUNT is set on the client (e.g., external Jupyter). See ML-12370.
    """
    mlrun.mlconf.storage.auto_mount_type = "s3"
    mlrun.mlconf.storage.auto_mount_params = (
        "endpoint_url=https://minio-lab.example.com,secret_name=minio-credentials"
    )
    os.environ["MLRUN_PVC_MOUNT"] = "some-pvc:/home/jovyan/"
    try:
        function = mlrun.new_function(
            "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
        )
        function.apply(mlrun.runtimes.mounts.auto_mount())

        env_dict = {
            var["name"]: var.get("value", var.get("valueFrom"))
            for var in function.spec.env
        }
        assert "AWS_ACCESS_KEY_ID" in env_dict
        assert env_dict["AWS_ENDPOINT_URL_S3"] == "https://minio-lab.example.com"
    finally:
        os.environ.pop("MLRUN_PVC_MOUNT", None)


def test_auto_mount_raises_without_config():
    """Test that auto_mount() raises ValueError when no mount type is configured."""
    mlrun.mlconf.storage.auto_mount_type = ""
    mlrun.mlconf.storage.auto_mount_params = ""

    # Clear env vars that could trigger other paths
    os.environ.pop("MLRUN_PVC_MOUNT", None)
    os.environ.pop("V3IO_ACCESS_KEY", None)

    with pytest.raises(ValueError, match="Failed to auto mount"):
        mlrun.runtimes.mounts.auto_mount()


def test_mount_s3_does_not_override_user_set_aws_access_key():
    """User-set plain AWS_ACCESS_KEY_ID survives mount_s3(secret_name=...).

    Regression for ML-12330: on IG4 the platform auto-mounts mount_s3 with
    secret_name='minio-credentials', which previously clobbered a user-supplied
    AWS_ACCESS_KEY_ID (e.g. set via the UI batch-run wizard) with a secretKeyRef.
    """
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.set_env("AWS_ACCESS_KEY_ID", "user-custom-value")

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="http://seaweedfs:8333",
        )
    )

    env_dict = {
        var["name"]: var.get("value", var.get("valueFrom")) for var in function.spec.env
    }
    # User's plain value is preserved (no secretKeyRef override).
    assert env_dict["AWS_ACCESS_KEY_ID"] == "user-custom-value"
    # AWS_SECRET_ACCESS_KEY was not user-set, so the secretKeyRef from the
    # modifier still wins — confirming the guard only fires per-key.
    assert env_dict["AWS_SECRET_ACCESS_KEY"] == {
        "secretKeyRef": {"key": "AWS_SECRET_ACCESS_KEY", "name": "minio-credentials"}
    }
    # Endpoint URL was not user-set, so the modifier's value wins.
    assert env_dict["AWS_ENDPOINT_URL_S3"] == "http://seaweedfs:8333"


def test_mount_s3_does_not_override_user_set_endpoint_url():
    """User-set plain AWS_ENDPOINT_URL_S3 survives mount_s3()."""
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.set_env("AWS_ENDPOINT_URL_S3", "https://my-corp-s3.example.com")

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="http://seaweedfs:8333",
        )
    )

    env_dict = {
        var["name"]: var.get("value", var.get("valueFrom")) for var in function.spec.env
    }
    assert env_dict["AWS_ENDPOINT_URL_S3"] == "https://my-corp-s3.example.com"


def test_mount_s3_without_secret_does_not_override_user_set_aws_access_key():
    """User-set plain AWS_ACCESS_KEY_ID also wins on the explicit-key path (no secret_name)."""
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.set_env("AWS_ACCESS_KEY_ID", "user-custom-value")

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            aws_access_key="modifier-access-key",
            aws_secret_key="modifier-secret-key",
        )
    )

    env_dict = {
        var["name"]: var.get("value", var.get("valueFrom")) for var in function.spec.env
    }
    assert env_dict["AWS_ACCESS_KEY_ID"] == "user-custom-value"
    # The non-user-set key is still filled by the modifier.
    assert env_dict["AWS_SECRET_ACCESS_KEY"] == "modifier-secret-key"


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


def test_mount_s3_marks_plain_value_writes_as_auto_mount_injected():
    """mount_s3(secret_name=...) flags only the plain-value writes.

    Regression for ML-12572: with secret_name set, only the endpoint URL is
    written as a plain value; access/secret keys are secretKeyRefs and must NOT
    be marked.
    """
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="http://seaweedfs:8333",
            aws_region="us-east-1",
        )
    )

    assert sorted(function.spec.auto_mount_injected_env_names) == [
        "AWS_ENDPOINT_URL_S3",
        "AWS_REGION",
    ]


def test_mount_s3_marks_plain_value_writes_on_explicit_key_path():
    """All plain-value writes are flagged on the explicit-key path (no secret_name)."""
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            aws_access_key="modifier-access-key",
            aws_secret_key="modifier-secret-key",
            endpoint_url="http://seaweedfs:8333",
        )
    )

    assert sorted(function.spec.auto_mount_injected_env_names) == [
        "AWS_ACCESS_KEY_ID",
        "AWS_ENDPOINT_URL_S3",
        "AWS_SECRET_ACCESS_KEY",
    ]


def test_has_user_set_plain_env_false_for_auto_mount_marked_env():
    """has_user_set_plain_env yields False for an auto-mount-injected plain value.

    Regression for ML-12572: the server-side project-secret resolver must be
    able to override mount_s3's plain writes.
    """
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="http://seaweedfs:8333",
        )
    )

    assert any(
        var["name"] == "AWS_ENDPOINT_URL_S3"
        and var.get("value") == "http://seaweedfs:8333"
        for var in function.spec.env
    )
    assert function.has_user_set_plain_env("AWS_ENDPOINT_URL_S3") is False


def test_user_set_env_after_auto_mount_clears_marker_and_wins():
    """A user-set plain value after auto-mount reclaims user-set semantics."""
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="http://seaweedfs:8333",
        )
    )
    assert "AWS_ENDPOINT_URL_S3" in function.spec.auto_mount_injected_env_names

    function.set_env("AWS_ENDPOINT_URL_S3", "https://real-s3.amazonaws.com")

    assert "AWS_ENDPOINT_URL_S3" not in function.spec.auto_mount_injected_env_names
    assert function.has_user_set_plain_env("AWS_ENDPOINT_URL_S3") is True


def test_project_secret_value_from_clears_auto_mount_marker():
    """Once a project secret writes a secretKeyRef for a key, the auto-mount
    marker is dropped — the auto-mount plain value has been replaced.
    """
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="http://seaweedfs:8333",
        )
    )
    assert "AWS_ENDPOINT_URL_S3" in function.spec.auto_mount_injected_env_names

    # Simulate what add_k8s_secrets_to_spec does on the server side.
    function.set_env_from_secret(
        "AWS_ENDPOINT_URL_S3",
        secret="mlrun-project-secrets-ppzdswyube",
        secret_key="AWS_ENDPOINT_URL_S3",
    )

    assert "AWS_ENDPOINT_URL_S3" not in function.spec.auto_mount_injected_env_names

    # Locate AWS_ENDPOINT_URL_S3 in spec.env; entries may be dicts (sanitized
    # from .apply()) or V1EnvVar (direct set_env_from_secret).
    endpoint_var = next(
        var
        for var in function.spec.env
        if mlrun.runtimes.utils.get_item_name(var) == "AWS_ENDPOINT_URL_S3"
    )
    plain_value = mlrun.runtimes.utils.get_item_name(endpoint_var, "value")
    assert plain_value is None
    assert getattr(endpoint_var, "value_from", None) is not None
    secret_key_ref = endpoint_var.value_from.secret_key_ref
    assert secret_key_ref.key == "AWS_ENDPOINT_URL_S3"
    assert secret_key_ref.name == "mlrun-project-secrets-ppzdswyube"


def test_kuberesourcespec_auto_mount_injected_env_names_round_trip():
    """The marker list survives spec serialization (SDK -> API)."""
    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="http://seaweedfs:8333",
        )
    )
    assert "AWS_ENDPOINT_URL_S3" in function.spec.auto_mount_injected_env_names

    spec_dict = function.spec.to_dict()
    assert "auto_mount_injected_env_names" in spec_dict
    assert "AWS_ENDPOINT_URL_S3" in spec_dict["auto_mount_injected_env_names"]

    # Reconstruct a spec the way new_function(runtime=<dict>) does.
    rebuilt = mlrun.runtimes.pod.KubeResourceSpec.from_dict(spec_dict)
    assert "AWS_ENDPOINT_URL_S3" in rebuilt.auto_mount_injected_env_names


def test_kuberesourcespec_defaults_auto_mount_injected_env_names_for_legacy_dict():
    """Spec dicts that pre-date the field default to an empty list (no KeyError)."""
    legacy_spec_dict = {
        "env": [{"name": "FOO", "value": "bar"}],
    }
    rebuilt = mlrun.runtimes.pod.KubeResourceSpec.from_dict(legacy_spec_dict)
    assert rebuilt.auto_mount_injected_env_names == []


def test_mount_s3_marker_survives_round_trip_on_application_runtime():
    """The actual ML-12572 reproducer: ApplicationRuntime + SDK↔API round-trip.

    For nuclio/application runtimes, mount_s3 runs in the SDK *before* the
    function spec is serialized and sent to the API server. The marker must
    survive `to_dict` / `new_function(runtime=dict)` so the server-side
    `has_user_set_plain_env` check sees it and project-secret injection wins.
    """
    function = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    function.apply(
        mlrun.runtimes.mounts.mount_s3(
            secret_name="minio-credentials",
            endpoint_url="https://minio-lab.iguazeng.com",
        )
    )
    assert "AWS_ENDPOINT_URL_S3" in function.spec.auto_mount_injected_env_names

    # SDK serializes the runtime; API server reconstructs it.
    runtime_dict = function.to_dict()
    rebuilt = mlrun.new_function(runtime=runtime_dict)

    assert "AWS_ENDPOINT_URL_S3" in rebuilt.spec.auto_mount_injected_env_names
    # And the server-side resolver's gate yields False, so project-secret wins.
    assert rebuilt.has_user_set_plain_env("AWS_ENDPOINT_URL_S3") is False
