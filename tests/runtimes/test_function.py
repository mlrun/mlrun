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

import pathlib
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
from deepdiff import DeepDiff

import mlrun
import mlrun.errors
from mlrun import code_to_function
from mlrun.datastore.datastore_profile import DatastoreProfileRabbitMQ
from mlrun.utils.helpers import resolve_git_reference_from_source
from tests.runtimes.test_base import TestAutoMount


def test_generate_nuclio_volumes():
    volume_1_name = "volume-name"
    volume_1 = {
        "name": volume_1_name,
        "flexVolume": {
            "driver": "v3io/fuse",
            "options": {
                "container": "users",
                "accessKey": "4dbc1521-f6f2-4b28-aeac-29073413b9ae",
                "subPath": "/pipelines/.mlrun",
            },
        },
    }
    volume_2_name = "second-volume-name"
    volume_2 = {
        "name": volume_2_name,
        "secret": {"secretName": "secret-name"},
    }
    volume_1_volume_mount_1 = {
        "name": volume_1_name,
        "mountPath": "/v3io/volume/mount/path",
    }
    volume_1_volume_mount_2 = {
        "name": volume_1_name,
        "mountPath": "/v3io/volume/mount/2/path",
    }
    volume_2_volume_mount_1 = {
        "name": volume_2_name,
        "mountPath": "/secret/second/volume/mount/path",
    }
    runtime = {
        "kind": "nuclio",
        "metadata": {"name": "some-function", "project": "default"},
        "spec": {
            "volumes": [volume_1, volume_2],
            "volume_mounts": [
                volume_1_volume_mount_1,
                volume_1_volume_mount_2,
                volume_2_volume_mount_1,
            ],
        },
    }
    expected_nuclio_volumes = [
        {"volume": volume_1, "volumeMount": volume_1_volume_mount_1},
        {"volume": volume_1, "volumeMount": volume_1_volume_mount_2},
        {"volume": volume_2, "volumeMount": volume_2_volume_mount_1},
    ]
    function = mlrun.new_function(runtime=runtime)
    nuclio_volumes = function.spec.generate_nuclio_volumes()
    assert (
        DeepDiff(
            expected_nuclio_volumes,
            nuclio_volumes,
            ignore_order=True,
        )
        == {}
    )


class TestAutoMountNuclio(TestAutoMount):
    def setup_method(self, method):
        super().setup_method(method)
        self.assets_path = (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )
        self.code_filename = str(self.assets_path / "sample_function.py")
        self.code_handler = "test_func"

    def _generate_runtime(self, disable_auto_mount=False):
        runtime = code_to_function(
            name=self.name,
            project=self.project,
            filename=self.code_filename,
            handler=self.code_handler,
            kind="nuclio",
            image=self.image_name,
            description="test function",
        )
        runtime.spec.disable_auto_mount = disable_auto_mount
        return runtime

    def _execute_run(self, runtime):
        runtime.deploy(project=self.project)


def test_http_trigger():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.with_http(
        workers=2,
        host="x",
        worker_timeout=5,
        extra_attributes={"yy": "123"},
    )

    trigger = function.spec.config["spec.triggers.http"]
    print(trigger)
    assert trigger["maxWorkers"] == 2
    assert trigger["attributes"]["ingresses"] == {"0": {"host": "x", "paths": ["/"]}}
    assert trigger["attributes"]["yy"] == "123"
    assert trigger["workerAvailabilityTimeoutMilliseconds"] == 5000
    assert (
        trigger["annotations"]["nginx.ingress.kubernetes.io/proxy-connect-timeout"]
        == "65"
    )


def test_nuclio_deploy_set_token_name():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    db = mlrun.get_run_db()
    db.token_provider = MagicMock(token_name="provider-nuclio-token")
    db.deploy_nuclio_function = MagicMock(
        return_value={"data": {"status": {}, "spec": function.spec}}
    )
    function._wait_for_function_deployment = MagicMock()
    function._update_credentials_from_remote_build = MagicMock()
    function._enrich_command_from_status = MagicMock(return_value={})

    function.deploy()

    assert function.spec.auth["token_name"] == "provider-nuclio-token"

    with mlrun.RuntimeConfigurationContext(auth_token_name="context-nuclio-token"):
        function.deploy()
        assert function.spec.auth["token_name"] == "context-nuclio-token"


def test_v3io_stream_trigger():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.add_v3io_stream_trigger(
        "v3io:///projects/x/y",
        name="mystream",
        extra_attributes={"yy": "123"},
        ack_window_size=10,
    )
    trigger = function.spec.config["spec.triggers.mystream"]
    assert trigger["attributes"]["containerName"] == "projects"
    assert trigger["attributes"]["streamPath"] == "x/y"
    assert trigger["password"] == mlrun.model.Credentials.generate_access_key
    assert trigger["attributes"]["yy"] == "123"
    assert trigger["attributes"]["ackWindowSize"] == 10


@pytest.mark.parametrize(
    "consumer_group,expected",
    [
        ("my_group", does_not_raise()),
        ("_mygroup", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
    ],
)
def test_v3io_stream_trigger_validate_consumer_group(consumer_group, expected):
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    with expected:
        function.add_v3io_stream_trigger(
            "v3io:///projects/x/y",
            name="mystream",
            group=consumer_group,
        )
        trigger = function.spec.config["spec.triggers.mystream"]
        assert trigger["attributes"]["consumerGroup"] == consumer_group


def test_rabbitmq_trigger_with_queue_name():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.add_rabbitmq_trigger(
        url="amqp://rabbitmq-host:5672",
        exchange_name="my-exchange",
        queue_name="my-queue",
        username="user",
        password="pass",
        prefetch_count=10,
        durable_exchange=True,
        durable_queue=True,
        num_workers=4,
    )
    trigger = function.spec.config["spec.triggers.rabbitmq"]
    assert trigger["kind"] == "rabbit-mq"
    assert trigger["url"] == "amqp://rabbitmq-host:5672"
    assert trigger["numWorkers"] == 4
    assert trigger["username"] == "user"
    assert trigger["password"] == "pass"
    assert trigger["attributes"]["exchangeName"] == "my-exchange"
    assert trigger["attributes"]["queueName"] == "my-queue"
    assert trigger["attributes"]["prefetchCount"] == 10
    assert trigger["attributes"]["durableExchange"] is True
    assert trigger["attributes"]["durableQueue"] is True


def test_rabbitmq_trigger_with_topics():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.add_rabbitmq_trigger(
        url="amqp://rabbitmq-host:5672",
        exchange_name="my-exchange",
        topics=["key1", "key2"],
        name="my-rabbitmq",
    )
    trigger = function.spec.config["spec.triggers.my-rabbitmq"]
    assert trigger["kind"] == "rabbit-mq"
    assert trigger["attributes"]["exchangeName"] == "my-exchange"
    assert trigger["attributes"]["topics"] == ["key1", "key2"]
    assert "queueName" not in trigger["attributes"]


def test_rabbitmq_trigger_extracts_credentials_from_url():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.add_rabbitmq_trigger(
        url="amqp://myuser:mypass@rabbitmq-host:5672",
        exchange_name="my-exchange",
        queue_name="my-queue",
    )
    trigger = function.spec.config["spec.triggers.rabbitmq"]
    assert trigger["url"] == "amqp://rabbitmq-host:5672"
    assert trigger["username"] == "myuser"
    assert trigger["password"] == "mypass"


def test_rabbitmq_trigger_decodes_url_encoded_credentials():
    """Test that URL-encoded special characters in credentials are decoded."""
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    # Password contains special characters: p@ss word (encoded as p%40ss%20word)
    function.add_rabbitmq_trigger(
        url="amqp://my%40user:p%40ss%20word@rabbitmq-host:5672",
        queue_name="my-queue",
    )
    trigger = function.spec.config["spec.triggers.rabbitmq"]
    assert trigger["url"] == "amqp://rabbitmq-host:5672"
    assert trigger["username"] == "my@user"
    assert trigger["password"] == "p@ss word"


def test_rabbitmq_trigger_error_handling_config():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.add_rabbitmq_trigger(
        url="amqp://rabbitmq-host:5672",
        exchange_name="my-exchange",
        queue_name="my-queue",
        on_error="ack",
        requeue_on_error=True,
    )
    trigger = function.spec.config["spec.triggers.rabbitmq"]
    assert trigger["attributes"]["onError"] == "ack"
    assert trigger["attributes"]["requeueOnError"] is True


@pytest.mark.parametrize(
    "queue_name,topics,expected",
    [
        # Both specified - error (mutually exclusive)
        ("queue", ["key"], pytest.raises(ValueError)),
        # Only queue_name - OK
        ("queue", None, does_not_raise()),
        # Only topics - OK
        (None, ["key"], does_not_raise()),
        # Neither specified - OK (let Nuclio handle validation)
        (None, None, does_not_raise()),
    ],
)
def test_rabbitmq_trigger_queue_or_topics_validation(queue_name, topics, expected):
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    with expected:
        function.add_rabbitmq_trigger(
            url="amqp://rabbitmq-host:5672",
            queue_name=queue_name,
            topics=topics,
        )


@pytest.mark.parametrize(
    "on_error,expected",
    [
        ("ack", does_not_raise()),
        ("nack", does_not_raise()),
        ("invalid", pytest.raises(ValueError)),
    ],
)
def test_rabbitmq_trigger_on_error_validation(on_error, expected):
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    with expected:
        function.add_rabbitmq_trigger(
            url="amqp://rabbitmq-host:5672",
            queue_name="my-queue",
            on_error=on_error,
        )


def test_rabbitmq_trigger_with_datastore_profile():
    # Create a mock profile
    mock_profile = DatastoreProfileRabbitMQ(
        name="test-rabbitmq",
        broker_url="amqp://profile-host:5672",
        exchange_name="profile-exchange",
        queue_name="profile-queue",
        username="profile-user",
        password="profile-pass",
        prefetch_count=5,
        durable_exchange=True,
    )

    with patch(
        "mlrun.datastore.datastore_profile.datastore_profile_read",
        return_value=mock_profile,
    ):
        function: mlrun.runtimes.RemoteRuntime = mlrun.new_function(
            "tst", kind="nuclio"
        )
        function.add_rabbitmq_trigger(url="ds://test-rabbitmq")

        trigger = function.spec.config["spec.triggers.rabbitmq"]
        assert trigger["kind"] == "rabbit-mq"
        assert trigger["url"] == "amqp://profile-host:5672"
        assert trigger["username"] == "profile-user"
        assert trigger["password"] == "profile-pass"
        assert trigger["attributes"]["exchangeName"] == "profile-exchange"
        assert trigger["attributes"]["queueName"] == "profile-queue"
        assert trigger["attributes"]["prefetchCount"] == 5
        assert trigger["attributes"]["durableExchange"] is True


def test_rabbitmq_trigger_explicit_falsy_values_override_profile():
    """Test that explicitly passed falsy values (0, False) override profile defaults."""
    # Create a profile with non-default values
    mock_profile = DatastoreProfileRabbitMQ(
        name="test-rabbitmq",
        broker_url="amqp://profile-host:5672",
        exchange_name="profile-exchange",
        queue_name="profile-queue",
        prefetch_count=10,  # Profile has 10
        durable_exchange=True,  # Profile has True
        durable_queue=True,  # Profile has True
        num_workers=5,  # Profile has 5
    )

    with patch(
        "mlrun.datastore.datastore_profile.datastore_profile_read",
        return_value=mock_profile,
    ):
        function: mlrun.runtimes.RemoteRuntime = mlrun.new_function(
            "tst", kind="nuclio"
        )
        # Explicitly pass falsy values that should override profile
        function.add_rabbitmq_trigger(
            url="ds://test-rabbitmq",
            prefetch_count=0,  # Explicit 0 should override profile's 10
            durable_exchange=False,  # Explicit False should override profile's True
            durable_queue=False,  # Explicit False should override profile's True
            num_workers=1,  # Explicit 1 should override profile's 5
        )

        trigger = function.spec.config["spec.triggers.rabbitmq"]
        # Verify explicit falsy values were used, not profile values
        assert trigger["attributes"]["prefetchCount"] == 0
        assert trigger["attributes"]["durableExchange"] is False
        assert trigger["attributes"]["durableQueue"] is False
        assert trigger["numWorkers"] == 1


def test_resolve_git_reference_from_source():
    cases = [
        # source, (repo, refs, branch)
        ("repo", ("repo", "", "")),
        ("repo#br", ("repo", "", "br")),
        ("repo#refs/heads/main", ("repo", "refs/heads/main", "")),
        ("repo#refs/heads/main#commit", ("repo", "refs/heads/main#commit", "")),
    ]
    for source, expected in cases:
        assert expected == resolve_git_reference_from_source(source)


@pytest.mark.parametrize("function_kind", ["serving", "remote"])
def test_update_credentials_from_remote_build(function_kind):
    secret_name = "secret-name"
    remote_data = {
        "metadata": {"credentials": {"access_key": secret_name}},
        "spec": {
            "env": [
                {"name": "V3IO_ACCESS_KEY", "value": secret_name},
                {"name": "MLRUN_AUTH_SESSION", "value": secret_name},
            ],
        },
    }

    function = mlrun.new_function("tst", kind=function_kind)
    function.metadata.credentials.access_key = "access_key"
    function.spec.env = [
        {"name": "V3IO_ACCESS_KEY", "value": "access_key"},
        {"name": "MLRUN_AUTH_SESSION", "value": "access_key"},
    ]
    function._update_credentials_from_remote_build(remote_data)

    assert function.metadata.credentials.access_key == secret_name
    assert function.spec.env == remote_data["spec"]["env"]


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("valid_tag", does_not_raise()),
        ("invalid%$tag", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("too-long-tag" * 10, pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
    ],
)
def test_invalid_tags(tag, expected, rundb_mock):
    function = mlrun.new_function("test", kind="nuclio", tag=tag)
    with expected:
        function.pre_deploy_validation()


@pytest.mark.parametrize(
    "command, args, expected_sidecars",
    (
        [
            None,
            ["a", "b"],
            [
                {
                    "name": "tst-sidecar",
                    "ports": [
                        {
                            "containerPort": None,
                            "name": "tst-sidecar-0",
                            "protocol": "TCP",
                        }
                    ],
                }
            ],
        ],
        [
            "abc",
            ["a", "b"],
            [
                {
                    "args": ["a", "b"],
                    "command": ["abc"],
                    "name": "tst-sidecar",
                    "ports": [
                        {
                            "containerPort": None,
                            "name": "tst-sidecar-0",
                            "protocol": "TCP",
                        }
                    ],
                }
            ],
        ],
    ),
)
def test_with_sidecar(command: str, args: list, expected_sidecars: list):
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.with_sidecar(
        command=command,
        args=args,
    )

    assert function.spec.config["spec.sidecars"] == expected_sidecars


@pytest.mark.parametrize(
    "external_url, expected_scheme",
    [
        ("my-gateway.default-tenant.app.example.com", "https://"),
        ("https://my-gateway.example.com", "https://"),
        ("http://my-gateway.example.com", "http://"),
    ],
)
def test_resolve_invocation_url_uses_https_for_external_urls(
    external_url, expected_scheme
):
    fn = mlrun.new_function("test-fn", kind="nuclio")
    fn.status.external_invocation_urls = [external_url]

    resolved = fn._resolve_invocation_url("/test-path", force_external_address=False)

    assert resolved.startswith(expected_scheme)
    assert "/test-path" in resolved


def test_with_source_archive_removes_inline_code(logs_stream):
    # Verify that when a Nuclio function already contains inline code and the user attaches a source archive
    # (without using with_repo),the inline code is removed and a warning is logged so the archive will actually be used.
    fn = mlrun.new_function("test-func", kind="nuclio")
    fn.spec.build.functionSourceCode = "some-code"
    source = "git://github.com/org/repo.git"

    # call with_source_archive with a dummy source
    fn.with_source_archive(source=source)

    # assert inline code was cleared
    assert fn.spec.build.functionSourceCode is None, "Inline code should be cleared"

    # assert warning was issued
    assert (
        "Cannot specify both code and source archive. Removing the code so the provided "
        "source archive will be used instead" in logs_stream.getvalue()
    )

    # assert that the source was set correctly
    assert fn.spec.build.source == source


class TestSetupModelMonitoring:
    def _nuclio_fn(self, name="test-fn"):
        return mlrun.new_function(name, kind="nuclio")

    def test_extra_instructions_as_dicts(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn()
        fn_name = fn.metadata.name
        extra = [
            {"name": fn_name, "input_schema": ["f1"]},
            {"name": fn_name, "output_schema": ["label"]},
        ]
        fn.setup_model_monitoring(extra_model_endpoint_instructions=extra)

        instructions = fn.spec.model_endpoints_instructions
        assert len(instructions) == 3  # 1 default + 2 extra
        assert all(isinstance(i, ModelEndpointInstruction) for i in instructions)
        # Verify dict-to-object conversion preserved the schema fields
        assert any(i.input_schema == ["f1"] for i in instructions)
        assert any(i.output_schema == ["label"] for i in instructions)

    def test_extra_instructions_as_objects(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn()
        fn_name = fn.metadata.name
        extra = [
            ModelEndpointInstruction(name=fn_name),
            ModelEndpointInstruction(name=fn_name, input_schema=["x"]),
        ]
        fn.setup_model_monitoring(extra_model_endpoint_instructions=extra)

        instructions = fn.spec.model_endpoints_instructions
        assert len(instructions) == 3
        assert all(isinstance(i, ModelEndpointInstruction) for i in instructions)
        assert any(i.input_schema == ["x"] for i in instructions)

    def test_extra_instructions_with_explicit_primary(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn()
        fn_name = fn.metadata.name
        primary = ModelEndpointInstruction(name=fn_name)
        extra = [{"name": fn_name}]
        fn.setup_model_monitoring(
            general_model_endpoint_instructions=primary,
            extra_model_endpoint_instructions=extra,
        )

        instructions = fn.spec.model_endpoints_instructions
        assert len(instructions) == 2
        assert all(i.name == fn_name for i in instructions)
        assert fn.spec.track_models is True

    def test_returns_self_for_chaining(self):
        fn = self._nuclio_fn()
        result = fn.setup_model_monitoring()
        assert result is fn

    def test_setup_model_monitoring_warns_on_override(self):
        import warnings

        fn = self._nuclio_fn()
        fn.setup_model_monitoring()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn.setup_model_monitoring()
        assert len(w) == 1
        assert "overridden" in str(w[0].message).lower()

    def test_extra_instructions_mixed_types_raises(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn()
        mixed = [
            ModelEndpointInstruction(name="ep-obj"),
            {"name": "ep-dict"},
        ]
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="mix"):
            fn.setup_model_monitoring(extra_model_endpoint_instructions=mixed)

    def test_extra_instructions_name_mismatch_raises(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn(name="my-fn")
        extra = [ModelEndpointInstruction(name="wrong-name")]
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="name mismatch"
        ):
            fn.setup_model_monitoring(extra_model_endpoint_instructions=extra)

    def test_extra_instructions_tag_mismatch_raises(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn(name="my-fn")
        fn.metadata.tag = "v1"
        extra = [ModelEndpointInstruction(name="my-fn", function_tag="v2")]
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="tag mismatch"
        ):
            fn.setup_model_monitoring(extra_model_endpoint_instructions=extra)

    def test_extra_instructions_no_tag_skips_tag_check(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn(name="my-fn")
        fn.metadata.tag = "v1"
        extra = [ModelEndpointInstruction(name="my-fn")]  # function_tag=None
        fn.setup_model_monitoring(extra_model_endpoint_instructions=extra)
        names = [i.name for i in fn.spec.model_endpoints_instructions]
        assert "my-fn" in names

    def test_name_mismatch_raises(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn(name="my-fn")
        instruction = ModelEndpointInstruction(name="wrong-name")
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="name mismatch"
        ):
            fn.setup_model_monitoring(general_model_endpoint_instructions=instruction)

    def test_function_tag_mismatch_raises(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn(name="my-fn")
        fn.metadata.tag = "v1"
        instruction = ModelEndpointInstruction(name="my-fn", function_tag="v2")
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="tag mismatch"
        ):
            fn.setup_model_monitoring(general_model_endpoint_instructions=instruction)

    def test_matching_tag_does_not_raise(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn(name="my-fn")
        fn.metadata.tag = "v1"
        instruction = ModelEndpointInstruction(name="my-fn", function_tag="v1")
        fn.setup_model_monitoring(general_model_endpoint_instructions=instruction)
        assert fn.spec.model_endpoints_instructions[0].function_tag == "v1"

    def test_no_function_tag_skips_tag_check(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        fn = self._nuclio_fn(name="my-fn")
        fn.metadata.tag = "v1"
        instruction = ModelEndpointInstruction(name="my-fn")  # function_tag=None
        fn.setup_model_monitoring(general_model_endpoint_instructions=instruction)
        assert fn.spec.model_endpoints_instructions[0].name == "my-fn"
