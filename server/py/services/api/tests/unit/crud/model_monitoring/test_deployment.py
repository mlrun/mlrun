# Copyright 2024 Iguazio
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
from collections.abc import Iterator
from unittest.mock import Mock, patch

import kafka.errors
import pytest

import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.runtimes
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    DatastoreProfileKafkaStream,
    DatastoreProfilePostgreSQL,
    DatastoreProfileV3io,
)

import services.api
import services.api.crud.model_monitoring.deployment as mm_dep


@pytest.fixture()
def monitoring_deployment() -> mm_dep.MonitoringDeployment:
    return mm_dep.MonitoringDeployment(
        project=Mock(spec=str),
        auth_info=Mock(spec=mm_dep.mlrun.common.schemas.AuthInfo),
        db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
        model_monitoring_access_key=None,
    )


class SecretTester(services.api.crud.secrets.Secrets):
    _secrets: dict[str, dict[str, str]] = {}

    def store_project_secrets(
        self,
        project: str,
        secrets: mlrun.common.schemas.SecretsData,
        allow_internal_secrets: bool = False,
        key_map_secret_key: str | None = None,
        allow_storing_key_maps: bool = False,
    ):
        self._secrets[project] = secrets.secrets

    def get_project_secret(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secret_key: str,
        token: str | None = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: str | None = None,
    ) -> str | None:
        return self._secrets.get(project, {}).get(secret_key)


class TestAppDeployment:
    """Test nominal flow of the app deployment"""

    @staticmethod
    @pytest.fixture(autouse=True)
    def _patch_build_function() -> Iterator[None]:
        with patch(
            "services.api.api.endpoints.nuclio._deploy_function",
            new=Mock(return_value=Mock(spec=mlrun.runtimes.ServingRuntime)),
        ):
            with patch("services.api.crud.Secrets", new=SecretTester):
                yield

    @staticmethod
    def test_app_dep(monitoring_deployment: mm_dep.MonitoringDeployment) -> None:
        monitoring_deployment.deploy_histogram_data_drift_app(
            image="mlrun/mlrun", overwrite=True
        )

    @pytest.mark.skipif(
        os.getenv("V3IO_FRAMESD") is None
        or os.getenv("V3IO_ACCESS_KEY") is None
        or os.getenv("V3IO_API") is None,
        reason="Configure Framsed to access V3IO store targets",
    )
    def test_credentials(
        self,
        monitoring_deployment: mm_dep.MonitoringDeployment,
    ) -> None:
        # new project case
        with patch(
            "services.api.crud.Functions.get_function",
            side_effect=mlrun.errors.MLRunNotFoundError,
        ):
            with pytest.raises(mlrun.errors.MLRunBadRequestError):
                monitoring_deployment.check_if_credentials_are_set()

            with pytest.raises(mlrun.errors.MLRunInvalidMMStoreTypeError):
                monitoring_deployment.set_credentials(
                    stream_path="kafka://stream",
                    tsdb_connection="wrong",
                )

            with pytest.raises(kafka.errors.NoBrokersAvailable):
                monitoring_deployment.set_credentials(
                    stream_path="kafka://stream",
                    tsdb_connection="v3io",
                )

            monitoring_deployment.set_credentials(
                stream_path="v3io",
            )

            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()

            monitoring_deployment.set_credentials(
                tsdb_connection="v3io",
            )

            with pytest.raises(mlrun.errors.MLRunConflictError):
                monitoring_deployment.set_credentials(
                    stream_path="v3io",
                    tsdb_connection="v3io",
                )
            monitoring_deployment.set_credentials(
                replace_creds=True,
            )

            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()
        # existing project - upgrade from 1.6.0 case
        monitoring_deployment.project = "1.6.0_project"
        with patch(
            "services.api.crud.Functions.get_function", return_value=Mock(spec={})
        ):
            with pytest.raises(mlrun.errors.MLRunConflictError):
                monitoring_deployment.set_credentials(
                    stream_path="v3io",
                    tsdb_connection="v3io",
                )
            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()
            for key, value in secrets.items():
                if (
                    key
                    != mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
                ):
                    assert value == "v3io"
            monitoring_deployment.set_credentials(
                stream_path="v3io",
                tsdb_connection="v3io",
                replace_creds=True,
            )

            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()


@pytest.mark.parametrize(
    "nuclio_annotations",
    [
        {
            "nuclio.io/kafka-metadata-retry-max": "10000",
            "nuclio.io/kafka-metadata-timeout": "1200s",
        },
        None,
    ],
)
@patch("mlrun.datastore.sources.KafkaSource.create_topics")
def test_apply_and_create_kafka_source(
    create_topics_mock: Mock,
    nuclio_annotations: dict[str, str] | None,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """Test that the Kafka trigger is set correctly"""
    replication_factor = 3

    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["sub.confluent.cloud:9092"],
        topics=[],
        sasl_user="usr1",
        sasl_pass="pass123",
        kwargs_public={
            "security_protocol": "SASL_SSL",
            "api_version_auto_timeout_ms": 15_000,
            "tls": {"enable": True},
            "new_topic": {"replication_factor": replication_factor},
            **(
                {"nuclio_annotations": nuclio_annotations} if nuclio_annotations else {}
            ),
        },
    )

    fn = mlrun.runtimes.ServingRuntime()
    monitoring_deployment._apply_and_create_kafka_source(
        kafka_profile=kafka_profile,
        function=fn,
        function_name="test-confluent-trigger",
        stream_args=mlrun.mlconf.model_endpoint_monitoring.serving_stream,
        ignore_stream_already_exists_failure=True,
    )

    create_topics_mock.assert_called_once_with(
        num_partitions=8, replication_factor=replication_factor
    )

    kafka_trigger_conf = fn.spec.config.get("spec.triggers.kafka")
    assert kafka_trigger_conf, "Expected a Kafka trigger"
    assert kafka_trigger_conf.get("kind") == "kafka-cluster", (
        "Expected `kafka-cluster` kind"
    )
    assert (
        fn.spec.base_spec.get("metadata", {}).get("annotations") == nuclio_annotations
    ), "The set annotations are different than expected"
    assert fn.spec.custom_scaling_metric_specs[0]["resource"]["target"] == {
        "type": "AverageValue",
        "averageValue": "400m",
    }


@patch("mlrun.datastore.sources.KafkaSource.create_topics")
def test_kafka_source_no_hpa_target_when_not_configured(
    create_topics_mock: Mock,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """ML-11991: When target_cpu is empty, custom_scaling_metric_specs
    should remain at its default (empty list)."""
    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["localhost:9092"],
        topics=[],
    )

    fn = mlrun.runtimes.ServingRuntime()
    stream_args = mlrun.mlconf.model_endpoint_monitoring.serving_stream
    original_target_cpu = stream_args.kafka.target_cpu
    try:
        stream_args.kafka.target_cpu = ""
        monitoring_deployment._apply_and_create_kafka_source(
            kafka_profile=kafka_profile,
            function=fn,
            function_name="model-monitoring-stream",
            stream_args=stream_args,
            ignore_stream_already_exists_failure=True,
        )
    finally:
        stream_args.kafka.target_cpu = original_target_cpu

    assert fn.spec.custom_scaling_metric_specs == []


class TestInjectMonitoringEnvVars:
    """Unit tests for MonitoringDeployment.inject_monitoring_env_vars (no I/O)."""

    def test_injects_into_empty_spec(self):
        fn = {}
        result = mm_dep.MonitoringDeployment.inject_monitoring_env_vars(
            fn, {"MODEL_MONITORING_URL": "http://stream:8080"}
        )
        env = result["spec"]["env"]
        assert {"name": "MODEL_MONITORING_URL", "value": "http://stream:8080"} in env

    def test_appends_new_vars(self):
        fn = {"spec": {"env": [{"name": "EXISTING", "value": "1"}]}}
        mm_dep.MonitoringDeployment.inject_monitoring_env_vars(
            fn,
            {
                "MODEL_ENDPOINT_UID": "uid-abc",
                "MODEL_ENDPOINTS_MAP": '{"ep": "uid-abc"}',
            },
        )
        env = fn["spec"]["env"]
        names = {e["name"] for e in env}
        assert "EXISTING" in names
        assert "MODEL_ENDPOINT_UID" in names
        assert "MODEL_ENDPOINTS_MAP" in names

    def test_replaces_existing_var(self):
        fn = {"spec": {"env": [{"name": "MODEL_MONITORING_URL", "value": "old"}]}}
        mm_dep.MonitoringDeployment.inject_monitoring_env_vars(
            fn, {"MODEL_MONITORING_URL": "new"}
        )
        env = fn["spec"]["env"]
        assert len(env) == 1
        assert env[0]["value"] == "new"

    def test_multiple_vars_single_uid(self):
        fn = {}
        mm_dep.MonitoringDeployment.inject_monitoring_env_vars(
            fn,
            {
                "MODEL_MONITORING_URL": "http://stream",
                "MODEL_ENDPOINT_UID": "uid-1",
            },
        )
        env = {e["name"]: e["value"] for e in fn["spec"]["env"]}
        assert env["MODEL_MONITORING_URL"] == "http://stream"
        assert env["MODEL_ENDPOINT_UID"] == "uid-1"
        assert "MODEL_ENDPOINTS_MAP" not in env


class TestBuildAndInjectMonitoringEnvVars:
    """Unit tests for _build_and_inject_monitoring_env_vars."""

    def _make_endpoint(self, name: str, uid: str) -> mlrun.common.schemas.ModelEndpoint:
        ep = mlrun.common.schemas.ModelEndpoint(
            metadata=mlrun.common.schemas.model_monitoring.ModelEndpointMetadata(
                project="proj", name=name, uid=uid
            ),
            spec=mlrun.common.schemas.model_monitoring.ModelEndpointSpec(),
            status=mlrun.common.schemas.model_monitoring.ModelEndpointStatus(),
        )
        return ep

    def test_single_endpoint_injects_uid_and_url(self):
        from mlrun.common.schemas.model_monitoring.constants import (
            ModelEndpointCreationStrategy,
        )

        dep = mm_dep.MonitoringDeployment(project="proj")
        ep = self._make_endpoint("ep1", "uid-111")
        instructions = [(ep, ModelEndpointCreationStrategy.INPLACE)]
        fn = dep._build_and_inject_monitoring_env_vars(
            function={},
            model_endpoints_instructions=instructions,
            stream_url="http://stream:8080",
        )
        env = {e["name"]: e["value"] for e in fn["spec"]["env"]}
        assert env["MODEL_MONITORING_URL"] == "http://stream:8080"
        assert env["MODEL_ENDPOINT_UID"] == "uid-111"
        assert "MODEL_ENDPOINTS_MAP" not in env

    def test_multiple_endpoints_injects_map(self):
        import json

        from mlrun.common.schemas.model_monitoring.constants import (
            ModelEndpointCreationStrategy,
        )

        dep = mm_dep.MonitoringDeployment(project="proj")
        ep1 = self._make_endpoint("ep1", "uid-1")
        ep2 = self._make_endpoint("ep2", "uid-2")
        instructions = [
            (ep1, ModelEndpointCreationStrategy.INPLACE),
            (ep2, ModelEndpointCreationStrategy.INPLACE),
        ]
        fn = dep._build_and_inject_monitoring_env_vars(
            function={},
            model_endpoints_instructions=instructions,
            stream_url=None,
        )
        env = {e["name"]: e["value"] for e in fn["spec"]["env"]}
        assert env["MODEL_ENDPOINT_UID"] == "uid-1"
        assert "MODEL_MONITORING_URL" not in env
        ep_map = json.loads(env["MODEL_ENDPOINTS_MAP"])
        assert ep_map == {"ep1": "uid-1", "ep2": "uid-2"}

    def test_no_stream_url_skips_url_var(self):
        from mlrun.common.schemas.model_monitoring.constants import (
            ModelEndpointCreationStrategy,
        )

        dep = mm_dep.MonitoringDeployment(project="proj")
        ep = self._make_endpoint("ep1", "uid-x")
        fn = dep._build_and_inject_monitoring_env_vars(
            function={},
            model_endpoints_instructions=[(ep, ModelEndpointCreationStrategy.INPLACE)],
            stream_url=None,
        )
        names = {e["name"] for e in fn["spec"]["env"]}
        assert "MODEL_MONITORING_URL" not in names
        assert "MODEL_ENDPOINT_UID" in names

    def test_empty_instructions_raises(self):
        dep = mm_dep.MonitoringDeployment(project="proj")
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="empty or malformed"
        ):
            dep._build_and_inject_monitoring_env_vars(
                function={},
                model_endpoints_instructions=[],
                stream_url=None,
            )


class TestCreateModelEndpointsInstructionsForNuclioApp:
    """Tests for _create_model_endpoints_instructions_for_nuclio_app."""

    def _function_dict(self, instructions: list, tag: str = "latest") -> dict:
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        return {
            "metadata": {"tag": tag},
            "spec": {
                "model_endpoints_instructions": [
                    i.to_dict() if isinstance(i, ModelEndpointInstruction) else i
                    for i in instructions
                ]
            },
        }

    @staticmethod
    def _mock_db_and_stream(stream_url, existing_endpoints=None):
        """Return context managers for get_stream_url and get_db().list_model_endpoints."""
        from unittest.mock import AsyncMock, MagicMock, patch

        db_mock = MagicMock()
        db_mock.list_model_endpoints.return_value = existing_endpoints or {}
        stream_patch = patch(
            "services.api.crud.model_monitoring.helpers.get_stream_url",
            new=AsyncMock(return_value=stream_url),
        )
        db_patch = patch(
            "framework.utils.singletons.db.get_db",
            return_value=db_mock,
        )
        return stream_patch, db_patch

    @pytest.mark.asyncio
    async def test_happy_path_returns_instructions_and_url(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        dep = mm_dep.MonitoringDeployment(project="proj")
        fn_dict = self._function_dict(
            [ModelEndpointInstruction(name="ep1", input_schema=["f1"])]
        )

        stream_patch, db_patch = self._mock_db_and_stream("http://stream:8080")
        with stream_patch, db_patch:
            (
                instructions,
                stream_url,
                _,
            ) = await dep._create_model_endpoints_instructions_for_nuclio_app(
                db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
                function=fn_dict,
                function_name="my-fn",
                project="proj",
            )

        assert len(instructions) == 1
        ep, strategy = instructions[0]
        assert ep.metadata.name == "ep1"
        assert ep.spec.feature_names == ["f1"]
        assert stream_url == "http://stream:8080"

    @pytest.mark.asyncio
    async def test_empty_instructions_returns_empty_list(self):
        dep = mm_dep.MonitoringDeployment(project="proj")
        fn_dict = self._function_dict([])

        stream_patch, db_patch = self._mock_db_and_stream("http://stream:8080")
        with stream_patch, db_patch:
            (
                instructions,
                stream_url,
                _,
            ) = await dep._create_model_endpoints_instructions_for_nuclio_app(
                db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
                function=fn_dict,
                function_name="my-fn",
                project="proj",
            )

        assert instructions == []

    @pytest.mark.asyncio
    async def test_instruction_objects_accepted_directly(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        dep = mm_dep.MonitoringDeployment(project="proj")
        fn_dict = {
            "metadata": {"tag": "latest"},
            "spec": {
                "model_endpoints_instructions": [
                    ModelEndpointInstruction(name="ep-obj")
                ]
            },
        }

        stream_patch, db_patch = self._mock_db_and_stream("http://stream:8080")
        with stream_patch, db_patch:
            (
                instructions,
                stream_url,
                _,
            ) = await dep._create_model_endpoints_instructions_for_nuclio_app(
                db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
                function=fn_dict,
                function_name="my-fn",
                project="proj",
            )

        assert len(instructions) == 1
        ep, _ = instructions[0]
        assert ep.metadata.name == "ep-obj"

    @pytest.mark.asyncio
    async def test_stream_url_none_logs_warning(self):
        from mlrun.common.schemas.model_monitoring.model_endpoints import (
            ModelEndpointInstruction,
        )

        dep = mm_dep.MonitoringDeployment(project="proj")
        fn_dict = self._function_dict([ModelEndpointInstruction(name="ep1")])

        stream_patch, db_patch = self._mock_db_and_stream(None)
        with stream_patch, db_patch:
            (
                instructions,
                stream_url,
                _,
            ) = await dep._create_model_endpoints_instructions_for_nuclio_app(
                db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
                function=fn_dict,
                function_name="my-fn",
                project="proj",
            )

        assert stream_url is None
        assert len(instructions) == 1

    @pytest.mark.asyncio
    async def test_invalid_instruction_type_raises(self):
        dep = mm_dep.MonitoringDeployment(project="proj")
        fn_dict = {
            "metadata": {"tag": "latest"},
            "spec": {"model_endpoints_instructions": [42]},
        }

        stream_patch, db_patch = self._mock_db_and_stream(None)
        with stream_patch, db_patch, pytest.raises(Exception):
            await dep._create_model_endpoints_instructions_for_nuclio_app(
                db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
                function=fn_dict,
                function_name="my-fn",
                project="proj",
            )


def _kafka_trigger_group(fn: mlrun.runtimes.ServingRuntime) -> str:
    return fn.spec.config["spec.triggers.kafka"]["attributes"]["consumerGroup"]


def _kafka_trigger_topic(fn: mlrun.runtimes.ServingRuntime) -> str:
    return fn.spec.config["spec.triggers.kafka"]["attributes"]["topics"][0]


@patch("mlrun.datastore.sources.KafkaSource.create_topics")
def test_kafka_consumer_group_is_per_function_with_profile_prefix(
    create_topics_mock: Mock,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """Consumer group = f"{profile.group}_{topic}".

    Each MM function gets its own consumer group so a rebalance in one
    function (e.g. stream HPA scaling) does not pause the others. The
    user-supplied profile ``group`` is preserved as a prefix rather than
    silently discarded.
    """
    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["localhost:9092"],
        topics=[],
        # Custom group — must be honored, not overridden.
        group="prod",
    )

    fn = mlrun.runtimes.ServingRuntime()
    monitoring_deployment._apply_and_create_kafka_source(
        kafka_profile=kafka_profile,
        function=fn,
        function_name="model-monitoring-stream",
        stream_args=mlrun.mlconf.model_endpoint_monitoring.serving_stream,
        ignore_stream_already_exists_failure=True,
    )

    topic = _kafka_trigger_topic(fn)
    group = _kafka_trigger_group(fn)
    assert group == f"prod_{topic}", (
        "Consumer group should be '<profile.group>_<topic>' "
        f"(got {group!r}, topic={topic!r})"
    )


@patch("mlrun.datastore.sources.KafkaSource.create_topics")
def test_kafka_consumer_group_defaults_to_serving_when_profile_group_is_none(
    create_topics_mock: Mock,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """If the profile group is None, the prefix falls back to 'serving'."""
    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["localhost:9092"],
        topics=[],
        group=None,
    )

    fn = mlrun.runtimes.ServingRuntime()
    monitoring_deployment._apply_and_create_kafka_source(
        kafka_profile=kafka_profile,
        function=fn,
        function_name="model-monitoring-writer",
        stream_args=mlrun.mlconf.model_endpoint_monitoring.writer_stream_args,
        ignore_stream_already_exists_failure=True,
    )

    topic = _kafka_trigger_topic(fn)
    group = _kafka_trigger_group(fn)
    assert group == f"serving_{topic}"


@patch(
    "mlrun.datastore.sources.KafkaSource.create_topics",
    side_effect=kafka.errors.TopicAlreadyExistsError(),
)
@patch(
    "services.api.crud.model_monitoring.deployment.MonitoringDeployment"
    "._migrate_kafka_consumer_group_offsets",
)
def test_kafka_migration_invoked_on_topic_already_exists(
    migrate_offsets_mock: Mock,
    create_topics_mock: Mock,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """When the topic already exists on a tolerant call path, migration
    is invoked with the profile group as the source and the per-function
    group as the destination. Idempotency lives inside the helper, so it
    runs on every re-enable and short-circuits itself."""
    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["localhost:9092"],
        topics=[],
        group="prod",
    )

    fn = mlrun.runtimes.ServingRuntime()
    monitoring_deployment._apply_and_create_kafka_source(
        kafka_profile=kafka_profile,
        function=fn,
        function_name="model-monitoring-stream",
        stream_args=mlrun.mlconf.model_endpoint_monitoring.serving_stream,
        ignore_stream_already_exists_failure=True,
    )

    migrate_offsets_mock.assert_called_once()
    call_kwargs = migrate_offsets_mock.call_args.kwargs
    assert call_kwargs["old_group"] == "prod"
    assert call_kwargs["new_group"] == f"prod_{call_kwargs['topic']}"


@patch(
    "mlrun.datastore.sources.KafkaSource.create_topics",
    side_effect=kafka.errors.TopicAlreadyExistsError(),
)
@patch(
    "services.api.crud.model_monitoring.deployment.MonitoringDeployment"
    "._migrate_kafka_consumer_group_offsets",
)
def test_kafka_migration_invoked_for_all_mm_functions(
    migrate_offsets_mock: Mock,
    create_topics_mock: Mock,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """Pre-PR, every MM function shared the same ``kafka_profile.group``.
    Migration must therefore run for writer/controller/apps too — not only
    for the stream function."""
    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["localhost:9092"],
        topics=[],
    )

    fn = mlrun.runtimes.ServingRuntime()
    monitoring_deployment._apply_and_create_kafka_source(
        kafka_profile=kafka_profile,
        function=fn,
        function_name="model-monitoring-writer",
        stream_args=mlrun.mlconf.model_endpoint_monitoring.writer_stream_args,
        ignore_stream_already_exists_failure=True,
    )

    migrate_offsets_mock.assert_called_once()


@patch(
    "mlrun.datastore.sources.KafkaSource.create_topics",
    side_effect=kafka.errors.TopicAlreadyExistsError(),
)
def test_kafka_migration_raises_on_kafka_failure(
    create_topics_mock: Mock,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """A failed migration must propagate: silently falling back to
    ``initial_offset=earliest`` would replay the whole topic and duplicate
    every already-processed event."""
    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["localhost:9092"],
        topics=[],
    )

    with patch(
        "kafka.admin.KafkaAdminClient",
        side_effect=kafka.errors.NoBrokersAvailable(),
    ):
        fn = mlrun.runtimes.ServingRuntime()
        with pytest.raises(kafka.errors.NoBrokersAvailable):
            monitoring_deployment._apply_and_create_kafka_source(
                kafka_profile=kafka_profile,
                function=fn,
                function_name="model-monitoring-stream",
                stream_args=mlrun.mlconf.model_endpoint_monitoring.serving_stream,
                ignore_stream_already_exists_failure=True,
            )


@patch(
    "mlrun.datastore.sources.KafkaSource.create_topics",
    side_effect=kafka.errors.TopicAlreadyExistsError(),
)
def test_kafka_migration_not_invoked_when_ignore_flag_false(
    create_topics_mock: Mock,
    monitoring_deployment: mm_dep.MonitoringDeployment,
) -> None:
    """When ``ignore_stream_already_exists_failure=False`` (e.g. controller
    deploy without overwrite), ``TopicAlreadyExistsError`` must propagate
    as it always has — migration logic is only triggered on the tolerant
    branch."""
    kafka_profile = DatastoreProfileKafkaStream(
        name="test-kafka-profile",
        brokers=["localhost:9092"],
        topics=[],
    )

    fn = mlrun.runtimes.ServingRuntime()
    with patch.object(
        monitoring_deployment, "_migrate_kafka_consumer_group_offsets"
    ) as migrate_offsets_mock:
        with pytest.raises(kafka.errors.TopicAlreadyExistsError):
            monitoring_deployment._apply_and_create_kafka_source(
                kafka_profile=kafka_profile,
                function=fn,
                function_name="model-monitoring-controller",
                stream_args=mlrun.mlconf.model_endpoint_monitoring.controller_stream_args,
                ignore_stream_already_exists_failure=False,
            )
    migrate_offsets_mock.assert_not_called()


# --- ML-12543: project.spec.model_monitoring persistence + OTel validation -----


class TestOtelEnableValidation:
    """deploy_monitoring_functions(otlp_enabled=True) must fail fast when the
    operator hasn't configured a usable OTLP endpoint.

    Note on bool vs. string: `mlconf.telemetry.enabled` is a Python bool at
    runtime (env vars are type-coerced from the schema default in
    mlrun/config.py). Tests set True/False, not the strings "true"/"false".
    """

    @staticmethod
    @pytest.fixture(autouse=True)
    def reset_telemetry_config(monkeypatch):
        # Keep tests deterministic regardless of repo defaults.
        monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", False, raising=False)
        monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "", raising=False)

    @staticmethod
    def test_raises_when_telemetry_disabled(
        monitoring_deployment: mm_dep.MonitoringDeployment, monkeypatch
    ) -> None:
        # Endpoint configured but master kill-switch off → reject.
        monkeypatch.setattr(
            mlrun.mlconf.telemetry,
            "otlp_endpoint",
            "https://otel.example.com:4317",
            raising=False,
        )
        monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", False, raising=False)

        with pytest.raises(
            mlrun.errors.MLRunBadRequestError, match="disabled telemetry"
        ):
            monitoring_deployment.deploy_monitoring_functions(otlp_enabled=True)

    @staticmethod
    def test_raises_when_endpoint_blank(
        monitoring_deployment: mm_dep.MonitoringDeployment, monkeypatch
    ) -> None:
        # Telemetry on but no endpoint → reject.
        monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", True, raising=False)
        monkeypatch.setattr(mlrun.mlconf.telemetry, "otlp_endpoint", "", raising=False)

        with pytest.raises(
            mlrun.errors.MLRunBadRequestError, match="otlp_endpoint is blank"
        ):
            monitoring_deployment.deploy_monitoring_functions(otlp_enabled=True)

    @staticmethod
    def test_passes_when_telemetry_enabled_and_endpoint_set(
        monitoring_deployment: mm_dep.MonitoringDeployment, monkeypatch
    ) -> None:
        """Regression guard for the original bug: the precheck used `!= "true"`
        which always fired against the bool True value. With the bool-truthy
        fix, this happy path must not raise the OTel validator at all.
        """
        monkeypatch.setattr(mlrun.mlconf.telemetry, "enabled", True, raising=False)
        monkeypatch.setattr(
            mlrun.mlconf.telemetry,
            "otlp_endpoint",
            "https://otel.example.com:4317",
            raising=False,
        )

        # Deploy will still fail later (no credentials configured in this fixture),
        # but it must NOT raise either of the OTel-precheck errors.
        with pytest.raises(Exception) as exc_info:
            monitoring_deployment.deploy_monitoring_functions(otlp_enabled=True)
        assert "disabled telemetry" not in str(exc_info.value)
        assert "otlp_endpoint is blank" not in str(exc_info.value)

    @staticmethod
    def test_does_not_check_when_otlp_disabled(
        monitoring_deployment: mm_dep.MonitoringDeployment,
    ) -> None:
        """When the project doesn't opt in to OTel, operator config is irrelevant.
        deploy will fail later on credentials, but NOT on the telemetry check."""
        with pytest.raises(Exception) as exc_info:
            monitoring_deployment.deploy_monitoring_functions(otlp_enabled=False)
        # Anything OTHER than our bad-request validator is acceptable here —
        # the test only proves the OTel pre-check didn't fire.
        assert "disabled telemetry" not in str(exc_info.value)
        assert "otlp_endpoint is blank" not in str(exc_info.value)


class TestPersistModelMonitoringSpec:
    """`_persist_model_monitoring_spec` writes exactly the fields it was given."""

    @staticmethod
    def test_writes_only_provided_fields(
        monitoring_deployment: mm_dep.MonitoringDeployment,
    ) -> None:
        with patch.object(services.api.crud.Projects, "patch_project") as patch_mock:
            monitoring_deployment._persist_model_monitoring_spec(
                enabled=True, otlp_enabled=True
            )

        patch_mock.assert_called_once()
        kwargs = patch_mock.call_args.kwargs
        assert kwargs["patch_mode"] == mlrun.common.schemas.PatchMode.additive
        # The patch dict must contain only the keys we passed — no stream/tsdb
        # type leaked in with default Nones.
        assert kwargs["project"] == {
            "spec": {
                "model_monitoring": {
                    "enabled": True,
                    "otlp_enabled": True,
                }
            }
        }

    @staticmethod
    def test_skips_patch_when_no_fields_set(
        monitoring_deployment: mm_dep.MonitoringDeployment,
    ) -> None:
        """Empty call → no DB write."""
        with patch.object(services.api.crud.Projects, "patch_project") as patch_mock:
            monitoring_deployment._persist_model_monitoring_spec()

        patch_mock.assert_not_called()


class TestResolveStreamTarget:
    """`_resolve_stream_target` maps a stream datastore profile class to its
    `StreamTarget` enum so the project spec gets a canonical type string.
    """

    @staticmethod
    def test_kafka_stream_resolves_to_kafka() -> None:
        profile = DatastoreProfileKafkaStream(
            name="p", brokers=["broker:9092"], topics=[]
        )
        assert (
            mm_dep.MonitoringDeployment._resolve_stream_target(profile)
            == mm_constants.StreamTarget.KAFKA
        )

    @staticmethod
    def test_kafka_source_subclass_resolves_to_kafka() -> None:
        # DatastoreProfileKafkaSource subclasses DatastoreProfileKafkaStream so
        # the isinstance check still classifies it as KAFKA.
        profile = DatastoreProfileKafkaSource(
            name="p", brokers=["broker:9092"], topics=[]
        )
        assert (
            mm_dep.MonitoringDeployment._resolve_stream_target(profile)
            == mm_constants.StreamTarget.KAFKA
        )

    @staticmethod
    def test_v3io_resolves_to_v3io() -> None:
        profile = DatastoreProfileV3io(name="p")
        assert (
            mm_dep.MonitoringDeployment._resolve_stream_target(profile)
            == mm_constants.StreamTarget.V3IO
        )

    @staticmethod
    def test_unknown_profile_returns_none() -> None:
        # Postgres isn't a stream profile — caller's contract is to get None
        # back so it can decide whether to skip the persist.
        profile = DatastoreProfilePostgreSQL(
            name="p", host="h", port=5432, user="u", password="p", database="d"
        )
        assert mm_dep.MonitoringDeployment._resolve_stream_target(profile) is None


class TestResolveTSDBTarget:
    """`_resolve_tsdb_target` maps a TSDB datastore profile class to its
    `TSDBTarget` enum.
    """

    @staticmethod
    def test_postgresql_resolves_to_timescaledb() -> None:
        profile = DatastoreProfilePostgreSQL(
            name="p", host="h", port=5432, user="u", password="p", database="d"
        )
        assert (
            mm_dep.MonitoringDeployment._resolve_tsdb_target(profile)
            == mm_constants.TSDBTarget.TimescaleDB
        )

    @staticmethod
    def test_v3io_resolves_to_v3io_tsdb() -> None:
        profile = DatastoreProfileV3io(name="p")
        assert (
            mm_dep.MonitoringDeployment._resolve_tsdb_target(profile)
            == mm_constants.TSDBTarget.V3IO_TSDB
        )

    @staticmethod
    def test_unknown_profile_returns_none() -> None:
        # Kafka isn't a TSDB profile.
        profile = DatastoreProfileKafkaStream(
            name="p", brokers=["broker:9092"], topics=[]
        )
        assert mm_dep.MonitoringDeployment._resolve_tsdb_target(profile) is None


class TestSetCredentialsPersistsResolvedTypes:
    """`set_credentials` derives stream_type/tsdb_type from the registered
    profiles and writes them through `_persist_model_monitoring_spec` so the
    project spec immediately reflects the chosen backends.
    """

    @staticmethod
    def test_kafka_stream_and_postgres_types_persisted(
        monitoring_deployment: mm_dep.MonitoringDeployment,
    ) -> None:
        stream_profile = DatastoreProfileKafkaStream(
            name="my-kafka", brokers=["broker:9092"], topics=[]
        )
        tsdb_profile = DatastoreProfilePostgreSQL(
            name="my-pg",
            host="h",
            port=5432,
            user="u",
            password="p",
            database="d",
        )

        with (
            patch.object(
                monitoring_deployment,
                "check_if_credentials_are_set",
                side_effect=mlrun.errors.MLRunBadRequestError,
            ),
            patch.object(
                monitoring_deployment,
                "_get_monitoring_mandatory_project_secrets",
                return_value={},
            ),
            patch.object(
                monitoring_deployment,
                "_validate_stream_profile",
                return_value=stream_profile,
            ),
            patch.object(
                monitoring_deployment,
                "_validate_and_get_tsdb_profile",
                return_value=tsdb_profile,
            ),
            patch.object(monitoring_deployment, "_create_tsdb_tables"),
            patch("services.api.crud.Secrets.store_project_secrets"),
            patch.object(
                monitoring_deployment, "_persist_model_monitoring_spec"
            ) as persist_mock,
        ):
            monitoring_deployment.set_credentials(
                tsdb_profile_name="my-pg",
                stream_profile_name="my-kafka",
            )

        # Server resolves the profile classes to enum values and persists onto
        # the project spec — the SDK's _enrich then picks these up on the
        # client side.
        persist_mock.assert_called_once_with(
            stream_type=mm_constants.StreamTarget.KAFKA,
            tsdb_type=mm_constants.TSDBTarget.TimescaleDB,
        )


class TestDisableModelMonitoringPersistsDisabledSpec:
    """`disable_model_monitoring` must declaratively reset
    `project.spec.model_monitoring.enabled` and `.otlp_enabled` to False
    regardless of which functions were torn down.
    """

    @staticmethod
    @pytest.mark.asyncio
    async def test_persists_disabled_after_teardown(
        monitoring_deployment: mm_dep.MonitoringDeployment,
    ) -> None:
        # Make the function-existence probe return falsy so the per-function
        # teardown loop is a no-op — we only want to exercise the
        # spec-persist line at the end of disable_model_monitoring.
        with (
            patch.object(
                monitoring_deployment,
                "_get_monitoring_application_to_delete",
                return_value=[],
            ),
            patch.object(
                monitoring_deployment, "_get_function_state", return_value=None
            ),
            patch.object(
                monitoring_deployment, "_persist_model_monitoring_spec"
            ) as persist_mock,
        ):
            await monitoring_deployment.disable_model_monitoring(
                delete_resources=True,
                delete_stream_function=False,
                delete_histogram_data_drift_app=True,
            )

        persist_mock.assert_called_once_with(enabled=False, otlp_enabled=False)
