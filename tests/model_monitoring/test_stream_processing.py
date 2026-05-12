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

import pytest

import mlrun
import mlrun.model_monitoring
from mlrun.common.schemas.model_monitoring.constants import (
    EventFieldType,
    NuclioMonitoringEnvVars,
)
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileKafkaStream,
    DatastoreProfilePostgreSQL,
    DatastoreProfileV3io,
)
from mlrun.model_monitoring.stream_processing import (
    EventStreamProcessor,
    ProcessHTTPEvent,
    TriggerRouter,
)

_MONITORING_STREAM_URI = "v3io:///projects/test/model-endpoints/stream"


@pytest.mark.parametrize(
    "tsdb_profile",
    [
        DatastoreProfileV3io(name="v3io-tsdb-test"),
        DatastoreProfilePostgreSQL(
            name="postgresql-tsdb-test",
            user="testuser",
            password="testpass",
            host="localhost",
            port=5432,
            database="postgres",
        ),
    ],
)
@pytest.mark.parametrize(
    "stream_profile",
    [
        DatastoreProfileV3io(name="v3io-stream-test"),
        DatastoreProfileKafkaStream(
            name="kafka-test", brokers=["localhost:9092"], topics=[]
        ),
    ],
)
def test_plot_monitoring_serving_graph(
    monkeypatch: pytest.MonkeyPatch,
    tsdb_profile: DatastoreProfile,
    stream_profile: DatastoreProfile,
) -> None:
    monkeypatch.setattr(mlrun.mlconf, "system_id", "123456")
    project_name = "test-stream-processing"
    project = mlrun.get_or_create_project(project_name, allow_cross_project=True)

    processor = EventStreamProcessor(project_name, 1000, 10, "mytarget")

    fn = project.set_function(
        kind="serving",
        name="my-fn",
    )

    tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
        project=project_name, profile=tsdb_profile
    )
    stream_path = mlrun.model_monitoring.get_stream_path(
        project=project_name, profile=stream_profile
    )

    processor.apply_monitoring_serving_graph(
        fn, tsdb_connector, stream_path, _MONITORING_STREAM_URI
    )

    graph = fn.spec.graph.plot(rankdir="TB")
    print()
    print(
        f"Graphviz graph definition with tsdb_connector={tsdb_connector} and stream_path={stream_path}"
    )
    print("Feed this to graphviz, or to https://dreampuf.github.io/GraphvizOnline")
    print()
    print(graph)


class _MockTrigger:
    def __init__(self, kind: str):
        self.kind = kind


class _MockEvent:
    def __init__(self, kind: str):
        self.trigger = _MockTrigger(kind)


class TestTriggerRouter:
    def test_http_trigger_routes_to_process_http(self):
        router = TriggerRouter()
        outlets = router.select_outlets(_MockEvent("http"))
        assert list(outlets) == ["ProcessHTTPEvent"]

    def test_stream_trigger_routes_to_stream_branch(self):
        router = TriggerRouter()
        for kind in ("v3io-stream", "kafka-cluster"):
            outlets = router.select_outlets(_MockEvent(kind))
            assert set(outlets) == {
                "FilterBatchComplete",
                "FilterError",
                "ForwardError",
            }, kind

    def test_unknown_trigger_routes_to_stream_branch(self):
        router = TriggerRouter()
        outlets = router.select_outlets(_MockEvent("cron"))
        assert set(outlets) == {"FilterBatchComplete", "FilterError", "ForwardError"}


class TestProcessHTTPEvent:
    """ProcessHTTPEvent.do() tests.

    _get_endpoint_schema is patched to return (None, None) so tests are
    isolated from the DB.  Tests that exercise schema-based normalisation
    supply schemas directly in the event body.
    """

    def _step(self, feature_names=None, label_names=None, function_uri=""):
        step = ProcessHTTPEvent(project="test-project")
        # Pre-populate cache so no DB call is made
        step._schema_cache["ep-123"] = (feature_names, label_names, function_uri)
        step._schema_cache["ep-1"] = (feature_names, label_names, function_uri)
        return step

    def test_valid_list_payload(self):
        step = self._step()
        result = step.do(
            {
                "model_endpoint_uid": "ep-123",
                "inputs": [[1.0, 2.0]],
                "outputs": [[0.8]],
                "model_endpoint_name": "my-model",
            }
        )
        assert result is not None
        assert result[EventFieldType.ENDPOINT_ID] == "ep-123"
        assert result[EventFieldType.MODEL] == "my-model"
        assert result["request"]["inputs"] == [[1.0, 2.0]]
        assert result["resp"]["outputs"] == [[0.8]]
        assert result[EventFieldType.FUNCTION_URI] == ""
        assert result["error"] is None

    def test_dict_inputs_transposed_by_schema(self):
        step = self._step(feature_names=["f1", "f2"], label_names=["pred"])
        result = step.do(
            {
                "model_endpoint_uid": "ep-123",
                "inputs": {"f2": 2.0, "f1": 1.0},
                "outputs": {"pred": 0.8},
            }
        )
        # 2 features → [[f1, f2]] (list-of-list); single label → [val] (flat)
        assert result["request"]["inputs"] == [[1.0, 2.0]]
        assert result["resp"]["outputs"] == [0.8]
        assert result["request"]["input_schema"] == ["f1", "f2"]
        assert result["resp"]["output_schema"] == ["pred"]

    def test_dict_inputs_without_schema_warns_and_uses_dict_order(self):
        step = self._step()
        result = step.do(
            {
                "model_endpoint_uid": "ep-123",
                "inputs": {"f1": 1.0, "f2": 2.0},
                "outputs": {"pred": 0.8},
            }
        )
        assert result is not None
        # No schema → transpose_by_key infers order from dict keys
        assert result["request"]["inputs"] == [[1.0, 2.0]]

    def test_scalar_inputs_wrapped_in_list(self):
        step = self._step()
        result = step.do(
            {
                "model_endpoint_uid": "ep-123",
                "inputs": 42.0,
                "outputs": 0.8,
            }
        )
        assert result["request"]["inputs"] == [42.0]
        assert result["resp"]["outputs"] == [0.8]

    def test_db_schema_used_when_not_in_event(self):
        step = self._step(feature_names=["a", "b"], label_names=["pred"])
        result = step.do(
            {
                "model_endpoint_uid": "ep-1",
                "inputs": {"b": 2.0, "a": 1.0},
                "outputs": {"pred": 0.9},
            }
        )
        # Schema from DB: ["a", "b"] → [[a_val, b_val]]
        assert result["request"]["inputs"] == [[1.0, 2.0]]
        assert result["resp"]["outputs"] == [0.9]
        assert result["request"]["input_schema"] == ["a", "b"]

    def test_when_added_if_missing(self):
        step = self._step()
        result = step.do(
            {"model_endpoint_uid": "ep-1", "inputs": [[1.0]], "outputs": [[0.8]]}
        )
        assert result["when"] is not None

    def test_when_preserved_if_provided(self):
        step = self._step()
        result = step.do(
            {
                "model_endpoint_uid": "ep-1",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )
        assert result["when"] == "2024-01-01T00:00:00Z"  # internal field name

    def test_missing_endpoint_id_returns_none(self):
        step = self._step()
        result = step.do({"inputs": [[1.0]], "outputs": [[0.9]]})
        assert result is None

    def test_missing_inputs_returns_none(self):
        step = self._step()
        result = step.do({"model_endpoint_uid": "ep-1", "outputs": [[0.9]]})
        assert result is None

    def test_missing_outputs_returns_none(self):
        step = self._step()
        result = step.do({"model_endpoint_uid": "ep-1", "inputs": [[1.0]]})
        assert result is None

    def test_model_empty_when_name_not_provided(self):
        step = self._step()
        result = step.do(
            {"model_endpoint_uid": "ep-1", "inputs": [[1.0]], "outputs": [[0.8]]}
        )
        assert result[EventFieldType.MODEL] == ""

    def test_optional_metadata_forwarded(self):
        step = self._step()
        result = step.do(
            {
                "model_endpoint_uid": "ep-1",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
                "timestamp": "2024-01-01T00:00:00Z",
                "latency": 123.4,
                "labels": {"env": "prod"},
                "metrics": {"accuracy": 0.99},
            }
        )
        assert result["when"] == "2024-01-01T00:00:00Z"  # internal field name
        assert result["microsec"] == 123.4
        assert result[EventFieldType.LABELS] == {"env": "prod"}
        assert result[EventFieldType.METRICS] == {"accuracy": 0.99}

    def test_request_id_generated_when_absent(self):
        step = self._step()
        result = step.do(
            {"model_endpoint_uid": "ep-1", "inputs": [[1.0]], "outputs": [[0.8]]}
        )
        assert result["request"]["id"] is not None
        assert len(result["request"]["id"]) > 0

    def test_function_uri_from_endpoint_schema(self):
        step = self._step(function_uri="my-project/my-fn:latest")
        result = step.do(
            {"model_endpoint_uid": "ep-1", "inputs": [[1.0]], "outputs": [[0.8]]}
        )
        assert result[EventFieldType.FUNCTION_URI] == "my-project/my-fn:latest"

    def test_function_uri_empty_for_user_ep(self):
        step = self._step(function_uri="")
        result = step.do(
            {"model_endpoint_uid": "ep-1", "inputs": [[1.0]], "outputs": [[0.8]]}
        )
        assert result[EventFieldType.FUNCTION_URI] == ""


class TestGetModelMonitoringUrl:
    """Unit tests for mlrun.get_model_monitoring_url env-var caching logic."""

    _ENV_VAR = NuclioMonitoringEnvVars.MODEL_MONITORING_URL

    def setup_method(self):
        # Ensure the env var is clear before each test
        os.environ.pop(self._ENV_VAR, None)

    def teardown_method(self):
        os.environ.pop(self._ENV_VAR, None)

    def test_returns_env_var_without_db_call(self, monkeypatch: pytest.MonkeyPatch):
        """When the env var is already set the DB must not be called."""
        os.environ[self._ENV_VAR] = "http://stream-pod-from-env/ingest"

        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == "http://stream-pod-from-env/ingest"
        mock_db.get_model_monitoring_url.assert_not_called()

    def test_fetches_from_db_when_env_var_absent(self, monkeypatch: pytest.MonkeyPatch):
        """When the env var is not set the URL is fetched from the DB."""
        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        mock_db.get_model_monitoring_url.return_value = (
            "http://stream-pod-from-db/ingest"
        )
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == "http://stream-pod-from-db/ingest"
        mock_db.get_model_monitoring_url.assert_called_once_with("my-project")

    def test_caches_db_result_in_env_var(self, monkeypatch: pytest.MonkeyPatch):
        """After a DB fetch the URL is stored in the env var for future calls."""
        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        mock_db.get_model_monitoring_url.return_value = (
            "http://stream-pod-from-db/ingest"
        )
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        mlrun.get_model_monitoring_url(project="my-project")

        assert os.environ.get(self._ENV_VAR) == "http://stream-pod-from-db/ingest"

    def test_second_call_uses_cache_not_db(self, monkeypatch: pytest.MonkeyPatch):
        """A second call must use the cached env var and skip the DB entirely."""
        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        mock_db.get_model_monitoring_url.return_value = (
            "http://stream-pod-from-db/ingest"
        )
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        mlrun.get_model_monitoring_url(project="my-project")
        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == "http://stream-pod-from-db/ingest"
        mock_db.get_model_monitoring_url.assert_called_once()  # only the first call

    def test_returns_none_when_db_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """When the DB returns None (no HTTP trigger) the env var is not set."""
        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        mock_db.get_model_monitoring_url.return_value = None
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url is None
        assert self._ENV_VAR not in os.environ
