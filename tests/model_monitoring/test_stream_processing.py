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

import json
import os
import unittest.mock

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
    _HTTP_ERROR_KEY,
    EventStreamProcessor,
    HTTPAckResponder,
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

    def _step(self, monkeypatch, feature_names=None, label_names=None, function_uri=""):
        ep = mlrun.common.schemas.ModelEndpoint(
            metadata=mlrun.common.schemas.ModelEndpointMetadata(
                name="my-model", project="test-project"
            ),
            spec=mlrun.common.schemas.ModelEndpointSpec(
                feature_names=feature_names or [],
                label_names=label_names or [],
                function_uri=function_uri,
            ),
            status=mlrun.common.schemas.ModelEndpointStatus(),
        )
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_endpoint.return_value = ep
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda *a, **kw: mock_db)
        return ProcessHTTPEvent(project="test-project")

    async def test_valid_list_payload(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
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

    async def test_dict_inputs_transposed_by_schema(self, monkeypatch):
        step = self._step(monkeypatch, feature_names=["f1", "f2"], label_names=["pred"])
        result = await step.do(
            {
                "model_endpoint_uid": "ep-123",
                "model_endpoint_name": "my-model",
                "inputs": {"f2": 2.0, "f1": 1.0},
                "outputs": {"pred": 0.8},
            }
        )
        # 2 features → [[f1, f2]] (list-of-list); single label → [val] (flat)
        assert result["request"]["inputs"] == [[1.0, 2.0]]
        assert result["resp"]["outputs"] == [0.8]
        assert result["request"]["input_schema"] == ["f1", "f2"]
        assert result["resp"]["output_schema"] == ["pred"]

    async def test_dict_inputs_without_schema_warns_and_uses_dict_order(
        self, monkeypatch
    ):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-123",
                "model_endpoint_name": "my-model",
                "inputs": {"f1": 1.0, "f2": 2.0},
                "outputs": {"pred": 0.8},
            }
        )
        assert result is not None
        # No schema → transpose_by_key infers order from dict keys
        assert result["request"]["inputs"] == [[1.0, 2.0]]

    async def test_scalar_inputs_wrapped_in_list(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-123",
                "model_endpoint_name": "my-model",
                "inputs": 42.0,
                "outputs": 0.8,
            }
        )
        assert result["request"]["inputs"] == [42.0]
        assert result["resp"]["outputs"] == [0.8]

    async def test_db_schema_used_when_not_in_event(self, monkeypatch):
        step = self._step(monkeypatch, feature_names=["a", "b"], label_names=["pred"])
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": {"b": 2.0, "a": 1.0},
                "outputs": {"pred": 0.9},
            }
        )
        # Schema from DB: ["a", "b"] → [[a_val, b_val]]
        assert result["request"]["inputs"] == [[1.0, 2.0]]
        assert result["resp"]["outputs"] == [0.9]
        assert result["request"]["input_schema"] == ["a", "b"]

    async def test_when_added_if_missing(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
            }
        )
        assert result["when"] is not None

    async def test_when_preserved_if_provided(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )
        assert result["when"] == "2024-01-01T00:00:00Z"  # internal field name

    async def test_missing_endpoint_id_returns_error_sentinel(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {"model_endpoint_name": "my-model", "inputs": [[1.0]], "outputs": [[0.9]]}
        )
        assert _HTTP_ERROR_KEY in result
        assert "model_endpoint_uid" in result[_HTTP_ERROR_KEY]

    async def test_missing_inputs_returns_error_sentinel(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "outputs": [[0.9]],
            }
        )
        assert _HTTP_ERROR_KEY in result
        assert "inputs" in result[_HTTP_ERROR_KEY]

    async def test_missing_outputs_returns_error_sentinel(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
            }
        )
        assert _HTTP_ERROR_KEY in result
        assert "outputs" in result[_HTTP_ERROR_KEY]

    async def test_missing_name_returns_error_sentinel(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {"model_endpoint_uid": "ep-1", "inputs": [[1.0]], "outputs": [[0.8]]}
        )
        assert _HTTP_ERROR_KEY in result
        assert "model_endpoint_name" in result[_HTTP_ERROR_KEY]

    async def test_optional_metadata_forwarded(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
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

    async def test_request_id_generated_when_absent(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
            }
        )
        assert result["request"]["id"] is not None
        assert len(result["request"]["id"]) > 0

    async def test_function_uri_from_endpoint_schema(self, monkeypatch):
        step = self._step(
            monkeypatch,
            feature_names=["f1"],
            label_names=["out"],
            function_uri="my-project/my-fn:latest",
        )
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
            }
        )
        assert result[EventFieldType.FUNCTION_URI] == "my-project/my-fn:latest"

    async def test_translation_exception_returns_error_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import mlrun.serving.system_steps

        monkeypatch.setattr(
            mlrun.serving.system_steps,
            "_to_listed_data",
            lambda data, schema: (_ for _ in ()).throw(ValueError("boom")),
        )
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
            }
        )
        assert _HTTP_ERROR_KEY in result
        assert "failed to translate event" in result[_HTTP_ERROR_KEY]
        assert "boom" in result[_HTTP_ERROR_KEY]

    async def test_function_uri_empty_for_user_ep(self, monkeypatch):
        step = self._step(monkeypatch, function_uri="")
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
            }
        )
        assert result[EventFieldType.FUNCTION_URI] == ""

    async def test_not_found_endpoint_returns_error_sentinel(self, monkeypatch):
        """When the endpoint does not exist, do() returns a 'not found' error sentinel."""
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_endpoint.side_effect = mlrun.errors.MLRunNotFoundError(
            "endpoint not found"
        )
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda *a, **kw: mock_db)
        step = ProcessHTTPEvent(project="test-project")

        result = await step.do(
            {
                "model_endpoint_uid": "ep-missing",
                "model_endpoint_name": "no-such-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
            }
        )

        assert _HTTP_ERROR_KEY in result
        assert "model endpoint not found" in result[_HTTP_ERROR_KEY]
        assert "ep-missing" in result[_HTTP_ERROR_KEY]


class TestGetEndpointSchema:
    """Unit tests for ProcessHTTPEvent._get_endpoint_schema cache logic."""

    def _make_ep(self, feature_names=None, label_names=None, function_uri=""):
        ep = mlrun.common.schemas.ModelEndpoint(
            metadata=mlrun.common.schemas.ModelEndpointMetadata(
                name="my-model", project="proj"
            ),
            spec=mlrun.common.schemas.ModelEndpointSpec(
                feature_names=feature_names or [],
                label_names=label_names or [],
                function_uri=function_uri,
            ),
            status=mlrun.common.schemas.ModelEndpointStatus(),
        )
        return ep

    def _mock_db(self, monkeypatch, ep):
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_endpoint.return_value = ep
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda *a, **kw: mock_db)
        return mock_db

    async def test_cache_miss_calls_db_and_populates_cache(self, monkeypatch):
        ep = self._make_ep(["f1"], ["out"], "proj/fn:latest")
        mock_db = self._mock_db(monkeypatch, ep)
        step = ProcessHTTPEvent(project="proj")

        result = await step._get_endpoint_schema("ep-1", "my-model")

        assert result == (["f1"], ["out"], "proj/fn:latest")
        mock_db.get_model_endpoint.assert_called_once()
        assert step._schema_cache["ep-1"] == (["f1"], ["out"], "proj/fn:latest")

    async def test_cache_hit_with_schema_skips_db(self, monkeypatch):
        mock_db = self._mock_db(monkeypatch, self._make_ep())
        step = ProcessHTTPEvent(project="proj")
        step._schema_cache["ep-1"] = (["f1"], ["out"], "proj/fn:latest")

        result = await step._get_endpoint_schema("ep-1", "my-model")

        assert result == (["f1"], ["out"], "proj/fn:latest")
        mock_db.get_model_endpoint.assert_not_called()

    async def test_cache_hit_with_none_schema_refreshes_from_db(self, monkeypatch):
        ep = self._make_ep(["f1"], ["out"], "proj/fn:latest")
        mock_db = self._mock_db(monkeypatch, ep)
        step = ProcessHTTPEvent(project="proj")
        step._schema_cache["ep-1"] = (None, None, "proj/fn:latest")

        result = await step._get_endpoint_schema("ep-1", "my-model")

        assert result == (["f1"], ["out"], "proj/fn:latest")
        mock_db.get_model_endpoint.assert_called_once()

    async def test_db_failure_propagates(self, monkeypatch):
        """Generic DB errors propagate from _get_endpoint_schema to the caller."""
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_endpoint.side_effect = Exception("connection error")
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda *a, **kw: mock_db)
        step = ProcessHTTPEvent(project="proj")

        with pytest.raises(Exception, match="connection error"):
            await step._get_endpoint_schema("ep-1", "my-model")

    async def test_not_found_error_propagates(self, monkeypatch):
        """MLRunNotFoundError from the DB is not swallowed — it propagates to the caller."""
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_endpoint.side_effect = mlrun.errors.MLRunNotFoundError(
            "endpoint not found"
        )
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda *a, **kw: mock_db)
        step = ProcessHTTPEvent(project="proj")

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            await step._get_endpoint_schema("ep-1", "my-model")

    async def test_expired_cache_entry_refreshed_from_db(self, monkeypatch):
        """After the TTL elapses the entry is evicted and the DB is called again."""
        from cachetools import TTLCache

        fake_time = [0.0]
        ep = self._make_ep(["f1"], ["out"], "proj/fn:latest")
        mock_db = self._mock_db(monkeypatch, ep)
        step = ProcessHTTPEvent(project="proj")

        # Replace the cache with a 1-second TTL driven by a fake timer.
        step._schema_cache = TTLCache(maxsize=100, ttl=1, timer=lambda: fake_time[0])

        await step._get_endpoint_schema("ep-1", "my-model")
        assert mock_db.get_model_endpoint.call_count == 1

        # Advance time past TTL — entry should be evicted on next access.
        fake_time[0] = 2.0

        await step._get_endpoint_schema("ep-1", "my-model")
        assert mock_db.get_model_endpoint.call_count == 2


class _MockContext:
    """Minimal serving context stub for unit-testing steps that call self.context.Response."""

    class Response:
        def __init__(self, body, content_type, status_code):
            self.body = body
            self.content_type = content_type
            self.status_code = status_code


class TestHTTPAckResponder:
    def _valid_event(self):
        return {
            EventFieldType.ENDPOINT_ID: "ep-123",
            EventFieldType.MODEL: "my-model",
            "request": {"inputs": [[1.0]]},
            "resp": {"outputs": [[0.8]]},
        }

    def _step(self):
        step = HTTPAckResponder()
        step.context = _MockContext()
        return step

    def test_valid_event_returns_202_accepted(self):
        import json

        step = self._step()
        result = step.do(self._valid_event())
        assert result.status_code == 202
        assert result.content_type == "application/json"
        body = json.loads(result.body)
        assert body["status"] == "accepted"
        assert body["endpoint_id"] == "ep-123"
        assert body["endpoint_name"] == "my-model"

    def test_error_sentinel_returns_400(self):
        step = self._step()
        result = step.do({_HTTP_ERROR_KEY: "missing required fields: inputs"})
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "missing required fields: inputs" in body["error"]

    def test_error_message_propagated(self):
        step = self._step()
        msg = "missing required fields: model_endpoint_name, outputs"
        result = step.do({_HTTP_ERROR_KEY: msg})
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "model_endpoint_name" in body["error"]


class TestGetModelMonitoringUrl:
    """Unit tests for mlrun.get_model_monitoring_url env-var caching logic."""

    _ENV_VAR = NuclioMonitoringEnvVars.MODEL_MONITORING_URL

    _ACTIVE_PROJECT_VAR = "MLRUN_ACTIVE_PROJECT"

    def setup_method(self):
        os.environ.pop(self._ENV_VAR, None)
        os.environ.pop(self._ACTIVE_PROJECT_VAR, None)

    def teardown_method(self):
        os.environ.pop(self._ENV_VAR, None)
        os.environ.pop(self._ACTIVE_PROJECT_VAR, None)

    def test_returns_env_var_without_db_call(self, monkeypatch: pytest.MonkeyPatch):
        """When the env var is already set the DB must not be called."""
        cached = "http://model-monitoring-stream.my-project.svc.cluster.local:8080"
        os.environ[self._ENV_VAR] = cached

        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == cached
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
        stream_url = "http://model-monitoring-stream.my-project.svc.cluster.local:8080"
        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        mock_db.get_model_monitoring_url.return_value = stream_url
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        mlrun.get_model_monitoring_url(project="my-project")
        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == stream_url
        mock_db.get_model_monitoring_url.assert_called_once()  # only the first call

    def test_returns_none_when_db_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """When the DB returns None (no HTTP trigger) the env var is not set."""
        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        mock_db.get_model_monitoring_url.return_value = None
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url is None
        assert self._ENV_VAR not in os.environ

    def test_raises_when_cached_url_project_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """MLRunInvalidArgumentError is raised when the cached URL belongs to a different project."""
        os.environ[self._ENV_VAR] = (
            "http://model-monitoring-stream.other-project.svc.cluster.local:8080"
        )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="my-project"):
            mlrun.get_model_monitoring_url(project="my-project")

    def test_no_error_when_cached_url_matches_project(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """No error is raised when the cached URL namespace label matches the project."""
        cached = "http://model-monitoring-stream.my-project.svc.cluster.local:8080"
        os.environ[self._ENV_VAR] = cached

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == cached

    def test_no_false_positive_for_project_name_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """'project' must not match a URL whose namespace is 'project-1' (substring false positive)."""
        os.environ[self._ENV_VAR] = (
            "http://model-monitoring-stream.project-1.svc.cluster.local:8080"
        )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="project"):
            mlrun.get_model_monitoring_url(project="project")

    def test_uses_active_project_when_no_project_given(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When project is omitted, MLRUN_ACTIVE_PROJECT is used."""
        os.environ[self._ACTIVE_PROJECT_VAR] = "active-project"
        mock_db = pytest.importorskip("unittest.mock").MagicMock()
        mock_db.get_model_monitoring_url.return_value = "http://stream/ingest"
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url()

        assert url == "http://stream/ingest"
        mock_db.get_model_monitoring_url.assert_called_once_with("active-project")

    def test_warns_when_no_project_given_and_cache_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When project is omitted and the URL is not cached, a warning is emitted."""
        mock = pytest.importorskip("unittest.mock")
        os.environ[self._ACTIVE_PROJECT_VAR] = "active-project"
        mock_db = mock.MagicMock()
        mock_db.get_model_monitoring_url.return_value = "http://stream/ingest"
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        with mock.patch("mlrun.run.logger") as mock_logger:
            mlrun.get_model_monitoring_url()

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "active-project" in str(call_args)

    def test_no_warning_when_project_explicitly_given(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When project is provided explicitly, the fallback warning is not emitted."""
        mock = pytest.importorskip("unittest.mock")
        mock_db = mock.MagicMock()
        mock_db.get_model_monitoring_url.return_value = "http://stream/ingest"
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        with mock.patch("mlrun.run.logger") as mock_logger:
            mlrun.get_model_monitoring_url(project="my-project")

        mock_logger.warning.assert_not_called()
