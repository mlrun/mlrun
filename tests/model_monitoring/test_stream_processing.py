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

import datetime
import json
import os
import unittest.mock

import pandas as pd
import pyarrow.parquet as pq
import pytest
import storey

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
    ProcessBeforeParquet,
    ProcessEndpointEvent,
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


def _find_step_call(graph_mock: unittest.mock.Mock, step_name: str):
    for call in graph_mock.add_step.call_args_list:
        if call.kwargs.get("name") == step_name:
            return call
    raise AssertionError(
        f"graph.add_step was not called with name={step_name!r}; "
        f"calls were: {graph_mock.add_step.call_args_list}"
    )


def _make_timescaledb_connector(monkeypatch: pytest.MonkeyPatch, project_name: str):
    monkeypatch.setattr(mlrun.mlconf, "system_id", "123456")
    return mlrun.model_monitoring.get_tsdb_connector(
        project=project_name,
        profile=DatastoreProfilePostgreSQL(
            name="postgresql-tsdb-test",
            user="testuser",
            password="testpass",
            host="localhost",
            port=5432,
            database="postgres",
        ),
    )


def test_timescaledb_stream_steps_read_max_events_and_flush_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TimescaleDB predictions target picks up mlconf values when no kwargs."""
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph, "max_events", 4242
    )
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph,
        "flush_after_seconds",
        77,
    )

    tsdb_connector = _make_timescaledb_connector(
        monkeypatch, project_name="test-stream-config-read"
    )
    graph = unittest.mock.Mock()

    tsdb_connector.apply_monitoring_stream_steps(graph)

    call = _find_step_call(graph, "TimescaleDBTarget")
    assert call.kwargs["max_events"] == 4242
    assert call.kwargs["flush_after_seconds"] == 77


def test_timescaledb_stream_steps_kwargs_override_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwargs override mlconf values for the predictions target."""
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph, "max_events", 4242
    )
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph,
        "flush_after_seconds",
        77,
    )

    tsdb_connector = _make_timescaledb_connector(
        monkeypatch, project_name="test-stream-kwargs-override"
    )
    graph = unittest.mock.Mock()

    tsdb_connector.apply_monitoring_stream_steps(
        graph,
        tsdb_batching_max_events=11,
        tsdb_batching_timeout_secs=22,
    )

    call = _find_step_call(graph, "TimescaleDBTarget")
    assert call.kwargs["max_events"] == 11
    assert call.kwargs["flush_after_seconds"] == 22


def test_timescaledb_handle_model_error_reads_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TimescaleDB errors target picks up mlconf values when no kwargs."""
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph, "max_events", 555
    )
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph,
        "flush_after_seconds",
        66,
    )

    tsdb_connector = _make_timescaledb_connector(
        monkeypatch, project_name="test-errors-config-read"
    )
    graph = unittest.mock.Mock()

    tsdb_connector.handle_model_error(graph)

    call = _find_step_call(graph, "timescaledb_error")
    assert call.kwargs["max_events"] == 555
    assert call.kwargs["flush_after_seconds"] == 66


def test_timescaledb_handle_model_error_kwargs_override_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwargs override mlconf values for the errors target."""
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph, "max_events", 555
    )
    monkeypatch.setattr(
        mlrun.mlconf.model_endpoint_monitoring.stream_graph,
        "flush_after_seconds",
        66,
    )

    tsdb_connector = _make_timescaledb_connector(
        monkeypatch, project_name="test-errors-kwargs-override"
    )
    graph = unittest.mock.Mock()

    tsdb_connector.handle_model_error(
        graph,
        tsdb_batching_max_events=7,
        tsdb_batching_timeout_secs=8,
    )

    call = _find_step_call(graph, "timescaledb_error")
    assert call.kwargs["max_events"] == 7
    assert call.kwargs["flush_after_seconds"] == 8


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

    async def test_zero_request_id_preserved(self, monkeypatch):
        step = self._step(monkeypatch)
        result = await step.do(
            {
                "model_endpoint_uid": "ep-1",
                "model_endpoint_name": "my-model",
                "inputs": [[1.0]],
                "outputs": [[0.8]],
                EventFieldType.REQUEST_ID: 0,
            }
        )
        assert result["request"]["id"] == 0

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


class TestProcessEndpointEvent:
    """ProcessEndpointEvent.do() request-id extraction.

    A request id of 0 is a valid id (e.g. execute_graph assigns row-index
    event ids) and must not be treated as missing.
    """

    @staticmethod
    def _event(request: dict, resp: dict) -> storey.Event:
        return storey.Event(
            body={
                EventFieldType.FUNCTION_URI: "test-project/fn",
                EventFieldType.MODEL: "my-model",
                "model_class": "MyModel",
                "when": "2025-07-24 05:00:10.000000",
                EventFieldType.ENDPOINT_ID: "ep-1",
                "microsec": 5,
                "request": request,
                "resp": resp,
            }
        )

    async def _do(self, request: dict, resp: dict) -> storey.Event:
        step = ProcessEndpointEvent(project="test-project")
        step.endpoints.add("ep-1")
        return await step.do(self._event(request=request, resp=resp))

    async def test_zero_request_id_is_kept(self):
        result = await self._do(
            request={"id": 0, "inputs": [[1.0, 2.0]]}, resp={"outputs": [[0.8]]}
        )
        assert result.body is not None
        assert result.body[0][EventFieldType.REQUEST_ID] == 0

    async def test_request_id_falls_back_to_resp_id(self):
        result = await self._do(
            request={"inputs": [[1.0, 2.0]]}, resp={"id": "resp-1", "outputs": [[0.8]]}
        )
        assert result.body is not None
        assert result.body[0][EventFieldType.REQUEST_ID] == "resp-1"

    async def test_missing_request_id_drops_event(self):
        result = await self._do(
            request={"inputs": [[1.0, 2.0]]}, resp={"outputs": [[0.8]]}
        )
        assert result.body is None


class TestProcessBeforeParquet:
    """ProcessBeforeParquet.do() must emit events with a Parquet-stable schema.

    On a schema-less endpoint the first event carries feature_names=None and later
    events carry the names MapFeatureNames generated, so the Parquet target inferred
    Arrow type null for one file and list<string> for the next. Reading the partition
    then failed with ArrowNotImplementedError depending on file discovery order
    (ML-12998). The same applies to the dict-shaped columns, whose inferred struct
    type varies with the keys present.
    """

    _TIMESTAMP = datetime.datetime(2026, 8, 11, 20, 37, 9, tzinfo=datetime.UTC)

    @classmethod
    def _event(
        cls,
        feature_names: list[str] | None,
        label_names: list[str] | None,
        labels: dict | None = None,
        metrics: dict | None = None,
        entities: dict | None = None,
    ) -> dict:
        """Build an event shaped like the one MapFeatureNames emits."""
        return {
            EventFieldType.ENDPOINT_ID: "ep-1",
            EventFieldType.ENDPOINT_NAME: "my-model",
            EventFieldType.TIMESTAMP: cls._TIMESTAMP,
            EventFieldType.REQUEST_ID: "req-1",
            EventFieldType.LATENCY: 5.0,
            EventFieldType.FEATURES: [1.0, 2.0],
            EventFieldType.NAMED_FEATURES: {"f0": 1.0, "f1": 2.0},
            EventFieldType.PREDICTION: [0.8],
            EventFieldType.NAMED_PREDICTIONS: {"p0": 0.8},
            EventFieldType.FEATURE_NAMES: feature_names,
            EventFieldType.LABEL_NAMES: label_names,
            EventFieldType.LABELS: labels,
            EventFieldType.METRICS: metrics,
            EventFieldType.ENTITIES: entities,
            "f0": 1.0,
            "f1": 2.0,
            "p0": 0.8,
        }

    @classmethod
    def _schema_less_event(cls) -> dict:
        """First event on an endpoint created without input/output schema."""
        return cls._event(feature_names=None, label_names=None)

    @classmethod
    def _schema_resolved_event(cls) -> dict:
        """Later event, after MapFeatureNames persisted the generated names."""
        return cls._event(feature_names=["f0", "f1"], label_names=["p0"])

    @pytest.mark.parametrize(
        "feature_names,label_names",
        [(None, None), (["f0", "f1"], ["p0"])],
        ids=["schema_less", "schema_resolved"],
    )
    def test_transient_fields_are_removed(self, feature_names, label_names):
        result = ProcessBeforeParquet().do(
            self._event(feature_names=feature_names, label_names=label_names)
        )

        # feature_names/label_names are consumed by MapFeatureNames and must not
        # reach the target - they are what flips between null and list<string>.
        for key in [
            EventFieldType.FEATURE_NAMES,
            EventFieldType.LABEL_NAMES,
            EventFieldType.FEATURES,
            EventFieldType.NAMED_FEATURES,
            EventFieldType.PREDICTION,
            EventFieldType.NAMED_PREDICTIONS,
        ]:
            assert key not in result

        # The mapped name-value pairs are what the target actually stores.
        assert result["f0"] == 1.0
        assert result["p0"] == 0.8

    @pytest.mark.parametrize(
        "labels,metrics,entities",
        [
            (None, None, None),
            ({}, {}, {}),
            ({"l1": "a"}, {"m1": 1.0}, {"e1": "x"}),
        ],
        ids=["none", "empty", "populated"],
    )
    def test_dict_fields_are_serialized(self, labels, metrics, entities):
        result = ProcessBeforeParquet().do(
            self._event(
                feature_names=["f0", "f1"],
                label_names=["p0"],
                labels=labels,
                metrics=metrics,
                entities=entities,
            )
        )

        for key, original in [
            (EventFieldType.LABELS, labels),
            (EventFieldType.METRICS, metrics),
            (EventFieldType.ENTITIES, entities),
        ]:
            assert isinstance(result[key], str), key
            assert json.loads(result[key]) == (original or {}), key

    def test_entities_are_still_split_into_columns(self):
        result = ProcessBeforeParquet().do(
            self._event(
                feature_names=["f0", "f1"], label_names=["p0"], entities={"e1": "x"}
            )
        )

        assert result["e1"] == "x"

    def test_parquet_schema_is_stable_across_flushes(self, tmp_path):
        """The end-to-end reproduction: two flushes must stay readable as one dataset.

        max_events=1 forces a file per event, mirroring the reported run where the
        two events were flushed separately and landed in the same hour partition.
        """
        target_dir = tmp_path / "parquet"

        flow = storey.build_flow(
            [
                storey.SyncEmitSource(key_field=EventFieldType.ENDPOINT_ID),
                ProcessBeforeParquet(),
                # Mirrors apply_parquet_target() in the monitoring serving graph.
                storey.ParquetTarget(
                    path=str(target_dir),
                    index_cols=[EventFieldType.ENDPOINT_ID],
                    partition_cols=["$key", "$year", "$month", "$day", "$hour"],
                    time_field=EventFieldType.TIMESTAMP,
                    infer_columns_from_data=True,
                    max_events=1,
                ),
            ]
        )

        controller = flow.run()
        controller.emit(self._schema_less_event())
        controller.emit(self._schema_resolved_event())
        controller.terminate()
        controller.await_termination()

        partition_dirs = {
            path.parent for path in target_dir.rglob("*.parquet") if path.is_file()
        }
        assert len(partition_dirs) == 1, (
            f"expected both events in one partition, got {partition_dirs}"
        )
        partition_dir = partition_dirs.pop()
        files = sorted(partition_dir.glob("*.parquet"))
        assert len(files) == 2, "expected one file per event"

        # PyArrow adopts the schema of whichever file it discovers first and casts the
        # rest to it. Asserting the per-file schemas match is what makes this test
        # deterministic: reading the directory only failed when the null-typed file
        # happened to sort first, which is what made the reported failure intermittent.
        schemas = [pq.read_schema(file) for file in files]
        assert schemas[0].equals(schemas[1]), (
            f"parquet files disagree on schema:\n{schemas[0]}\nvs\n{schemas[1]}"
        )

        df = pd.read_parquet(partition_dir)

        assert len(df) == 2
        assert EventFieldType.FEATURE_NAMES not in df.columns
        assert EventFieldType.LABEL_NAMES not in df.columns


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

    @staticmethod
    def _stream_url(project: str) -> str:
        """Build a realistic nuclio model-monitoring-stream service URL for *project*."""
        return (
            f"http://nuclio-{project}-model-monitoring-stream"
            f".default-tenant.svc.cluster.local:8080"
        )

    def setup_method(self):
        os.environ.pop(self._ENV_VAR, None)
        os.environ.pop(self._ACTIVE_PROJECT_VAR, None)

    def teardown_method(self):
        os.environ.pop(self._ENV_VAR, None)
        os.environ.pop(self._ACTIVE_PROJECT_VAR, None)

    def test_returns_env_var_without_db_call(self, monkeypatch: pytest.MonkeyPatch):
        """When the env var is already set the DB must not be called."""
        cached = self._stream_url("my-project")
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
        stream_url = self._stream_url("my-project")
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

    def test_refreshes_cache_when_cached_url_project_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When the cached URL belongs to a different project the cache is refreshed from the DB and a warning is
        logged."""
        os.environ[self._ENV_VAR] = self._stream_url("other-project")
        refreshed = self._stream_url("my-project")
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_monitoring_url.return_value = refreshed
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        with unittest.mock.patch("mlrun.projects.project.logger") as mock_logger:
            url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == refreshed
        assert os.environ[self._ENV_VAR] == refreshed
        mock_db.get_model_monitoring_url.assert_called_once_with("my-project")
        mock_logger.warning.assert_called_once()
        assert "my-project" in str(mock_logger.warning.call_args)

    def test_no_error_when_cached_url_matches_project(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """No refetch happens when the cached URL's service segment encodes the requested project."""
        cached = self._stream_url("my-project")
        os.environ[self._ENV_VAR] = cached
        mock_db = unittest.mock.MagicMock()
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == cached
        mock_db.get_model_monitoring_url.assert_not_called()

    def test_no_false_positive_for_project_name_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """'project' must not match a URL encoding 'project-1' (substring false positive) — cache is refreshed."""
        os.environ[self._ENV_VAR] = self._stream_url("project-1")
        refreshed = self._stream_url("project")
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_monitoring_url.return_value = refreshed
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="project")

        assert url == refreshed
        mock_db.get_model_monitoring_url.assert_called_once_with("project")

    def test_refreshes_when_cached_url_does_not_match_nuclio_pattern(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A cached URL that doesn't follow nuclio-<project>-model-monitoring-stream
        (e.g. DNS-truncated with a hash) is treated as a mismatch and refreshed."""
        os.environ[self._ENV_VAR] = (
            "http://nuclio-very-long-project-name-mod-abc12345"
            ".default-tenant.svc.cluster.local:8080"
        )
        refreshed = self._stream_url("my-project")
        mock_db = unittest.mock.MagicMock()
        mock_db.get_model_monitoring_url.return_value = refreshed
        monkeypatch.setattr(mlrun.db, "get_run_db", lambda: mock_db)

        url = mlrun.get_model_monitoring_url(project="my-project")

        assert url == refreshed
        mock_db.get_model_monitoring_url.assert_called_once_with("my-project")

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

        with mock.patch("mlrun.projects.project.logger") as mock_logger:
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

        with mock.patch("mlrun.projects.project.logger") as mock_logger:
            mlrun.get_model_monitoring_url(project="my-project")

        mock_logger.warning.assert_not_called()
