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
import logging
import typing
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import mlrun
import mlrun.artifacts
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.errors
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.serving.states
from mlrun.common.schemas import alert as alert_objects
from mlrun.common.schemas.model_monitoring import ResultData
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
)
from mlrun.model_monitoring.applications._application_steps import (
    _OTEL_BRANCH_SOURCE,
    _ApplicationErrorHandler,
    _PrepareMonitoringEvent,  # noqa: F401
    _PrepareOTelEvent,
    _PushToMonitoringWriter,
)
from mlrun.model_monitoring.applications.results import (
    _ModelMonitoringApplicationStats,
)
from mlrun.utils import Logger, logger


class TestEventPreparation:
    ENDPOINT_ID = "test-ep-id"
    ENDPOINT_NAME = "test-ep-name"
    APPLICATION_NAME = "test-app"
    ENDPOINT_UPDATED = mlrun.utils.now_date().isoformat()

    @classmethod
    @pytest.fixture
    def controller_event(cls) -> dict[str, typing.Any]:
        return {
            mm_constants.ApplicationEvent.ENDPOINT_ID: cls.ENDPOINT_ID,
            mm_constants.ApplicationEvent.ENDPOINT_NAME: cls.ENDPOINT_NAME,
            mm_constants.ApplicationEvent.APPLICATION_NAME: cls.APPLICATION_NAME,
            mm_constants.ApplicationEvent.ENDPOINT_UPDATED: cls.ENDPOINT_UPDATED,
        }

    @classmethod
    def test_prepare_monitoring_event(
        cls, controller_event: dict[str, typing.Any], tmp_path: Path
    ) -> None:
        with patch.object(
            mlrun.db.get_run_db(),
            "get_model_endpoint",
            Mock(
                return_value=mlrun.common.schemas.model_monitoring.ModelEndpoint(
                    metadata=mlrun.common.schemas.model_monitoring.ModelEndpointMetadata(
                        project="my-proj",
                        name="my-endpoint",
                    ),
                    spec=mlrun.common.schemas.ModelEndpointSpec(
                        function_name="my-func",
                        function_tag="my-tag",
                        monitoring_feature_set_uri=mlrun.utils.generate_object_uri(
                            project="my-proj", name="my-serving"
                        ),
                    ),
                    status=mlrun.common.schemas.model_monitoring.ModelEndpointStatus(),
                )
            ),
        ) as patch_get_model_endpoint:
            with patch.object(
                mlrun.db.get_run_db(),
                "get_project",
                Mock(
                    return_value=mlrun.projects.MlrunProject(
                        spec=mlrun.projects.ProjectSpec(artifact_path=str(tmp_path))
                    )
                ),
            ):
                logger.info(
                    "Set up a mock server with a `_PrepareMonitoringEvent` step"
                )

                fn = typing.cast(
                    mlrun.runtimes.ServingRuntime,
                    mlrun.code_to_function(
                        filename=__file__,
                        name="model-monitoring-context-preparation",
                        kind=mlrun.run.RuntimeKinds.serving,
                    ),
                )
                graph = fn.set_topology(mlrun.serving.states.StepKinds.flow)

                graph.to(
                    "_PrepareMonitoringEvent", application_name=cls.APPLICATION_NAME
                ).respond()
                server = fn.to_mock_server()
                monitoring_context = typing.cast(
                    mm_context.MonitoringApplicationContext,
                    server.test(body=controller_event),
                )

                logger.info("Test `monitoring_context` functionality")

                monitoring_context.logger.debug(
                    "Checking `get_endpoint_record` was called"
                )
                patch_get_model_endpoint.assert_called_once()

                monitoring_context.logger.debug("Logging an artifact")
                artifact = monitoring_context.log_artifact(
                    "my-app-data",
                    body=b"Sometimes, context is important.",
                    format="txt",
                    labels={"framework": "deepeval"},
                )

                monitoring_context.logger.debug("Checking logged artifact labels")
                assert {
                    "framework": "deepeval",
                    "mlrun/producer-type": "model-monitoring-app",
                    "mlrun/app-name": cls.APPLICATION_NAME,
                    "mlrun/endpoint-id": cls.ENDPOINT_ID,
                    "mlrun/endpoint-name": cls.ENDPOINT_NAME,
                }.items() <= artifact.labels.items()
                assert artifact.key == f"my-app-data-{cls.ENDPOINT_ID}", (
                    "By default monitoring context concat endpoint id to artifact key"
                )

                dataset = monitoring_context.log_dataset(
                    key="my-app-df",
                    df=pd.DataFrame({"a": [1, 2, 3]}),
                    labels={"framework": "deepeval"},
                )
                assert {
                    "framework": "deepeval",
                    "mlrun/producer-type": "model-monitoring-app",
                    "mlrun/app-name": cls.APPLICATION_NAME,
                    "mlrun/endpoint-id": cls.ENDPOINT_ID,
                    "mlrun/endpoint-name": cls.ENDPOINT_NAME,
                }.items() <= dataset.labels.items()
                assert dataset.key == f"my-app-df-{cls.ENDPOINT_ID}", (
                    "By default monitoring context concat endpoint id to dataset key"
                )
                server.wait_for_completion()
                monitoring_context.logger.debug("I'm done")


class Pusher:
    def __init__(self, filename: str) -> None:
        self.stream_filename = filename

    def push(self, data: list[dict[str, typing.Any]], partition_key: str) -> None:
        data = data[0]
        with open(self.stream_filename, "w") as json_file:
            json.dump(data, json_file)
            json_file.write("\n")


@pytest.fixture
def pusher(tmp_path: Path) -> Pusher:
    return Pusher(filename=f"{tmp_path}/test_stream.txt")


@pytest.fixture
def push_to_monitoring_writer():
    return _PushToMonitoringWriter(project="demo-project")


@pytest.fixture
def monitoring_context() -> mm_context.MonitoringApplicationContext:
    mock_monitoring_context = Mock(spec=mm_context.MonitoringApplicationContext)
    mock_monitoring_context.log_stream = Logger(
        name="test_data_drift_app", level=logging.DEBUG
    )
    mock_monitoring_context._artifacts_manager = Mock(
        spec=mlrun.artifacts.manager.ArtifactManager
    )
    mock_monitoring_context.application_name = "test_data_drift_app"
    mock_monitoring_context.endpoint_id = "test_endpoint_id"
    mock_monitoring_context.endpoint_name = "test_endpoint_name"
    mock_monitoring_context.start_infer_time = pd.Timestamp(
        "2022-01-01 00:00:00.000000"
    )
    mock_monitoring_context.end_infer_time = pd.Timestamp("2022-01-01 00:00:00.000000")
    mock_monitoring_context.sample_df_stats = {}
    return mock_monitoring_context


@patch("mlrun.model_monitoring.helpers.get_output_stream")
def test_push_result_to_monitoring_writer_stream(
    mock_get_output_stream: Mock,
    pusher: Pusher,
    push_to_monitoring_writer: _PushToMonitoringWriter,
    monitoring_context: mm_context.MonitoringApplicationContext,
):
    """
    Test that the `_PushToMonitoringWriter` step pushes the results to the monitoring writer stream. In addition,
    test that the extra data is not pushed to the stream if it exceeds the maximum size of 998 characters.
    """
    mock_get_output_stream.return_value = pusher
    results = [
        ModelMonitoringApplicationResult(
            name="res1",
            value=1,
            status=mm_constants.ResultStatusApp.detected,
            extra_data={"extra_data": "extra_data"},
            kind=mm_constants.ResultKindApp.data_drift,
        ),
        ModelMonitoringApplicationResult(
            name="res2",
            value=2,
            status=mm_constants.ResultStatusApp.detected,
            extra_data={"extra_data": "extra_data" * 1000},
            kind=mm_constants.ResultKindApp.data_drift,
        ),
        ModelMonitoringApplicationMetric(name="met", value=2),
    ]

    for result in results:
        push_to_monitoring_writer.do(([result], monitoring_context))

        with open(pusher.stream_filename) as file:
            for line in file:
                loaded_data = json.loads(line.strip())
            if isinstance(result, ModelMonitoringApplicationResult):
                event_kind = mm_constants.WriterEventKind.RESULT
                result = result.to_dict()
                data_from_file = json.loads(loaded_data["data"])

                if len(result["result_extra_data"]) <= 998:
                    assert (
                        data_from_file[ResultData.RESULT_EXTRA_DATA]
                        == result[ResultData.RESULT_EXTRA_DATA]
                    )
                else:
                    assert (
                        data_from_file[ResultData.RESULT_EXTRA_DATA]
                        != result[ResultData.RESULT_EXTRA_DATA]
                    )
                    result["extra_data"] = "{}"
            else:
                event_kind = mm_constants.WriterEventKind.METRIC
                result = result.to_dict()

            assert loaded_data == {
                "application_name": "test_data_drift_app",
                "endpoint_id": "test_endpoint_id",
                "endpoint_name": "test_endpoint_name",
                "start_infer_time": "2022-01-01 00:00:00.000000",
                "end_infer_time": "2022-01-01 00:00:00.000000",
                "event_kind": event_kind.value,
                "data": json.dumps(result),
            }


class TestPrepareOTelEvent:
    PROJECT = "my-proj"
    APP = "my-app"
    EP_ID = "ep-1234"
    EP_NAME = "ep-name"
    BASE_ATTRS = {
        "project": PROJECT,
        "app.name": APP,
        "function.name": APP,
        "endpoint.uid": EP_ID,
        "endpoint.name": EP_NAME,
    }

    @classmethod
    @pytest.fixture
    def app_ctx(cls) -> Mock:
        ctx = Mock(spec=mm_context.MonitoringApplicationContext)
        ctx.project_name = cls.PROJECT
        ctx.application_name = cls.APP
        ctx.endpoint_id = cls.EP_ID
        ctx.endpoint_name = cls.EP_NAME
        return ctx

    @staticmethod
    def _by_name(metrics: list[dict[str, typing.Any]]) -> dict[str, dict]:
        return {m["metric_name"]: m for m in metrics}

    @classmethod
    def test_result_and_metric_shape(cls, app_ctx: Mock) -> None:
        """Results carry `result.kind` + `result.status` and the
        `mlrun.model_monitoring.result.` prefix; metrics use the
        `.metric.` prefix and don't get those extra attributes."""
        results = [
            ModelMonitoringApplicationResult(
                name="general_drift",
                value=0.42,
                kind=mm_constants.ResultKindApp.data_drift,
                status=mm_constants.ResultStatusApp.detected,
            ),
            ModelMonitoringApplicationMetric(name="hellinger", value=0.1),
        ]
        event = _PrepareOTelEvent().do((results, app_ctx))
        by_name = cls._by_name(event["metrics"])

        result_entry = by_name["mlrun.model_monitoring.result.general_drift"]
        assert result_entry == {
            "metric_name": "mlrun.model_monitoring.result.general_drift",
            "value": 0.42,
            "type": "gauge",
            "attributes": {
                **cls.BASE_ATTRS,
                "result.kind": "data_drift",
                "result.status": "detected",
            },
        }
        metric_entry = by_name["mlrun.model_monitoring.metric.hellinger"]
        assert metric_entry == {
            "metric_name": "mlrun.model_monitoring.metric.hellinger",
            "value": 0.1,
            "type": "gauge",
            "attributes": cls.BASE_ATTRS,
        }

    @classmethod
    def test_stats_entries_skipped(cls, app_ctx: Mock) -> None:
        """Histogram drift stats are a side payload and have no OTel
        instrument — they're filtered out."""
        results = [
            _ModelMonitoringApplicationStats(
                name=mm_constants.StatsKind.CURRENT_STATS,
                timestamp="2026-05-14T00:00:00",
                stats={"feat": {"mean": 0.5}},
            ),
            ModelMonitoringApplicationMetric(name="some_metric", value=1.0),
        ]
        event = _PrepareOTelEvent().do((results, app_ctx))
        assert [m["metric_name"] for m in event["metrics"]] == [
            "mlrun.model_monitoring.metric.some_metric"
        ]

    @classmethod
    def test_none_attributes_stripped(cls) -> None:
        """The OTel SDK warns on None-valued attributes; the step must
        drop them rather than forward them."""
        ctx = Mock(spec=mm_context.MonitoringApplicationContext)
        ctx.project_name = cls.PROJECT
        ctx.application_name = cls.APP
        ctx.endpoint_id = None
        ctx.endpoint_name = None
        results = [ModelMonitoringApplicationMetric(name="m", value=1.0)]
        event = _PrepareOTelEvent().do((results, ctx))
        attrs = event["metrics"][0]["attributes"]
        assert "endpoint.uid" not in attrs
        assert "endpoint.name" not in attrs
        assert attrs == {
            "project": cls.PROJECT,
            "app.name": cls.APP,
            "function.name": cls.APP,
        }

    @classmethod
    def test_empty_results(cls, app_ctx: Mock) -> None:
        assert _PrepareOTelEvent().do(([], app_ctx)) == {"metrics": []}


class TestApplicationErrorHandler:
    PROJECT = "my-proj"
    APP = "my-app"

    @staticmethod
    def _make_event(
        *,
        origin_state: str | None,
        body: typing.Any = None,
        error: Exception | None = None,
    ) -> Mock:
        event = Mock()
        event.body = body
        # Raise + catch so event.error has a real traceback for
        # traceback.format_exception() to render.
        try:
            raise error or RuntimeError("kaboom")
        except Exception as e:
            event.error = e
        event.timestamp = "2026-05-14T00:00:00"
        # None is a valid value: handler dispatch is "in frozenset of
        # branch names", and None is not in that set, so the main-app
        # path runs — covers the defensive case.
        event.origin_state = origin_state
        return event

    @classmethod
    def _captured_event(cls, generate_event: Mock) -> alert_objects.Event:
        assert generate_event.called, "Handler did not generate an alert event"
        return generate_event.call_args.kwargs["event_data"]

    @classmethod
    def test_main_app_failure_uses_body_fields(cls) -> None:
        """Failure on the main app step: body is the controller event;
        endpoint id and app name come from there. Alert entity id is
        plain `<project>_<app>` — no source suffix."""
        handler = _ApplicationErrorHandler(project=cls.PROJECT)
        body = Mock()
        body.application_name = cls.APP
        body.endpoint_id = "ep-1234"
        event = cls._make_event(origin_state="DemoMonitoringApp", body=body)

        with patch("mlrun.get_run_db") as get_db:
            handler.do(event)
        alert = cls._captured_event(get_db.return_value.generate_event)

        assert alert.entity.ids == [f"{cls.PROJECT}_{cls.APP}"]
        assert alert.value_dict["Application Class"] == cls.APP
        assert alert.value_dict["Endpoint ID"] == "ep-1234"
        # No source tag on regular app failures — keeps backward compat
        # with existing alert configs.
        assert "Source" not in alert.value_dict

    @classmethod
    @pytest.mark.parametrize(
        "origin_state", ["PrepareOTelEvent", "OTelMetricsExporter"]
    )
    def test_otel_branch_failure_is_tagged(cls, origin_state: str) -> None:
        """Failure on the OTel branch (either prep or exporter):
        application_name is pinned on the handler (body shape doesn't
        carry it); alert entity id is suffixed with `_otel_exporter` so
        alert configs can route this failure mode separately."""
        handler = _ApplicationErrorHandler(
            project=cls.PROJECT, application_name=cls.APP
        )
        # Different bodies per step — neither has application_name.
        body = [Mock()] if origin_state == "PrepareOTelEvent" else {"metrics": []}
        event = cls._make_event(origin_state=origin_state, body=body)

        with patch("mlrun.get_run_db") as get_db:
            handler.do(event)
        alert = cls._captured_event(get_db.return_value.generate_event)

        expected_id = f"{cls.PROJECT}_{cls.APP}_{_OTEL_BRANCH_SOURCE}"
        assert alert.entity.ids == [expected_id]
        assert alert.value_dict["Source"] == _OTEL_BRANCH_SOURCE
        assert alert.value_dict["Application Class"] == cls.APP
        # Endpoint id isn't reliably available on the branch event body.
        assert alert.value_dict["Endpoint ID"] is None

    @classmethod
    def test_otel_branch_failure_without_pinned_app_name_raises(cls) -> None:
        """If wiring forgot to pin application_name on the handler, an
        OTel-branch failure has no way to identify the application —
        raise loudly rather than emit a malformed alert."""
        handler = _ApplicationErrorHandler(project=cls.PROJECT)
        event = cls._make_event(origin_state="OTelMetricsExporter", body={})

        with patch("mlrun.get_run_db"):
            with pytest.raises(
                mlrun.errors.MLRunRuntimeError, match="application_name"
            ):
                handler.do(event)

    @classmethod
    def test_main_app_failure_works_without_origin_state(cls) -> None:
        """Defensive: if storey didn't set origin_state for some reason,
        the handler still works for the main-app path (body has the
        controller event fields)."""
        handler = _ApplicationErrorHandler(project=cls.PROJECT)
        body = Mock()
        body.application_name = cls.APP
        body.endpoint_id = "ep-x"
        event = cls._make_event(origin_state=None, body=body)

        with patch("mlrun.get_run_db") as get_db:
            handler.do(event)
        alert = cls._captured_event(get_db.return_value.generate_event)
        assert alert.entity.ids == [f"{cls.PROJECT}_{cls.APP}"]
        assert "Source" not in alert.value_dict
