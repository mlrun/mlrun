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

from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import mlrun
import mlrun.utils
from mlrun.common.schemas.model_monitoring import ResultKindApp, ResultStatusApp
from mlrun.datastore.datastore_profile import DatastoreProfileKafkaStream
from mlrun.model_monitoring.applications import (
    ExistingDataHandling,
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
    MonitoringApplicationContext,
)


class NoOpApp(ModelMonitoringApplicationBase):
    def do_tracking(self, monitoring_context: MonitoringApplicationContext):
        pass


class InProgressApp0(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info(
            "This test app is failing on purpose - ignore the failure!",
            project=monitoring_context.project_name,
        )
        raise ValueError


class InProgressApp1(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info(
            "It should work now",
            project=monitoring_context.project_name,
        )
        return ModelMonitoringApplicationResult(
            name="res0",
            value=0,
            status=ResultStatusApp.irrelevant,
            kind=ResultKindApp.mm_app_anomaly,
        )


class ModelEndpointAccessApp(ModelMonitoringApplicationBase):
    def do_tracking(self, monitoring_context: MonitoringApplicationContext) -> None:
        monitoring_context.logger.info(
            "Accessing the model endpoint",
            project=monitoring_context.project_name,
        )
        model_endpoint = monitoring_context.model_endpoint
        monitoring_context.logger.info(
            "Model endpoint labels",
            labels=model_endpoint.metadata.labels,
        )


class SampleDFAccessApp(ModelMonitoringApplicationBase):
    def do_tracking(self, monitoring_context: MonitoringApplicationContext) -> None:
        monitoring_context.logger.info(
            "Accessing the model endpoint's sample data",
            project=monitoring_context.project_name,
        )
        sample_df = monitoring_context.sample_df
        assert sample_df is not None
        monitoring_context.logger.info(
            "Read the sample data",
            sample_df=sample_df,
        )


@pytest.mark.filterwarnings("error")
def test_no_deprecation_instantiation() -> None:
    NoOpApp()


class TestEvaluate:
    @staticmethod
    @pytest.fixture(autouse=True)
    def _set_project() -> Iterator[None]:
        project = mlrun.get_or_create_project("test", allow_cross_project=True)
        with patch.object(
            project, "get_function", Mock(side_effect=mlrun.errors.MLRunNotFoundError)
        ):
            with patch("mlrun.db.nopdb.NopDB.get_project", Mock(return_value=project)):
                yield

    @staticmethod
    def test_local_no_params() -> None:
        func_name = "test-app"
        run = InProgressApp0.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "created"  # Should be "error", see ML-8507
        run = InProgressApp1.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "completed"
        assert run.status.results == {
            "return": {
                "result_name": "res0",
                "result_value": 0.0,
                "result_kind": 4,
                "result_status": -1,
                "result_extra_data": "{}",
            }
        }, "The run results are different than expected"

    @staticmethod
    def test_model_endpoint_blocked(capsys: pytest.CaptureFixture) -> None:
        """Test that the logs contain the error message about the blocked model endpoint access"""
        run = ModelEndpointAccessApp.evaluate(func_path=__file__)
        assert run.state() == "created"  # Should be "error", see ML-8507
        captured = capsys.readouterr()
        assert (
            "mlrun.errors.MLRunValueError: You have NOT provided the model endpoint's name and ID: "
            "`endpoint_name`=None and `endpoint_id`=None, "
            "but you have tried to access `monitoring_context.model_endpoint`"
            in captured.out
        ), "The error message is different than expected or was not captured"

    @staticmethod
    def test_invalid_sample_df_access(capsys: pytest.CaptureFixture) -> None:
        """Test that the logs contain the error message about sample data access"""
        run = SampleDFAccessApp.evaluate(func_path=__file__)
        assert run.state() == "created"  # Should be "error", see ML-8507
        captured = capsys.readouterr()
        assert (
            "You have tried to access `monitoring_context.sample_df`, but have not provided it directly"
            in captured.out
        ), "The error message is different than expected or was not captured"

    @staticmethod
    @pytest.mark.parametrize("method", ["to_job", "evaluate"])
    def test_valid_sample_df_access(
        method: str, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        project = mlrun.get_or_create_project(
            "local-test-sample-df", context=str(tmp_path)
        )
        project.artifact_path = str(tmp_path)
        sample_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_artifact_path = project.log_dataset("sample-df", df=sample_df).target_path

        if method == "to_job":
            job = SampleDFAccessApp.to_job(func_path=__file__)
            run = job.run(local=True, inputs={"sample_data": ds_artifact_path})
        elif method == "evaluate":
            run = SampleDFAccessApp.evaluate(
                func_path=__file__, run_local=True, sample_data=ds_artifact_path
            )
        else:
            raise NotImplementedError

        assert run.state() == "completed"
        captured = capsys.readouterr()
        assert (
            "You have tried to access `monitoring_context.sample_df`, but have not provided it directly"
            not in captured.out
        ), "The captured error was not expected"

        assert (
            "Read the sample data" in captured.out
        ), "The expected log message was not found in the captured output"

    @staticmethod
    @pytest.mark.parametrize(
        ("endpoints", "start", "end", "run_local", "write_output", "error_msg"),
        [
            (
                [("ep-name", "ep-uid")],
                datetime(2025, 5, 3),
                datetime(2025, 5, 4),
                False,
                True,
                "`stream_profile` is relevant only when running locally",
            ),
            (
                [("ep-name", "ep-uid")],
                datetime(2025, 5, 3),
                datetime(2025, 5, 4),
                True,
                False,
                "`stream_profile` is relevant only when writing the outputs",
            ),
            (
                None,
                datetime(2025, 5, 3),
                datetime(2025, 5, 4),
                False,
                True,
                "Custom `start` and `end` times .+ supported only with endpoints data",
            ),
            (
                None,
                None,
                None,
                False,
                False,
                "or passing `stream_profile` are supported only with endpoints data",
            ),
        ],
    )
    def test_invalid_params(
        endpoints: Optional[list[tuple[str, str]]],
        start: Optional[datetime],
        end: Optional[datetime],
        run_local: bool,
        write_output: bool,
        error_msg: str,
    ) -> None:
        with pytest.raises(mlrun.errors.MLRunValueError, match=error_msg):
            ModelEndpointAccessApp.evaluate(
                func_path=__file__,
                endpoints=endpoints,
                start=start,
                end=end,
                run_local=run_local,
                write_output=write_output,
                stream_profile=DatastoreProfileKafkaStream(
                    name="should-not-be-passed-on-remote",
                    brokers=["broker-address:9092"],
                    topics=[],
                ),
            )

    @staticmethod
    def test_invalid_infra(capsys: pytest.CaptureFixture) -> None:
        ModelEndpointAccessApp.evaluate(
            func_path=__file__,
            endpoints=[("ep-name", "ep-uid")],
            start=datetime(2025, 5, 3),
            end=datetime(2025, 5, 4),
            run_local=True,
            write_output=True,
            stream_profile=DatastoreProfileKafkaStream(
                name="should-not-be-passed-on-remote",
                brokers=["broker-address:9092"],
                topics=[],
            ),
        )
        captured = capsys.readouterr()
        assert (
            "Writing outputs to the databases is blocked as the model monitoring infrastructure is disabled.\n"
            "To unblock, enable model monitoring with `project.enable_model_monitoring()`."
            in captured.out
        ), "The error message is different than expected or was not captured"

    @staticmethod
    @pytest.mark.parametrize(
        ("pass_sample", "pass_reference"), [(True, False), (False, True), (True, True)]
    )
    def test_invalid_custom_dataframe_with_write_output(
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        pass_sample: bool,
        pass_reference: bool,
    ) -> None:
        """write_output=True with sample_data/reference_data should be blocked"""
        project = mlrun.get_or_create_project(
            "local-test-sample-df", context=str(tmp_path)
        )
        project.artifact_path = str(tmp_path)

        if pass_sample:
            sample_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            sample_artifact_path = project.log_dataset(
                "sample-df", df=sample_df
            ).target_path
        else:
            sample_artifact_path = None

        if pass_reference:
            reference_df = pd.DataFrame({"a": [1, 0, 1], "b": [5, 5, 6]})
            reference_artifact_path = project.log_dataset(
                "reference-df", df=reference_df
            ).target_path
        else:
            reference_artifact_path = None

        ModelEndpointAccessApp.evaluate(
            func_path=__file__,
            endpoints=[("ep-name", "ep-uid")],
            sample_data=sample_artifact_path,
            reference_data=reference_artifact_path,
            start=datetime(2025, 5, 3),
            end=datetime(2025, 5, 4),
            run_local=True,
            write_output=True,
            stream_profile=DatastoreProfileKafkaStream(
                name="kafka-stream", brokers=["broker-address:9092"], topics=[]
            ),
        )
        captured = capsys.readouterr()
        assert (
            "Writing the results of an application to the TSDB is possible only when "
            "working with endpoints, without any custom data-frame input"
            in captured.out
        ), "The error message is different than expected or was not captured"


@pytest.mark.parametrize(
    ("start", "end", "base_period", "expectation"),
    [
        (None, None, None, does_not_raise()),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            None,
            does_not_raise(),
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            0,
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="`base_period` must be a nonnegative integer .*",
            ),
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1).isoformat(),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            None,
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="The start and end times must either both include time zone information or both be naive",
            ),
        ),
    ],
)
def test_window_generator_validation(
    start: Optional[str],
    end: Optional[str],
    base_period: Optional[int],
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        next(
            ModelMonitoringApplicationBase._window_generator(
                start=start,
                end=end,
                base_period=base_period,
                application_schedules=None,
                endpoint_id="",
                application_name="",
                existing_data_handling=ExistingDataHandling.fail_on_overlap,
            )
        )


@pytest.mark.parametrize(
    ("start", "end", "base_period", "expected_windows"),
    [
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            None,
            [
                (
                    datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
                ),
            ],
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1),
            datetime(2008, 9, 2, 6, 2, 1),
            600,
            [
                (
                    datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 1, 20, 2, 1, tzinfo=timezone.utc),
                ),
                (
                    datetime(2008, 9, 1, 20, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 6, 2, 1, tzinfo=timezone.utc),
                ),
            ],
        ),
        (
            datetime(2024, 12, 26, 14, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 12, 26, 14, 4, 0, tzinfo=timezone.utc),
            1,
            [
                (
                    datetime(2024, 12, 26, 14, 0, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 1, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 1, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 2, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 2, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 3, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 3, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 4, 0, tzinfo=timezone.utc),
                ),
            ],
        ),
    ],
)
def test_windows(
    start: datetime,
    end: datetime,
    base_period: Optional[int],
    expected_windows: list[tuple[datetime, datetime]],
) -> None:
    assert (
        list(
            ModelMonitoringApplicationBase._window_generator(
                start=start.isoformat(),
                end=end.isoformat(),
                base_period=base_period,
                application_schedules=None,
                endpoint_id="",
                application_name="",
                existing_data_handling=ExistingDataHandling.fail_on_overlap,
            )
        )
        == expected_windows
    ), "The generated windows are different than expected"


@pytest.mark.parametrize(
    ("base_period", "start_dt", "end_dt", "expectation"),
    [
        (
            600,
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="The difference between `end` and `start` must be a multiple of "
                "`base_period`:.*Consider changing the `end` time to.*",
            ),
        ),
        (
            10,
            datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 7, 1, 0, 10, 0, tzinfo=timezone.utc),
            does_not_raise(),
        ),
        (
            15,
            datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 7, 1, 0, 10, 0, tzinfo=timezone.utc),
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="The difference between `end` and `start` must be a multiple of "
                "`base_period`:.*The `base_period` is longer than the difference between `end` and `start`.*",
            ),
        ),
    ],
)
def test_validate_and_get_window_length(
    base_period: int,
    start_dt: datetime,
    end_dt: datetime,
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        window_length = ModelMonitoringApplicationBase._validate_and_get_window_length(
            base_period=base_period, start_dt=start_dt, end_dt=end_dt
        )
        assert window_length == timedelta(
            minutes=base_period
        ), "The window length is different than expected"


def test_job_handler() -> None:
    assert (
        ModelMonitoringApplicationBase.get_job_handler(
            "package.subpackage.module.AppClass"
        )
        == "package.subpackage.module.AppClass::_handler"
    )


@pytest.mark.parametrize(
    ("result", "expected_flattened_result"),
    [
        (
            ModelMonitoringApplicationMetric(name="m1", value=98),
            {"metric_name": "m1", "metric_value": 98},
        ),
        (
            [
                ModelMonitoringApplicationMetric(name="m0", value=-2),
                ModelMonitoringApplicationResult(
                    name="r0",
                    value=0,
                    status=ResultStatusApp.no_detection,
                    kind=ResultKindApp.mm_app_anomaly,
                ),
            ],
            [
                {"metric_name": "m0", "metric_value": -2},
                {
                    "result_name": "r0",
                    "result_value": 0,
                    "result_status": 0,
                    "result_kind": 4,
                    "result_extra_data": "{}",
                },
            ],
        ),
    ],
)
def test_flatten_data_result(
    result: Union[
        ModelMonitoringApplicationMetric,
        ModelMonitoringApplicationResult,
        list[Union[ModelMonitoringApplicationMetric, ModelMonitoringApplicationResult]],
    ],
    expected_flattened_result: Union[dict, list[dict]],
) -> None:
    assert (
        ModelMonitoringApplicationBase._flatten_data_result(result)
        == expected_flattened_result
    ), "The flattened result is different than expected"


class TestToJob:
    @staticmethod
    @pytest.fixture
    def project(tmpdir: Path) -> mlrun.projects.MlrunProject:
        return mlrun.get_or_create_project("test-to-job", context=str(tmpdir))

    @staticmethod
    @pytest.fixture
    def _set_project(project: mlrun.projects.MlrunProject) -> Iterator[None]:
        with patch("mlrun.db.nopdb.NopDB.get_project", Mock(return_value=project)):
            yield

    @staticmethod
    def test_base_is_blocked(project: mlrun.projects.MlrunProject) -> None:
        with pytest.raises(
            ValueError,
            match="You must provide a handler to the model monitoring application class",
        ):
            ModelMonitoringApplicationBase.to_job(project=project)

    @staticmethod
    @pytest.mark.usefixtures("_set_project")
    def test_with_class_handler(project: mlrun.projects.MlrunProject) -> None:
        job = ModelMonitoringApplicationBase.to_job(
            func_path=__file__,
            class_handler="NoOpApp",
            project=project,
        )
        assert isinstance(job, mlrun.runtimes.KubejobRuntime)
        run = job.run(local=True)
        assert run.state() == "completed"


@pytest.fixture
def project(tmpdir: Path) -> mlrun.MlrunProject:
    return mlrun.get_or_create_project("test-endpoints-handler", context=str(tmpdir))


@pytest.mark.parametrize(
    "endpoints", ["all", ["model-ep-1"], [("model-ep-1", "model-ep-1-uid")]]
)
@pytest.mark.usefixtures("rundb_mock")
def test_handle_endpoints_type_evaluate(
    project: mlrun.MlrunProject, endpoints: Union[str, list[str], list[tuple[str, str]]]
) -> None:
    endpoints_output = ModelMonitoringApplicationBase._handle_endpoints_type_evaluate(
        project, endpoints
    )
    assert endpoints_output == [("model-ep-1", "model-ep-1-uid")]


@pytest.mark.parametrize(
    ("endpoints", "err_msg"),
    [
        ("*", 'A string input for `endpoints` can only be "all"'),
        ([], "The endpoints list cannot be empty"),
        ([1], r"Could not resolve endpoints as list of \[\(name, uid\)\]"),
    ],
)
def test_handle_endpoints_type_evaluate_error(
    project: mlrun.MlrunProject, endpoints: Union[str, list[str]], err_msg: str
) -> None:
    with pytest.raises(mlrun.errors.MLRunValueError, match=err_msg):
        ModelMonitoringApplicationBase._handle_endpoints_type_evaluate(
            project, endpoints
        )


@pytest.mark.parametrize(
    (
        "class_name",
        "func_name",
        "class_handler",
        "handler_to_class",
        "expectation",
        "expected_log",
    ),
    [
        (
            "App1",
            None,
            None,
            "App1::_handler",
            does_not_raise("app1-batch"),
            True,
        ),
        (
            "App1",
            None,
            "remote_module.AppClass",
            "remote_module.AppClass::_handler",
            does_not_raise("appclass-batch"),
            True,
        ),
        (
            "App1",
            "keep-my-batch",
            None,
            "AppClass::_handler",
            does_not_raise("keep-my-batch"),
            False,
        ),
        (
            "App",
            " app with space",
            None,
            None,
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="The function name does not comply with the required pattern",
            ),
            False,
        ),
    ],
)
@patch("mlrun.utils.logger", spec=mlrun.utils.Logger)
def test_determine_job_name(
    logger: Mock,
    class_name: str,
    func_name: Optional[str],
    class_handler: Optional[str],
    handler_to_class: str,
    expectation: AbstractContextManager,
    expected_log: bool,
) -> None:
    app_class = type(class_name, (ModelMonitoringApplicationBase,), {})
    with expectation as expected_result:
        assert (
            app_class._determine_job_name(
                func_name=func_name,
                class_handler=class_handler,
                handler_to_class=handler_to_class,
            )
            == expected_result
        )
    if expected_log:
        logger.info.assert_called_once()
    else:
        logger.info.assert_not_called()


@pytest.fixture
def run_context() -> mlrun.MLClientCtx:
    return mlrun.MLClientCtx.from_dict(
        {
            "kind": "run",
            "metadata": {
                "annotations": {},
                "iteration": 0,
                "labels": {
                    "host": "M-K",
                    "kind": "local",
                    "owner": "admin",
                    "v3io_user": "admin",
                },
                "name": "take-my-name-hold-my-hand--handler",
                "project": "mm-job-mep-data",
                "uid": "c8da9e6d9d9447b48109a29adb4bd4c2",
            },
            "spec": {
                "affinity": {},
                "data_stores": [],
                "function": "mm-job-mep-data/take-my-name-hold-my-hand@588be7f7fb3dcab77e59c5f3003056f90fa2b55b",
                "handler": "Some_application::_handler",
                "hyper_param_options": {},
                "hyperparams": {},
                "inputs": {},
                "log_level": "info",
                "node_selector": {},
                "notifications": [],
                "output_path": "v3io:///projects/mm-job-mep-data/artifacts",
                "outputs": [],
                "parameters": {
                    "application_name": "take-my-name-hold-my_hand",
                    "base_period": None,
                    "end": "2025-07-27T10:01:36.665785+00:00",
                    "endpoints": ["classifier-0"],
                    "existing_data_handling": "fail_on_overlap",
                    "start": "2025-07-27T10:00:31.527024+00:00",
                    "stream_profile": None,
                    "write_output": False,
                },
                "retry": {},
                "state_thresholds": {
                    "executing": "24h",
                    "image_pull_backoff": "1h",
                    "pending_not_scheduled": "-1",
                    "pending_scheduled": "1h",
                },
                "tolerations": {},
            },
            "status": {
                "artifact_uris": {},
                "last_update": "2025-07-27T15:31:38.482028+00:00",
                "results": {},
                "retries": [],
                "retry_count": None,
                "start_time": "2025-07-27T15:31:38.482028+00:00",
                "state": "running",
            },
        }
    )


def test_get_application_name(run_context: mlrun.MLClientCtx) -> None:
    assert (
        ModelMonitoringApplicationBase._get_application_name(run_context)
        == "take-my-name-hold-my-hand"
    ), "The application name is different than expected"
