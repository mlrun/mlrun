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

import inspect
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from nuclio.request import Logger as NuclioLogger

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
from mlrun import MLClientCtx, MlrunProject
from mlrun.errors import MLRunValueError
from mlrun.model_monitoring.applications.context import MonitoringApplicationContext
from mlrun.serving import GraphContext, GraphServer


@pytest.mark.parametrize("method", ["log_artifact", "log_dataset"])
def test_log_object_signature(method: str) -> None:
    """Future-proof the `log_x` method of MM app context with respect to the project object"""
    monitoring_parameters = list(
        inspect.signature(
            getattr(MonitoringApplicationContext, method)
        ).parameters.keys()
    )
    project_parameters = list(
        inspect.signature(getattr(MlrunProject, method)).parameters.keys()
    )
    assert (
        project_parameters <= monitoring_parameters
    ), f"All MlrunProject {method} params should appear in MonitoringApplicationContext {method}"


def test_from_graph_context(tmp_path: Path) -> None:
    with patch.object(
        mlrun.db.get_run_db(),
        "get_project",
        Mock(
            return_value=mlrun.projects.MlrunProject(
                spec=mlrun.projects.ProjectSpec(artifact_path=str(tmp_path))
            )
        ),
    ) as get_project_mock:
        app_ctx = MonitoringApplicationContext._from_graph_ctx(
            application_name="app-context-from-graph",
            event={},
            graph_context=GraphContext(
                server=GraphServer(function_uri="project-name/function-name"),
                logger=NuclioLogger(level=logging.DEBUG),
            ),
        )
        app_ctx.logger.info("Test from graph_context logger")
        get_project_mock.assert_called_once()


@pytest.mark.parametrize(
    "ml_ctx_dict", [{"metadata": {"project": "some-local-proj"}}, {}]
)
def test_from_ml_context_error(ml_ctx_dict: dict[str, str]) -> None:
    ml_ctx = MLClientCtx.from_dict(ml_ctx_dict)
    with pytest.raises(MLRunValueError, match="Could not load project from context"):
        MonitoringApplicationContext._from_ml_ctx(
            application_name="app-context-from-ml",
            event={},
            context=ml_ctx,
        )


@patch("mlrun.db.nopdb.NopDB.get_project")
def test_from_ml_context(mock: Mock) -> None:
    project_name = "my-proj"
    ml_ctx = MLClientCtx.from_dict({"metadata": {"project": project_name}})
    assert ml_ctx.project == project_name
    app_ctx = MonitoringApplicationContext._from_ml_ctx(
        application_name="app-context-from-ml",
        event={},
        context=ml_ctx,
    )
    app_ctx.logger.info("MM app context from `MLClientCtx`")
    mock.assert_called_once()


@pytest.mark.parametrize(
    "start,end,expected",
    [
        # less than 1 hour -> "hour"
        ("2024-01-01T00:00:00", "2024-01-01T00:30:00", "hour"),
        # exactly 1 hour -> not <1h -> "day"
        ("2024-01-01T00:00:00", "2024-01-01T01:00:00", "day"),
        # several hours but less than a day -> "day"
        ("2024-01-01T00:00:00", "2024-01-01T12:00:00", "day"),
        # less than 30 days but >= 1 day -> "month"
        ("2024-01-01T00:00:00", "2024-01-11T00:00:00", "month"),
        # less than 365 days but >= 30 days -> "year"
        ("2024-01-01T00:00:00", "2024-04-15T00:00:00", "year"),
        # more than a year -> "year"
        ("2023-01-01T00:00:00", "2024-06-01T00:00:00", "year"),
    ],
)
@patch("mlrun.store_manager.get_or_create_store")
def test_granularity_for_time_ranges(mock_get_store, start, end, expected):
    # Patch store_manager to return a store with get_storage_options
    store = Mock()
    store.get_storage_options.return_value = {}
    mock_get_store.return_value = (store, None, None)

    # Minimal project and artifacts_logger
    project = Mock()
    project.name = "test-proj"
    artifacts_logger = Mock()
    artifacts_logger.log_artifact = Mock()
    artifacts_logger.log_dataset = Mock()

    # Build event dict expected by MonitoringApplicationContext
    event = {
        mm_constants.ApplicationEvent.START_INFER_TIME: str(pd.Timestamp(start)),
        mm_constants.ApplicationEvent.END_INFER_TIME: str(pd.Timestamp(end)),
        mm_constants.ApplicationEvent.ENDPOINT_ID: "endpoint-1",
        mm_constants.ApplicationEvent.ENDPOINT_NAME: "endpoint-name",
    }

    # Create the context
    app_ctx = MonitoringApplicationContext(
        application_name="test-app",
        event=event,
        project=project,
        artifacts_logger=artifacts_logger,
        logger=logging.getLogger("test"),
        nuclio_logger=NuclioLogger(level=logging.DEBUG),
    )

    assert app_ctx.granularity == expected
