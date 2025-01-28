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

import pytest
from nuclio.request import Logger as NuclioLogger

import mlrun
from mlrun import MLClientCtx, MlrunProject
from mlrun.errors import MLRunValueError
from mlrun.model_monitoring.applications.context import MonitoringApplicationContext
from mlrun.serving import GraphContext, GraphServer


@pytest.mark.parametrize("method", ["log_artifact", "log_dataset", "log_model"])
def test_log_object_signature(method: str) -> None:
    """Future-proof the `log_x` method of MM app context with respect to the project object"""
    assert inspect.signature(
        getattr(MonitoringApplicationContext, method)
    ) == inspect.signature(getattr(MlrunProject, method))


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
