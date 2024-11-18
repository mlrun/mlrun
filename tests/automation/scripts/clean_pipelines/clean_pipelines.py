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
#
import json
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest

from automation.scripts.clean_pipelines.clean_pipelines import (
    _filter_project_runs,
    _get_list_runs_filter,
    _validate_and_convert_date,
)

PROJECT_ANNOTATION = "mlrun/project"


def create_mock_run(project_name: str, run_name: str) -> MagicMock:
    """
    Helper function to create a mock PipelineRun with a specified project name.

    :param project_name: The project name to set in the annotations.
    :param run_name: The name of the run for easier identification in tests.
    :return: A MagicMock object mimicking a PipelineRun with a workflow_manifest.
    """
    mock_run = MagicMock()
    mock_run.workflow_manifest.return_value = {
        "spec": {
            "templates": [
                {"metadata": {"annotations": {PROJECT_ANNOTATION: project_name}}}
            ]
        },
        "metadata": {"name": run_name},
    }
    return mock_run


@pytest.mark.parametrize(
    "date_input, expected_output, expectation",
    [
        # Valid date without timezone, assume UTC
        ("2024-11-05T15:30:00", "2024-11-05T15:30:00Z", does_not_raise()),
        # Valid date with UTC timezone
        ("2024-11-05T15:30:00Z", "2024-11-05T15:30:00Z", does_not_raise()),
        # Valid date with different timezone (convert to UTC)
        ("2024-11-05T15:30:00+02:00", "2024-11-05T13:30:00Z", does_not_raise()),
        # Valid date with timezone-aware string
        ("2024-11-05T15:30:00-05:00", "2024-11-05T20:30:00Z", does_not_raise()),
        # Date with timezone info but no time
        ("2024-11-05", "2024-11-05T00:00:00Z", does_not_raise()),
        ("2024/11/05T09:00", "2024-11-05T09:00:00Z", does_not_raise()),
        # Invalid date format
        ("invalid-date", "", pytest.raises(ValueError)),
        # Overflow date (not a realistic timestamp)
        ("9999-99-99T99:99:99Z", "", pytest.raises(ValueError)),
    ],
)
def test_validate_and_convert_date(date_input, expected_output, expectation):
    with expectation:
        assert _validate_and_convert_date(date_input) == expected_output


@pytest.mark.parametrize(
    "project_name, end_date, start_date, expected_filter",
    [
        # Specific project, end date only
        (
            "test-project",
            "2024-11-05T15:30:00Z",
            "",
            json.dumps(
                {
                    "predicates": [
                        {
                            "key": "created_at",
                            "op": 7,
                            "timestamp_value": "2024-11-05T15:30:00Z",
                        },
                        {"key": "name", "op": 9, "string_value": "test-project"},
                    ]
                }
            ),
        ),
        # Wildcard project, end date only
        (
            "*",
            "2024-11-05T15:30:00Z",
            "",
            json.dumps(
                {
                    "predicates": [
                        {
                            "key": "created_at",
                            "op": 7,
                            "timestamp_value": "2024-11-05T15:30:00Z",
                        },
                    ]
                }
            ),
        ),
        # Specific project with both start and end dates
        (
            "test-project",
            "2024-11-05T15:30:00Z",
            "2024-10-01T00:00:00Z",
            json.dumps(
                {
                    "predicates": [
                        {
                            "key": "created_at",
                            "op": 7,
                            "timestamp_value": "2024-11-05T15:30:00Z",
                        },
                        {"key": "name", "op": 9, "string_value": "test-project"},
                        {
                            "key": "created_at",
                            "op": 5,
                            "timestamp_value": "2024-10-01T00:00:00Z",
                        },
                    ]
                }
            ),
        ),
        # Wildcard project with both start and end dates
        (
            "*",
            "2024-11-05T15:30:00Z",
            "2024-10-01T00:00:00Z",
            json.dumps(
                {
                    "predicates": [
                        {
                            "key": "created_at",
                            "op": 7,
                            "timestamp_value": "2024-11-05T15:30:00Z",
                        },
                        {
                            "key": "created_at",
                            "op": 5,
                            "timestamp_value": "2024-10-01T00:00:00Z",
                        },
                    ]
                }
            ),
        ),
    ],
)
def test_get_list_runs_filter(project_name, end_date, start_date, expected_filter):
    generated_filter = _get_list_runs_filter(project_name, end_date, start_date)
    assert json.loads(generated_filter) == json.loads(expected_filter)


@pytest.mark.parametrize(
    "project_name, runs, expected_filtered_runs",
    [
        # Specific project name with annotations in workflow_manifest
        (
            "project-1",
            [
                create_mock_run("project-1", "run-1"),
                create_mock_run("project-2", "run-2"),
                create_mock_run("project-1", "run-3"),
            ],
            ["run-1", "run-3"],
        ),
        # Wildcard project name, should return all runs
        (
            "*",
            [
                create_mock_run("project-1", "run-1"),
                create_mock_run("project-2", "run-2"),
                create_mock_run("project-3", "run-3"),
            ],
            ["run-1", "run-2", "run-3"],
        ),
        # No matching project names
        (
            "non-existent-project",
            [
                create_mock_run("project-1", "run-1"),
                create_mock_run("project-2", "run-2"),
            ],
            [],
        ),
        # Empty list of runs, should return an empty list regardless of project name
        (
            "project-1",
            [],
            [],
        ),
    ],
)
def test_filter_project_runs(project_name, runs, expected_filtered_runs):
    filtered_runs = _filter_project_runs(project_name, runs)
    filtered_run_names = [
        run.workflow_manifest().get("metadata", {}).get("name") for run in filtered_runs
    ]
    assert filtered_run_names == expected_filtered_runs
