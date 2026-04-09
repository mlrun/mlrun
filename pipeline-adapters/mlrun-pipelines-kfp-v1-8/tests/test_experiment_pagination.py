# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for KFP experiment listing pagination in _get_candidate_experiments_for_projects.

Verifies that the method correctly follows next_page_token to retrieve all
experiments across multiple pages, preventing silent data loss.

See: issues/bug-kfp-experiment-listing-no-pagination/problem.md
"""

import unittest.mock
from unittest.mock import MagicMock

import kfp_server_api
import pytest

import mlrun.utils
import mlrun_pipelines.client


@pytest.fixture
def client(monkeypatch):
    """Create a Client instance with mocked infrastructure."""
    client_klass = mlrun_pipelines.client.Client
    client_klass.get_kfp_healthz = unittest.mock.MagicMock()
    monkeypatch.setattr("kubernetes.config.load_incluster_config", lambda: None)
    monkeypatch.setattr(client_klass, "_determine_server_major_version", lambda self: 2)
    return client_klass(logger=mlrun.utils.logger)


def _make_experiment(name, experiment_id=None):
    """Helper to create a mock ApiExperiment."""
    exp = MagicMock(spec=kfp_server_api.ApiExperiment)
    exp.name = name
    exp.id = experiment_id or f"id-{name}"
    return exp


def _make_list_response(experiments, next_page_token=None):
    """Helper to create a mock list_experiment response."""
    response = MagicMock()
    response.experiments = experiments if experiments else None
    response.next_page_token = next_page_token
    return response


DEFAULT_PAGE_SIZE = mlrun.common.schemas.PipelinesPagination.default_page_size


class TestGetCandidateExperimentsPagination:
    """Tests verifying that _get_candidate_experiments_for_projects paginates correctly."""

    def test_multi_page_experiments_returns_all_pages(self, client, monkeypatch):
        """
        When list_experiment returns a next_page_token, the method follows it
        and returns experiments from all pages.
        """
        page1_experiments = [_make_experiment(f"x-myproj-exp{i}") for i in range(5)]
        page2_experiments = [
            _make_experiment("myproj-exp1"),
        ]

        page1_response = _make_list_response(
            page1_experiments, next_page_token="page2_token"
        )
        page2_response = _make_list_response(page2_experiments, next_page_token=None)

        mock_list = MagicMock(side_effect=[page1_response, page2_response])
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        result = client._get_candidate_experiments_for_projects(
            ["myproj"], page_size=DEFAULT_PAGE_SIZE
        )

        assert len(result) == 1
        assert result[0].name == "myproj-exp1"
        assert mock_list.call_count == 2

    def test_page_token_passed_on_subsequent_calls(self, client, monkeypatch):
        """
        Verify that the next_page_token from page 1 is passed as page_token
        to the second call.
        """
        page1_response = _make_list_response(
            [_make_experiment("myproject-exp1")],
            next_page_token="there_are_more_results",
        )
        page2_response = _make_list_response(
            [_make_experiment("myproject-exp2")], next_page_token=None
        )

        mock_list = MagicMock(side_effect=[page1_response, page2_response])
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        result = client._get_candidate_experiments_for_projects(
            ["myproject"], page_size=DEFAULT_PAGE_SIZE
        )

        assert len(result) == 2
        assert mock_list.call_count == 2

        first_call_kwargs = mock_list.call_args_list[0].kwargs
        second_call_kwargs = mock_list.call_args_list[1].kwargs
        assert first_call_kwargs["page_token"] is None
        assert second_call_kwargs["page_token"] == "there_are_more_results"

    def test_single_page_works_correctly(self, client, monkeypatch):
        """
        Regression test: Single page of results works correctly.
        """
        experiments = [
            _make_experiment("proj-a-exp1"),
            _make_experiment("proj-a-exp2"),
            _make_experiment("other-proj-exp1"),
        ]
        response = _make_list_response(experiments, next_page_token=None)
        mock_list = MagicMock(return_value=response)
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        result = client._get_candidate_experiments_for_projects(
            ["proj-a"], page_size=DEFAULT_PAGE_SIZE
        )

        assert len(result) == 2
        assert all(exp.name.startswith("proj-a") for exp in result)
        assert mock_list.call_count == 1

    def test_empty_results(self, client, monkeypatch):
        """
        Regression test: Empty experiment list is handled correctly.
        """
        response = _make_list_response(None, next_page_token=None)
        mock_list = MagicMock(return_value=response)
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        result = client._get_candidate_experiments_for_projects(
            ["nonexistent"], page_size=DEFAULT_PAGE_SIZE
        )

        assert result == []

    def test_overlapping_names_experiments_on_later_pages_found(
        self, client, monkeypatch
    ):
        """
        Experiments on later pages are found when project names overlap.

        Given projects with overlapping name substrings:
        - "data-pipeline" (target project, 1 experiment on page 2)
        - "pp-data-pipeline" (similar project, many experiments on page 1)
        """
        page1_experiments = [
            _make_experiment(f"pp-data-pipeline-exp{i}") for i in range(10)
        ]
        page2_experiments = [
            _make_experiment("data-pipeline-exp1"),
        ]

        page1_response = _make_list_response(
            page1_experiments, next_page_token="next_page"
        )
        page2_response = _make_list_response(page2_experiments, next_page_token=None)

        mock_list = MagicMock(side_effect=[page1_response, page2_response])
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        result = client._get_candidate_experiments_for_projects(
            ["data-pipeline"], page_size=DEFAULT_PAGE_SIZE
        )

        assert len(result) == 1
        assert result[0].name == "data-pipeline-exp1"

    def test_all_matching_experiments_across_pages(self, client, monkeypatch):
        """
        When a project has many experiments spanning multiple pages, all are returned.
        """
        page1_experiments = [_make_experiment(f"bigproject-exp{i}") for i in range(10)]
        page2_experiments = [
            _make_experiment(f"bigproject-exp{i}") for i in range(10, 15)
        ]

        page1_response = _make_list_response(page1_experiments, next_page_token="page2")
        page2_response = _make_list_response(page2_experiments, next_page_token=None)

        mock_list = MagicMock(side_effect=[page1_response, page2_response])
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        result = client._get_candidate_experiments_for_projects(
            ["bigproject"], page_size=DEFAULT_PAGE_SIZE
        )

        assert len(result) == 15

    def test_page_size_passed_to_api(self, client, monkeypatch):
        """
        Verify that page_size is passed to the list_experiment API.
        """
        response = _make_list_response(
            [_make_experiment("proj-exp1")], next_page_token=None
        )
        mock_list = MagicMock(return_value=response)
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        client._get_candidate_experiments_for_projects(
            ["proj"], page_size=DEFAULT_PAGE_SIZE
        )

        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["page_size"] == DEFAULT_PAGE_SIZE

    def test_three_pages_of_results(self, client, monkeypatch):
        """
        Verify pagination works across three pages.
        """
        page1 = _make_list_response(
            [_make_experiment("proj-a")], next_page_token="tok2"
        )
        page2 = _make_list_response(
            [_make_experiment("proj-b")], next_page_token="tok3"
        )
        page3 = _make_list_response([_make_experiment("proj-c")], next_page_token=None)

        mock_list = MagicMock(side_effect=[page1, page2, page3])
        monkeypatch.setattr(client._experiment_api, "list_experiment", mock_list)

        result = client._get_candidate_experiments_for_projects(
            ["proj"], page_size=DEFAULT_PAGE_SIZE
        )

        assert len(result) == 3
        assert mock_list.call_count == 3
