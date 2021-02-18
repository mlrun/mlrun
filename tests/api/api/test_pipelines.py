import http
import importlib
import unittest.mock

import deepdiff
import fastapi.testclient
import kfp
import kfp_server_api.models
import pytest
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s


@pytest.fixture
def kfp_client_mock(monkeypatch) -> kfp.Client:
    mlrun.api.utils.singletons.k8s.get_k8s().is_running_inside_kubernetes_cluster = (
        unittest.mock.Mock(return_value=True)
    )
    kfp_client_mock = unittest.mock.Mock()
    monkeypatch.setattr(kfp, "Client", lambda *args, **kwargs: kfp_client_mock)
    return kfp_client_mock


def test_list_pipelines_not_exploding_on_no_k8s(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("/api/projects/*/pipelines")
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=[], total_size=0, next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_empty_list(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    runs = []
    _mock_list_runs(kfp_client_mock, runs)
    response = client.get("/api/projects/*/pipelines")
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=runs, total_size=len(runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_names_only(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    runs = _generate_run_mocks()
    expected_runs = [run.name for run in runs]
    _mock_list_runs(kfp_client_mock, runs)
    response = client.get(
        "/api/projects/*/pipelines",
        params={"format": mlrun.api.schemas.Format.name_only},
    )
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=expected_runs, total_size=len(runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_metadata_only(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    runs = _generate_run_mocks()
    expected_runs = [run.to_dict() for run in runs]
    expected_runs = mlrun.api.crud.pipelines._format_runs(
        expected_runs, mlrun.api.schemas.Format.metadata_only
    )
    _mock_list_runs(kfp_client_mock, runs)
    response = client.get(
        "/api/projects/*/pipelines",
        params={"format": mlrun.api.schemas.Format.metadata_only},
    )
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=expected_runs, total_size=len(runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_full(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    runs = _generate_run_mocks()
    expected_runs = [run.to_dict() for run in runs]
    _mock_list_runs(kfp_client_mock, runs)
    response = client.get(
        "/api/projects/*/pipelines", params={"format": mlrun.api.schemas.Format.full}
    )
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=expected_runs, total_size=len(runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_specific_project(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    project = "project-name"
    runs = _generate_run_mocks()
    expected_runs = [run.name for run in runs]
    _mock_list_runs_with_one_run_per_page(kfp_client_mock, runs)
    mlrun.api.crud.pipelines._resolve_pipeline_project = unittest.mock.Mock(
        return_value=project
    )
    response = client.get(
        f"/api/projects/{project}/pipelines",
        params={"format": mlrun.api.schemas.Format.name_only},
    )
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=expected_runs, total_size=len(expected_runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)

    # revert mock setting (it's global function, without reloading it the mock will persist to following tests)
    importlib.reload(mlrun.api.crud.pipelines)


def _generate_run_mocks():
    return [
        kfp_server_api.models.api_run.ApiRun(
            id="id1",
            name="run1",
            description="desc1",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id1"
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id2",
            name="run2",
            description="desc2",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id2"
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id3",
            name="run3",
            description="desc3",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id3"
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id4",
            name="run4",
            description="desc4",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id4"
            ),
        ),
    ]


def _mock_list_runs_with_one_run_per_page(kfp_client_mock: kfp.Client, runs):
    expected_page_tokens = [""]
    for i in range(2, len(runs) + 1):
        expected_page_tokens.append(i)
    expected_page_tokens.append(None)

    def list_runs_mock(*args, page_token=None, page_size=None, **kwargs):
        assert expected_page_tokens.pop(0) == page_token
        assert mlrun.api.schemas.PipelinesPagination.max_page_size == page_size
        return kfp_server_api.models.api_list_runs_response.ApiListRunsResponse(
            [runs.pop(0)], 1, next_page_token=expected_page_tokens[0]
        )

    kfp_client_mock._run_api.list_runs = list_runs_mock


def _mock_list_runs(
    kfp_client_mock: kfp.Client,
    runs,
    expected_page_token="",
    expected_page_size=mlrun.api.schemas.PipelinesPagination.default_page_size,
    expected_sort_by="",
    expected_filter="",
):
    def list_runs_mock(
        *args, page_token=None, page_size=None, sort_by=None, filter=None, **kwargs
    ):
        assert expected_page_token == page_token
        assert expected_page_size == page_size
        assert expected_sort_by == sort_by
        assert expected_filter == filter
        return kfp_server_api.models.api_list_runs_response.ApiListRunsResponse(
            runs, len(runs)
        )

    kfp_client_mock._run_api.list_runs = list_runs_mock


def _assert_list_pipelines_response(
    expected_response: mlrun.api.schemas.PipelinesOutput, response
):
    assert response.status_code == http.HTTPStatus.OK.value
    assert (
        deepdiff.DeepDiff(
            expected_response.dict(),
            response.json(),
            ignore_order=True,
        )
        == {}
    )
