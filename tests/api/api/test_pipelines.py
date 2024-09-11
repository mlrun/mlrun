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
import datetime
import http
import importlib
import json
import unittest.mock

import deepdiff
import fastapi.testclient
import kfp
import kfp_server_api.models
import pytest
import sqlalchemy.orm
from mlrun_pipelines.models import PipelineRun

import mlrun.common.formatters
import mlrun.common.schemas
import server.api.crud
import server.api.utils.auth.verifier
import tests.conftest


def test_list_pipelines_not_exploding_on_no_k8s(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("projects/*/pipelines")
    expected_response = mlrun.common.schemas.PipelinesOutput(
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
    response = client.get("projects/*/pipelines")
    expected_response = mlrun.common.schemas.PipelinesOutput(
        runs=runs, total_size=len(runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_formats(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    for format_ in [
        mlrun.common.formatters.PipelineFormat.full,
        mlrun.common.formatters.PipelineFormat.metadata_only,
        mlrun.common.formatters.PipelineFormat.name_only,
    ]:
        runs = _generate_list_runs_mocks()
        expected_runs = [PipelineRun(run.to_dict()) for run in runs]
        expected_runs = server.api.crud.Pipelines()._format_runs(expected_runs, format_)
        _mock_list_runs(kfp_client_mock, runs)
        response = client.get(
            "projects/*/pipelines",
            params={"format": format_},
        )
        expected_response = mlrun.common.schemas.PipelinesOutput(
            runs=expected_runs, total_size=len(runs), next_page_token=None
        )
        _assert_list_pipelines_response(expected_response, response)


def test_get_pipeline_formats(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    for format_ in [
        mlrun.common.formatters.PipelineFormat.full,
        mlrun.common.formatters.PipelineFormat.metadata_only,
        mlrun.common.formatters.PipelineFormat.summary,
        mlrun.common.formatters.PipelineFormat.name_only,
    ]:
        api_run_detail = _generate_get_run_mock()
        _mock_get_run(kfp_client_mock, api_run_detail)
        response = client.get(
            f"projects/*/pipelines/{api_run_detail.run.id}",
            params={"format": format_},
        )
        expected_run = server.api.crud.Pipelines()._format_run(
            PipelineRun(api_run_detail),
            format_,
        )
        _assert_get_pipeline_response(expected_run, response)


def test_get_pipeline_no_project_opa_validation(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    format_ = (mlrun.common.formatters.PipelineFormat.summary,)
    project = "project-name"
    server.api.crud.Pipelines().resolve_project_from_pipeline = unittest.mock.Mock(
        return_value=project
    )
    server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions = (
        unittest.mock.AsyncMock()
    )
    api_run_detail = _generate_get_run_mock()
    _mock_get_run(kfp_client_mock, api_run_detail)
    response = client.get(
        f"projects/*/pipelines/{api_run_detail.run.id}",
        params={"format": format_},
    )
    assert (
        server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions.call_args[
            0
        ][1]
        == project
    )
    assert response.json()["run"]["project"] == project


def test_get_pipeline_specific_project(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    for format_ in [
        mlrun.common.formatters.PipelineFormat.full,
        mlrun.common.formatters.PipelineFormat.metadata_only,
        mlrun.common.formatters.PipelineFormat.summary,
        mlrun.common.formatters.PipelineFormat.name_only,
    ]:
        project = "project-name"
        api_run_detail = _generate_get_run_mock()
        _mock_get_run(kfp_client_mock, api_run_detail)
        server.api.crud.Pipelines().resolve_project_from_pipeline = unittest.mock.Mock(
            return_value=project
        )
        response = client.get(
            f"projects/{project}/pipelines/{api_run_detail.run.id}",
            params={"format": format_},
        )
        expected_run = server.api.crud.Pipelines()._format_run(
            PipelineRun(api_run_detail), format_
        )
        _assert_get_pipeline_response(expected_run, response)

        # revert mock setting (it's global function, without reloading it the mock will persist to following tests)
        importlib.reload(server.api.crud)


def test_list_pipelines_time_fields_default(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    created_at = datetime.datetime.now()
    workflow_manifest = _generate_workflow_manifest()
    runs = [
        kfp_server_api.models.api_run.ApiRun(
            id="id1",
            name="run",
            description="desc",
            created_at=created_at,
            finished_at="1970-01-01 00:00:00+00:00",
            scheduled_at="1970-01-01 00:00:00+00:00",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id",
                workflow_manifest=workflow_manifest,
            ),
        )
    ]

    _mock_list_runs(kfp_client_mock, runs)
    response = client.get(
        "projects/*/pipelines",
        params={"format": mlrun.common.formatters.PipelineFormat.metadata_only},
    )
    response = response.json()["runs"][0]

    assert response["created_at"] == str(created_at)
    assert not response["finished_at"], (
        "Expected value to be None after format,"
        " since field has not been specified yet"
    )
    assert not response["scheduled_at"], (
        "Expected value to be None after format,"
        " since field has not been specified yet"
    )


@pytest.mark.parametrize(
    "project_name, run_name_filter, expected_runs_ids",
    [
        ("test-project", "workflow", ["id3", "id4"]),
        ("test-project", "project", ["id1", "id2"]),
        ("test-project", "test", ["id1", "id2", "id3", "id4"]),
        ("test-project", "another", ["id4"]),
        ("test-project", "test-project-", []),
        ("*", "project", ["id1", "id2"]),
        ("*", "workflow", ["id3", "id4"]),
    ],
)
def test_list_pipelines_name_contains(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
    project_name: str,
    run_name_filter: str,
    expected_runs_ids: list,
) -> None:
    project_name = "test-project"
    server.api.crud.Pipelines().resolve_project_from_pipeline = unittest.mock.Mock(
        return_value=project_name
    )
    runs = _generate_list_runs_project_name_mocks()
    expected_page_size = (
        mlrun.common.schemas.PipelinesPagination.default_page_size
        if project_name == "*"
        else mlrun.common.schemas.PipelinesPagination.max_page_size
    )
    _mock_list_runs(
        kfp_client_mock,
        runs,
        expected_page_size=expected_page_size,
        expected_filter=mlrun.utils.get_kfp_project_filter(project_name=project_name),
    )
    response = client.get(
        f"projects/{project_name}/pipelines",
        params={
            "name-contains": run_name_filter,
        },
    )

    expected_runs = server.api.crud.Pipelines()._format_runs(
        [PipelineRun(run.to_dict()) for run in runs if run.id in expected_runs_ids]
    )
    expected_response = mlrun.common.schemas.PipelinesOutput(
        runs=expected_runs, total_size=len(expected_runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_specific_project(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    project = "project-name"
    runs = _generate_list_runs_mocks()
    expected_runs = [run.name for run in runs]
    _mock_list_runs_with_one_run_per_page(kfp_client_mock, runs)
    server.api.crud.Pipelines().resolve_project_from_pipeline = unittest.mock.Mock(
        return_value=project
    )
    response = client.get(
        f"projects/{project}/pipelines",
        params={"format": mlrun.common.formatters.PipelineFormat.name_only},
    )
    expected_response = mlrun.common.schemas.PipelinesOutput(
        runs=expected_runs, total_size=len(expected_runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)

    # revert mock setting (it's global function, without reloading it the mock will persist to following tests)
    importlib.reload(server.api.crud)


def test_create_pipeline(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    project = "getting-started-tutorial-iguazio"
    pipeline_file_path = (
        tests.conftest.tests_root_directory
        / "api"
        / "api"
        / "assets"
        / "pipelines.yaml"
    )
    with open(str(pipeline_file_path)) as file:
        contents = file.read()
    _mock_pipelines_creation(kfp_client_mock)
    response = client.post(
        f"projects/{project}/pipelines",
        data=contents,
        headers={"content-type": "application/yaml"},
    )
    response_body = response.json()
    assert response_body["id"] == "some-run-id"


def _generate_get_run_mock() -> kfp_server_api.models.api_run_detail.ApiRunDetail:
    workflow_manifest = _generate_workflow_manifest()
    workflow_manifest_with_status = _generate_workflow_manifest(with_status=True)
    return kfp_server_api.models.api_run_detail.ApiRunDetail(
        run=kfp_server_api.models.api_run.ApiRun(
            id="id1",
            name="run1",
            description="desc1",
            created_at="0001-01-01 00:00:00+00:00",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id1",
                workflow_manifest=workflow_manifest,
            ),
        ),
        pipeline_runtime=kfp_server_api.models.api_pipeline_runtime.ApiPipelineRuntime(
            workflow_manifest=workflow_manifest_with_status
        ),
    )


def test_get_pipeline_nonexistent_project(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    format_ = (mlrun.common.formatters.PipelineFormat.summary,)
    project = "n0_pr0ject"
    api_run_detail = _generate_get_run_mock()
    _mock_get_run(kfp_client_mock, api_run_detail)
    response = client.get(
        f"projects/{project}/pipelines/{api_run_detail.run.id}",
        params={"format": format_},
    )
    assert response.status_code == http.HTTPStatus.NOT_FOUND.value


def _generate_list_runs_mocks():
    workflow_manifest = _generate_workflow_manifest()
    return [
        kfp_server_api.models.api_run.ApiRun(
            id="id1",
            name="run1",
            description="desc1",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id1",
                workflow_manifest=workflow_manifest,
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id2",
            name="run2",
            description="desc2",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id2",
                workflow_manifest=workflow_manifest,
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id3",
            name="run3",
            description="desc3",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id3",
                workflow_manifest=workflow_manifest,
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id4",
            name="run4",
            description="desc4",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id4",
                workflow_manifest=workflow_manifest,
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id5",
            name="run5",
            description="desc5",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id5",
                workflow_manifest=workflow_manifest,
            ),
            error="error",
        ),
    ]


def _generate_list_runs_project_name_mocks():
    """
    Generate mock runs for KFP taking into account the naming patterns used by MLRun in a real world scenario
    """
    workflow_manifest = _generate_workflow_manifest()
    return [
        kfp_server_api.models.api_run.ApiRun(
            id="id1",
            name="test-project 0000-00-00 00-00-01",
            description="desc1",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id1",
                workflow_manifest=workflow_manifest,
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id2",
            name="test-project 0000-00-00 00-00-02",
            description="desc2",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id2",
                workflow_manifest=workflow_manifest,
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id3",
            name="test-project-test-workflow 0000-00-00 00-00-03",
            description="desc3",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id3",
                workflow_manifest=workflow_manifest,
            ),
        ),
        kfp_server_api.models.api_run.ApiRun(
            id="id4",
            name="test-project-test-another-workflow 0000-00-00 00-00-04",
            description="desc4",
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id4",
                workflow_manifest=workflow_manifest,
            ),
        ),
    ]


def _generate_workflow_manifest(with_status=False):
    workflow_manifest = {
        "metadata": {
            "name": "minimal-pipeline-rmtvd",
            "namespace": "default-tenant",
            "creationTimestamp": "2021-08-23T00:01:31Z",
            "labels": {
                "pipeline/runid": "c74810e9-a5ae-4ad4-bb1f-efd38e529c0f",
                "pipelines.kubeflow.org/kfp_sdk_version": "1.0.1",
                "workflows.argoproj.io/completed": "true",
                "workflows.argoproj.io/phase": "Succeeded",
            },
            "annotations": {
                "pipelines.kubeflow.org/kfp_sdk_version": "1.0.1",
                "pipelines.kubeflow.org/pipeline_compilation_time": "2021-08-23T00:01:30.667929",
                "pipelines.kubeflow.org/pipeline_spec": '{"description": "demonstrating mlrun usage", "inputs": [{"'
                'default": "False", "name": "fail", "optional": true, "type": "Boolean"}], "name": "minimal pipelin'
                'e"}',
                "pipelines.kubeflow.org/run_name": "my-pipeline 2021-08-23 00-01-30",
            },
        },
        "spec": {
            "templates": [
                {
                    "name": "hedi-simple-func-do-something",
                    "inputs": {"parameters": [{"name": "fail"}]},
                    "outputs": {
                        "artifacts": [
                            {
                                "name": "mlpipeline-ui-metadata",
                                "path": "/mlpipeline-ui-metadata.json",
                                "optional": True,
                            }
                        ]
                    },
                    "metadata": {
                        "annotations": {
                            "mlrun/function-uri": "default/hedi-simple-func@a5b181289c7ee40f7fba2a31ed73ff65043dfd2"
                            "7",
                            "mlrun/pipeline-step-type": "run",
                            "mlrun/project": "default",
                            "sidecar.istio.io/inject": "false",
                        },
                        "labels": {"pipelines.kubeflow.org/cache_enabled": "true"},
                    },
                    "container": {
                        "name": "",
                        "image": "datanode-registry.iguazio-platform.app.vmdev27.lab.iguazeng.com:80/quay.io/mlrun/"
                        "mlrun:0.7.0-rc5",
                        "command": [
                            "python",
                            "-m",
                            "mlrun",
                            "run",
                            "--kfp",
                            "--from-env",
                            "--workflow",
                            "c74810e9-a5ae-4ad4-bb1f-efd38e529c0f",
                            "--name",
                            "hedi-simple-func-do_something",
                            "-f",
                            "db://default/hedi-simple-func@a5b181289c7ee40f7fba2a31ed73ff65043dfd27",
                            "-p",
                            "fail={{inputs.parameters.fail}}",
                            "--label",
                            "v3io_user=iguazio",
                            "--label",
                            "owner=iguazio",
                            "-o",
                            "run_id",
                            "--handler",
                            "do_something",
                            "",
                        ],
                        "env": [
                            {
                                "name": "MLRUN_NAMESPACE",
                                "valueFrom": {
                                    "fieldRef": {"fieldPath": "metadata.namespace"}
                                },
                            }
                        ],
                        "resources": {},
                    },
                }
            ],
            "entrypoint": "minimal-pipeline",
            "arguments": {"parameters": [{"name": "fail", "value": "False"}]},
            "serviceAccountName": "pipeline-runner",
            "ttlSecondsAfterFinished": 14400,
        },
    }
    if with_status:
        workflow_manifest["status"] = {
            "phase": "Succeeded",
            "startedAt": "2021-08-23T00:01:31Z",
            "finishedAt": "2021-08-23T00:02:06Z",
            "nodes": {
                "minimal-pipeline-rmtvd": {
                    "id": "minimal-pipeline-rmtvd",
                    "name": "minimal-pipeline-rmtvd",
                    "displayName": "minimal-pipeline-rmtvd",
                    "type": "DAG",
                    "templateName": "minimal-pipeline",
                    "phase": "Succeeded",
                    "startedAt": "2021-08-23T00:01:31Z",
                    "finishedAt": "2021-08-23T00:02:06Z",
                    "inputs": {"parameters": [{"name": "fail", "value": "False"}]},
                    "children": [],
                    "outboundNodes": [],
                }
            },
        }
    return json.dumps(workflow_manifest)


def _mock_pipelines_creation(kfp_client_mock: kfp.Client):
    def _mock_create_experiment(name, description=None, namespace=None):
        return kfp_server_api.models.ApiExperiment(
            id="some-exp-id",
            name=name,
            description=description,
        )

    def _mock_run_pipeline(
        experiment_id,
        job_name,
        pipeline_package_path=None,
        params=None,
        pipeline_id=None,
        version_id=None,
    ):
        return kfp_server_api.models.ApiRun(id="some-run-id", name=job_name)

    kfp_client_mock.create_experiment = _mock_create_experiment
    kfp_client_mock.run_pipeline = _mock_run_pipeline


def _mock_list_runs_with_one_run_per_page(kfp_client_mock: kfp.Client, runs):
    expected_page_tokens = [""]
    for i in range(2, len(runs) + 1):
        expected_page_tokens.append(i)
    expected_page_tokens.append(None)

    def list_runs_mock(*args, page_token=None, page_size=None, **kwargs):
        assert expected_page_tokens.pop(0) == page_token
        assert mlrun.common.schemas.PipelinesPagination.max_page_size == page_size
        return kfp_server_api.models.api_list_runs_response.ApiListRunsResponse(
            [runs.pop(0)], 1, next_page_token=expected_page_tokens[0]
        )

    kfp_client_mock.list_runs = list_runs_mock


def _mock_list_runs(
    kfp_client_mock: kfp.Client,
    runs,
    expected_page_token="",
    expected_page_size=mlrun.common.schemas.PipelinesPagination.default_page_size,
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

    kfp_client_mock.list_runs = list_runs_mock


def _mock_get_run(
    kfp_client_mock: kfp.Client,
    api_run_detail: kfp_server_api.models.api_run_detail.ApiRunDetail,
):
    def get_run_mock(*args, **kwargs):
        return api_run_detail

    kfp_client_mock.get_run = get_run_mock


def _assert_list_pipelines_response(
    expected_response: mlrun.common.schemas.PipelinesOutput, response
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


def _assert_get_pipeline_response(expected_response: dict, response):
    assert response.status_code == http.HTTPStatus.OK.value
    assert (
        deepdiff.DeepDiff(
            expected_response,
            response.json(),
            ignore_order=True,
        )
        == {}
    )
