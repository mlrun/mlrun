# Copyright 2018 Iguazio
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

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s
import tests.conftest


@pytest.fixture
def kfp_client_mock(monkeypatch) -> kfp.Client:
    mlrun.api.utils.singletons.k8s.get_k8s().is_running_inside_kubernetes_cluster = (
        unittest.mock.Mock(return_value=True)
    )
    kfp_client_mock = unittest.mock.Mock()
    monkeypatch.setattr(kfp, "Client", lambda *args, **kwargs: kfp_client_mock)
    mlrun.mlconf.kfp_url = "http://ml-pipeline.custom_namespace.svc.cluster.local:8888"
    return kfp_client_mock


def test_list_pipelines_not_exploding_on_no_k8s(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("projects/*/pipelines")
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
    response = client.get("projects/*/pipelines")
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=runs, total_size=len(runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)


def test_list_pipelines_formats(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    for format_ in [
        mlrun.api.schemas.PipelinesFormat.full,
        mlrun.api.schemas.PipelinesFormat.metadata_only,
        mlrun.api.schemas.PipelinesFormat.name_only,
    ]:
        runs = _generate_list_runs_mocks()
        expected_runs = [run.to_dict() for run in runs]
        expected_runs = mlrun.api.crud.Pipelines()._format_runs(
            db, expected_runs, format_
        )
        _mock_list_runs(kfp_client_mock, runs)
        response = client.get(
            "projects/*/pipelines",
            params={"format": format_},
        )
        expected_response = mlrun.api.schemas.PipelinesOutput(
            runs=expected_runs, total_size=len(runs), next_page_token=None
        )
        _assert_list_pipelines_response(expected_response, response)


def test_get_pipeline_formats(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    for format_ in [
        mlrun.api.schemas.PipelinesFormat.full,
        mlrun.api.schemas.PipelinesFormat.metadata_only,
        mlrun.api.schemas.PipelinesFormat.summary,
        mlrun.api.schemas.PipelinesFormat.name_only,
    ]:
        api_run_detail = _generate_get_run_mock()
        _mock_get_run(kfp_client_mock, api_run_detail)
        response = client.get(
            f"projects/*/pipelines/{api_run_detail.run.id}",
            params={"format": format_},
        )
        expected_run = mlrun.api.crud.Pipelines()._format_run(
            db, api_run_detail.to_dict()["run"], format_, api_run_detail.to_dict()
        )
        _assert_get_pipeline_response(expected_run, response)


def test_get_pipeline_no_project_opa_validation(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    format_ = (mlrun.api.schemas.PipelinesFormat.summary,)
    project = "project-name"
    mlrun.api.crud.Pipelines().resolve_project_from_pipeline = unittest.mock.Mock(
        return_value=project
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions = (
        unittest.mock.AsyncMock()
    )
    api_run_detail = _generate_get_run_mock()
    _mock_get_run(kfp_client_mock, api_run_detail)
    response = client.get(
        f"projects/*/pipelines/{api_run_detail.run.id}",
        params={"format": format_},
    )
    assert (
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions.call_args[
            0
        ][
            1
        ]
        == project
    )
    assert response.json()["run"]["project"] == project


def test_get_pipeline_specific_project(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    for format_ in [
        mlrun.api.schemas.PipelinesFormat.full,
        mlrun.api.schemas.PipelinesFormat.metadata_only,
        mlrun.api.schemas.PipelinesFormat.summary,
        mlrun.api.schemas.PipelinesFormat.name_only,
    ]:
        project = "project-name"
        api_run_detail = _generate_get_run_mock()
        _mock_get_run(kfp_client_mock, api_run_detail)
        mlrun.api.crud.Pipelines().resolve_project_from_pipeline = unittest.mock.Mock(
            return_value=project
        )
        response = client.get(
            f"projects/{project}/pipelines/{api_run_detail.run.id}",
            params={"format": format_},
        )
        expected_run = mlrun.api.crud.Pipelines()._format_run(
            db, api_run_detail.to_dict()["run"], format_, api_run_detail.to_dict()
        )
        _assert_get_pipeline_response(expected_run, response)

        # revert mock setting (it's global function, without reloading it the mock will persist to following tests)
        importlib.reload(mlrun.api.crud)


def test_list_pipelines_specific_project(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    project = "project-name"
    runs = _generate_list_runs_mocks()
    expected_runs = [run.name for run in runs]
    _mock_list_runs_with_one_run_per_page(kfp_client_mock, runs)
    mlrun.api.crud.Pipelines().resolve_project_from_pipeline = unittest.mock.Mock(
        return_value=project
    )
    response = client.get(
        f"projects/{project}/pipelines",
        params={"format": mlrun.api.schemas.PipelinesFormat.name_only},
    )
    expected_response = mlrun.api.schemas.PipelinesOutput(
        runs=expected_runs, total_size=len(expected_runs), next_page_token=None
    )
    _assert_list_pipelines_response(expected_response, response)

    # revert mock setting (it's global function, without reloading it the mock will persist to following tests)
    importlib.reload(mlrun.api.crud)


def test_create_pipeline(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    project = "some-project"
    pipeline_file_path = (
        tests.conftest.tests_root_directory
        / "api"
        / "api"
        / "assets"
        / "pipelines.yaml"
    )
    with open(str(pipeline_file_path), "r") as file:
        contents = file.read()
    _mock_pipelines_creation(kfp_client_mock)
    response = client.post(
        f"projects/{project}/pipelines",
        data=contents,
        headers={"content-type": "application/yaml"},
    )
    response_body = response.json()
    assert response_body["id"] == "some-run-id"


def test_create_pipeline_legacy(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    kfp_client_mock: kfp.Client,
) -> None:
    pipeline_file_path = (
        tests.conftest.tests_root_directory
        / "api"
        / "api"
        / "assets"
        / "pipelines.yaml"
    )
    with open(str(pipeline_file_path), "r") as file:
        contents = file.read()
    _mock_pipelines_creation(kfp_client_mock)
    response = client.post(
        "submit_pipeline",
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
            pipeline_spec=kfp_server_api.models.api_pipeline_spec.ApiPipelineSpec(
                pipeline_id="pipe_id1",
                workflow_manifest=workflow_manifest,
            ),
        ),
        pipeline_runtime=kfp_server_api.models.api_pipeline_runtime.ApiPipelineRuntime(
            workflow_manifest=workflow_manifest_with_status
        ),
    )


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
        params={},
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


def _mock_get_run(
    kfp_client_mock: kfp.Client,
    api_run_detail: kfp_server_api.models.api_run_detail.ApiRunDetail,
):
    def get_run_mock(*args, **kwargs):
        return api_run_detail

    kfp_client_mock.get_run = get_run_mock


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
