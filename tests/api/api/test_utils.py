from http import HTTPStatus

from deepdiff import DeepDiff
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
from mlrun.api.api.utils import _parse_submit_run_body


def test_parse_submit_job_body_override_values(db: Session, client: TestClient):
    task_name = "task_name"
    task_project = "task-project"
    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": task_project},
        },
        "function": {
            "spec": {
                "volumes": [
                    {
                        "name": "override-volume-name",
                        "flexVolume": {
                            "driver": "v3io/fuse",
                            "options": {
                                "container": "users",
                                "accessKey": "4dbc1521-f6f2-4b28-aeac-29073413b9ae",
                                "subPath": "/pipelines/.mlrun",
                            },
                        },
                    },
                    {
                        "name": "new-volume-name",
                        "secret": {"secretName": "secret-name"},
                    },
                ],
                "volume_mounts": [
                    {
                        "name": "old-volume-name",
                        "mountPath": "/v3io/old/volume/mount/path",
                    },
                    {
                        "name": "override-volume-name",
                        "mountPath": "/v3io/volume/mount/path",
                    },
                    {
                        "name": "new-volume-name",
                        "mountPath": "/secret/volume/mount/path",
                    },
                ],
                "env": [
                    {"name": "OVERRIDE_ENV_VAR_KEY", "value": "new-env-var-value"},
                    {"name": "NEW_ENV_VAR_KEY", "value": "env-var-value"},
                ],
                "resources": {
                    "limits": {"cpu": "250m", "memory": "64Mi", "nvidia.com/gpu": "2"},
                    "requests": {"cpu": "200m", "memory": "32Mi"},
                },
                "image_pull_policy": "Always",
                "replicas": "3",
            }
        },
    }
    parsed_function_object, task = _parse_submit_run_body(db, submit_job_body)
    assert parsed_function_object.metadata.name == function_name
    assert parsed_function_object.metadata.project == project
    assert parsed_function_object.metadata.tag == function_tag
    assert (
        DeepDiff(
            parsed_function_object.spec.resources,
            submit_job_body["function"]["spec"]["resources"],
            ignore_order=True,
        )
        == {}
    )
    assert (
        parsed_function_object.spec.image_pull_policy
        == submit_job_body["function"]["spec"]["image_pull_policy"]
    )
    assert (
        parsed_function_object.spec.replicas
        == submit_job_body["function"]["spec"]["replicas"]
    )

    _assert_volumes_and_volume_mounts(
        parsed_function_object, submit_job_body, original_function
    )
    _assert_env_vars(parsed_function_object, submit_job_body, original_function)


def test_parse_submit_job_body_keep_resources(db: Session, client: TestClient):
    task_name = "task_name"
    task_project = "task-project"
    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": task_project},
        },
        "function": {"spec": {"resources": {"limits": {}, "requests": {}}}},
    }
    parsed_function_object, task = _parse_submit_run_body(db, submit_job_body)
    assert parsed_function_object.metadata.name == function_name
    assert parsed_function_object.metadata.project == project
    assert parsed_function_object.metadata.tag == function_tag
    assert (
        DeepDiff(
            parsed_function_object.spec.resources,
            submit_job_body["function"]["spec"]["resources"],
            ignore_order=True,
        )
        != {}
    )
    assert (
        DeepDiff(
            parsed_function_object.spec.resources,
            original_function["spec"]["resources"],
            ignore_order=True,
        )
        == {}
    )


def test_parse_submit_job_imported_function_project_assignment(
    db: Session, client: TestClient, monkeypatch
):
    task_name = "task_name"
    task_project = "task-project"
    _mock_import_function(monkeypatch)
    submit_job_body = {
        "task": {
            "spec": {"function": "hub://gen_class_data"},
            "metadata": {"name": task_name, "project": task_project},
        },
        "function": {"spec": {"resources": {"limits": {}, "requests": {}}}},
    }
    parsed_function_object, task = _parse_submit_run_body(db, submit_job_body)
    assert parsed_function_object.metadata.project == task_project


def test_get_obj_path(db: Session, client: TestClient):
    cases = [
        {"path": "/local/path", "expected_path": "/local/path"},
        {
            "path": "/local/path",
            "schema": "v3io",
            "expected_path": "v3io:///local/path",
        },
        {"path": "/User/my/path", "expected_path": "v3io:///users/admin/my/path"},
        {
            "path": "/User/my/path",
            "schema": "v3io",
            "expected_path": "v3io:///users/admin/my/path",
        },
        {
            "path": "/User/my/path",
            "user": "hedi",
            "expected_path": "v3io:///users/hedi/my/path",
        },
        {
            "path": "/v3io/projects/my-proj/my/path",
            "expected_path": "v3io:///projects/my-proj/my/path",
        },
        {
            "path": "/v3io/projects/my-proj/my/path",
            "schema": "v3io",
            "expected_path": "v3io:///projects/my-proj/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data",
            "expected_path": "/home/jovyan/data/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data",
            "real_path": "/root",
            "expected_path": "/root/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data/",
            "real_path": "/root",
            "expected_path": "/root/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data/",
            "real_path": "/root/",
            "expected_path": "/root/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data",
            "real_path": "/root",
            "expected_path": "/root/my/path",
        },
    ]
    for case in cases:
        old_real_path = mlrun.mlconf.httpdb.real_path
        old_data_volume = mlrun.mlconf.httpdb.data_volume
        if case.get("real_path"):
            mlrun.mlconf.httpdb.real_path = case["real_path"]
        if case.get("data_volume"):
            mlrun.mlconf.httpdb.data_volume = case["data_volume"]
        result_path = mlrun.api.api.utils.get_obj_path(
            case.get("schema"), case.get("path"), case.get("user")
        )
        assert result_path == case["expected_path"]
        if case.get("real_path"):
            mlrun.mlconf.httpdb.real_path = old_real_path
        if case.get("data_volume"):
            mlrun.mlconf.httpdb.data_volume = old_data_volume


def _mock_import_function(monkeypatch):
    def _mock_import_function_to_dict(*args, **kwargs):
        _, _, _, original_function = _generate_original_function()
        return original_function

    monkeypatch.setattr(
        mlrun.run, "import_function_to_dict", _mock_import_function_to_dict
    )


def _mock_original_function(client):
    (
        project,
        function_name,
        function_tag,
        original_function,
    ) = _generate_original_function()
    resp = client.post(
        f"/api/func/{project}/{function_name}",
        json=original_function,
        params={"tag": function_tag},
    )
    assert resp.status_code == HTTPStatus.OK.value
    return project, function_name, function_tag, original_function


def _generate_original_function():
    function_name = "function_name"
    project = "some-project"
    function_tag = "function_tag"
    original_function = {
        "kind": "job",
        "metadata": {"name": function_name, "tag": function_tag, "project": project},
        "spec": {
            "volumes": [
                {
                    "name": "old-volume-name",
                    "flexVolume": {
                        "driver": "v3io/fuse",
                        "options": {
                            "container": "bigdata",
                            "accessKey": "1acf6fa2-f3b3-4c37-a9c6-759e555e0018",
                            "subPath": "/admin/data",
                        },
                    },
                },
                {
                    "name": "override-volume-name",
                    "flexVolume": {
                        "driver": "v3io/fuse",
                        "options": {
                            "container": "bigdata",
                            "accessKey": "c7f736ec-567b-42eb-b7c0-1aea8a66f880",
                            "subPath": "/iguazio/.db",
                        },
                    },
                },
            ],
            "volume_mounts": [
                {
                    "name": "old-volume-name",
                    "mountPath": "/v3io/old/volume/mount/path",
                },
                {
                    "name": "override-volume-name",
                    "mountPath": "/v3io/override/volume/mount/path",
                },
            ],
            "resources": {
                "limits": {"cpu": "40m", "memory": "128Mi", "nvidia.com/gpu": "7"},
                "requests": {"cpu": "15m", "memory": "86Mi"},
            },
            "env": [
                {"name": "OLD_ENV_VAR_KEY", "value": "old-env-var-value"},
                {"name": "OVERRIDE_ENV_VAR_KEY", "value": "override-env-var-value"},
            ],
            "image_pull_policy": "IfNotPresent",
            "replicas": "1",
        },
    }
    return project, function_name, function_tag, original_function


def _assert_volumes_and_volume_mounts(
    parsed_function_object, submit_job_body, original_function
):
    """
    expected volumes and volume mounts:
    0: old volume from original function (the first one there)
    1: volume that was in original function but was overridden with body (the first one in the body)
    2: new volume from the body (the second in the body)
    """
    assert (
        DeepDiff(
            original_function["spec"]["volumes"][0],
            parsed_function_object.spec.volumes[0],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            original_function["spec"]["volumes"][1],
            parsed_function_object.spec.volumes[1],
            ignore_order=True,
        )
        != {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volumes"][0],
            parsed_function_object.spec.volumes[1],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volumes"][1],
            parsed_function_object.spec.volumes[2],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            original_function["spec"]["volume_mounts"][0],
            parsed_function_object.spec.volume_mounts[0],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            original_function["spec"]["volume_mounts"][1],
            parsed_function_object.spec.volume_mounts[1],
            ignore_order=True,
        )
        != {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volume_mounts"][0],
            parsed_function_object.spec.volume_mounts[0],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volume_mounts"][1],
            parsed_function_object.spec.volume_mounts[1],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volume_mounts"][2],
            parsed_function_object.spec.volume_mounts[2],
            ignore_order=True,
        )
        == {}
    )


def _assert_env_vars(parsed_function_object, submit_job_body, original_function):
    """
    expected env vars:
    0: old env var from original function (the first one there)
    1: env var that was in original function but was overridden with body (the first one in the body)
    2: new env var from the body (the second in the body)
    """
    assert (
        original_function["spec"]["env"][0]["name"]
        == parsed_function_object.spec.env[0]["name"]
    )
    assert (
        original_function["spec"]["env"][0]["value"]
        == parsed_function_object.spec.env[0]["value"]
    )

    assert (
        original_function["spec"]["env"][1]["name"]
        == parsed_function_object.spec.env[1].name
    )
    assert (
        submit_job_body["function"]["spec"]["env"][0]["name"]
        == parsed_function_object.spec.env[1].name
    )
    assert (
        original_function["spec"]["env"][1]["value"]
        != parsed_function_object.spec.env[1].value
    )
    assert (
        submit_job_body["function"]["spec"]["env"][0]["value"]
        == parsed_function_object.spec.env[1].value
    )

    assert (
        submit_job_body["function"]["spec"]["env"][1]["name"]
        == parsed_function_object.spec.env[2].name
    )
    assert (
        submit_job_body["function"]["spec"]["env"][1]["value"]
        == parsed_function_object.spec.env[2].value
    )
