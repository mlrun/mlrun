from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.api.utils import _parse_submit_job_body


def test_parse_submit_job_body(db: Session, client: TestClient):
    function_name = 'function_name'
    project = 'some_project'
    function_tag = 'function_tag'
    task_name = 'task_name'
    task_project = 'task_project'
    function = {
        "kind": "job",
        "metadata": {
            "name": function_name,
            "tag": function_tag,
            "project": project,
        },
        "spec": {},
    }

    resp = client.post(f"/api/func/{project}/{function_name}", json=function, params={'tag': function_tag})
    assert resp.status_code == HTTPStatus.OK.value
    body = {
        "task": {
            "spec": {
                "function": f"{project}/{function_name}:{function_tag}",
            },
            "metadata": {
                "name": task_name,
                "project": task_project,
            },
        },
        "function": {
            "spec": {
                "volumes": [
                    {
                        "name": "v3io-volume-name",
                        "flexVolume": {
                            "driver": "v3io/fuse",
                            "options": {
                                "container": "users",
                                "accessKey": "4dbc1521-f6f2-4b28-aeac-29073413b9ae",
                                "subPath": "/pipelines/.mlrun"
                            }
                        }
                    },
                    {
                        "name": "secret-volume-name",
                        "secret": {
                            "secretName": "secret-name"
                        }
                    }
                ],
                "volume_mounts": [
                    {
                        "name": "v3io-volume-name",
                        "mountPath": "/v3io/volume/mount/path"
                    },
                    {
                        "name": "secret-volume-name",
                        "mountPath": "/secret/volume/mount/path"
                    }
                ],
                "env": [
                    {
                        "name": "SOME_ENV_VAR_KEY",
                        "value": "some-env-var-value"
                    }
                ],
                "resources": {
                    "limits": {"cpu": "250m", "memory": "64Mi", "nvidia.com/gpu": "2"},
                    "requests": {"cpu": "200m", "memory": "32Mi"},
                },
            }
        },
    }
    function_object, task = _parse_submit_job_body(db, body)
    assert function_object.metadata.name == function_name
    assert function_object.metadata.project == project
    assert function_object.metadata.tag == function_tag
    assert function_object.spec.env == body['function']['spec']['env']
    assert function_object.spec.resources == body['function']['spec']['resources']
    assert function_object.spec.volumes == body['function']['spec']['volumes']
    assert function_object.spec.volume_mounts == body['function']['spec']['volume_mounts']
