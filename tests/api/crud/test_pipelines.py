import json

import mlrun.api.crud.pipelines
import mlrun.errors
import mlrun.run
import mlrun.utils.helpers


def test_resolve_pipeline_project():
    cases = [
        {
            "expected_project": "project-from-deploy-p",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "-p",
                        "project-from-deploy-p",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy--project",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "--project",
                        "project-from-deploy--project",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy-f",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "-f",
                        "db://project-from-deploy-f/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy--func-url",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "--func-url",
                        "db://project-from-deploy--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-deploy-precedence-p",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "deploy",
                        "--func-url",
                        "db://project-from-deploy--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                        "-p",
                        "project-from-deploy-precedence-p",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run--project",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--project",
                        "project-from-run--project",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run-f",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "-f",
                        "db://project-from-run-f/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run--func-url",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--func-url",
                        "db://project-from-run--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run-r",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "-r",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-run-r'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run--runtime",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--runtime",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-run--runtime'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-run-precedence--project",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "run",
                        "--func-url",
                        "db://project-from-deploy--func-url/tf2-serving@2db2ec7d89c0c8c9d1b9a86279d8440ebc230597",
                        "--project",
                        "project-from-run-precedence--project",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-build-r",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "build",
                        "-r",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-build-r'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-build--runtime",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "build",
                        "--runtime",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-build--runtime'}}",
                    ]
                }
            },
        },
        {
            "expected_project": "project-from-build-precedence--runtime",
            "template": {
                "container": {
                    "command": [
                        "python",
                        "-m",
                        "mlrun",
                        "build",
                        "--runtime",
                        "{'kind': 'job', 'metadata': {'project': 'project-from-build--runtime'}}",
                        "--project",
                        "project-from-build-precedence--runtime",
                    ]
                }
            },
        },
        {
            "expected_project": mlrun.mlconf.default_project,
            "template": {"dag": {"asdasd": "asdasd"}},
        },
    ]
    for case in cases:
        workflow_manifest = {"spec": {"templates": [case["template"]]}}
        pipeline = {
            "pipeline_spec": {"workflow_manifest": json.dumps(workflow_manifest)}
        }
        project = mlrun.api.crud.pipelines._resolve_pipeline_project(pipeline)
        assert project == case["expected_project"]
