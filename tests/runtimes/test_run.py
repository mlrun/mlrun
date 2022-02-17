import copy

from deepdiff import DeepDiff

import mlrun


def _get_runtime():
    runtime = {
        "kind": "job",
        "metadata": {
            "name": "spark-submit",
            "project": "default",
            "categories": [],
            "tag": "",
            "hash": "7b3064c6b334535a5d949ebe9cfc61a094f98c78",
            "updated": "2020-10-21T22:40:35.042132+00:00",
            "credentials": {"access_key": "some-access-key"},
        },
        "spec": {
            "command": "spark-submit",
            "args": [
                "--class",
                "org.apache.spark.examples.SparkPi",
                "/spark/examples/jars/spark-examples_2.11-2.4.4.jar",
            ],
            "image": "iguazio/shell:3.0_b5533_20201020062229",
            "mode": "pass",
            "volumes": [],
            "volume_mounts": [],
            "env": [],
            "description": "",
            "build": {"commands": []},
            "affinity": None,
            "disable_auto_mount": False,
            "priority_class_name": "",
        },
        "verbose": False,
    }
    return runtime


def test_new_function_from_runtime():
    runtime = _get_runtime()
    function = mlrun.new_function(runtime=runtime)
    default_resources = mlrun.mlconf.default_function_pod_resources.to_dict()
    runtime["spec"]["resources"] = default_resources
    assert DeepDiff(function.to_dict(), runtime, ignore_order=True,) == {}


def test_new_function_args_without_command():
    runtime = _get_runtime()
    runtime["spec"]["command"] = ""
    function = mlrun.new_function(runtime=runtime)
    default_resources = mlrun.mlconf.default_function_pod_resources.to_dict()
    runtime["spec"]["resources"] = default_resources
    assert DeepDiff(function.to_dict(), runtime, ignore_order=True,) == {}


def test_new_function_with_resources():
    runtime = _get_runtime()
    for test_case in [
        {
            "resources": {"requests": {"cpu": "50mi"}},
            "default_resources": {
                "requests": {"cpu": "25mi", "memory": "1M", "gpu": ""},
                "limits": {"cpu": "1", "memory": "1G", "gpu": ""},
            },
            "expected_resources": {
                "requests": {"cpu": "50mi", "memory": "1M", "gpu": ""},
                "limits": {"cpu": "1", "memory": "1G", "gpu": ""},
            },
        },
    ]:
        expected_runtime = copy.copy(runtime)
        expected_runtime["spec"]["resources"] = test_case.get("expected_resources")
        runtime["spec"]["resources"] = test_case.get("resources")
        mlrun.mlconf.default_function_pod_resources = test_case.get("default_resources")

        function = mlrun.new_function(runtime=runtime)
        assert DeepDiff(function.to_dict(), expected_runtime, ignore_order=True,) == {}
