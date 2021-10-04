from deepdiff import DeepDiff

import mlrun


def test_new_function_from_runtime():
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

    function = mlrun.new_function(runtime=runtime)
    assert DeepDiff(runtime, function.to_dict(), ignore_order=True,) == {}


def test_new_function_args_without_command():
    runtime = {
        "kind": "job",
        "metadata": {
            "name": "spark-submit",
            "project": "default",
            "categories": [],
            "tag": "",
            "hash": "7b3064c6b334535a5d949ebe9cfc61a094f98c78",
            "updated": "2020-10-21T22:40:35.042132+00:00",
        },
        "spec": {
            "command": "",
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
    function = mlrun.new_function(runtime=runtime)
    assert DeepDiff(runtime, function.to_dict(), ignore_order=True,) == {}
