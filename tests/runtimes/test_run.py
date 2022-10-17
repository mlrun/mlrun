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
            "tolerations": None,
            "security_context": None,
        },
        "verbose": False,
    }
    return runtime


def test_new_function_from_runtime():
    runtime = _get_runtime()
    function = mlrun.new_function(runtime=runtime)
    expected_runtime = runtime
    expected_runtime["spec"][
        "preemption_mode"
    ] = mlrun.mlconf.function_defaults.preemption_mode
    assert (
        DeepDiff(
            function.to_dict(),
            expected_runtime,
            ignore_order=True,
        )
        == {}
    )


def test_new_function_args_without_command():
    runtime = _get_runtime()
    runtime["spec"]["command"] = ""
    function = mlrun.new_function(runtime=runtime)
    expected_runtime = runtime
    expected_runtime["spec"][
        "preemption_mode"
    ] = mlrun.mlconf.function_defaults.preemption_mode
    assert (
        DeepDiff(
            function.to_dict(),
            expected_runtime,
            ignore_order=True,
        )
        == {}
    )


def test_new_function_with_resources():
    runtime = _get_runtime()
    for test_case in [
        {
            "resources": {"requests": {"cpu": "50mi"}},
            "default_resources": {
                "requests": {
                    "cpu": "25mi",
                    "memory": "1M",
                    "gpu": None,
                    "ephemeral_storage": None,
                },
                "limits": {
                    "cpu": "1",
                    "memory": "1G",
                    "gpu": None,
                    "ephemeral_storage": None,
                },
            },
            "expected_resources": {
                "requests": {"cpu": "50mi", "memory": "1M"},
                "limits": {"cpu": "1", "memory": "1G"},
            },
        },
        {
            "resources": {"requests": {"cpu": "50mi"}},
            "default_resources": {
                "requests": {
                    "cpu": "25mi",
                    "memory": "1M",
                    "gpu": "1",
                    "ephemeral_storage": None,
                },
                "limits": {
                    "cpu": "1",
                    "memory": "1G",
                    "gpu": "1",
                    "ephemeral_storage": None,
                },
            },
            "expected_resources": {
                "requests": {"cpu": "50mi", "memory": "1M"},
                "limits": {"cpu": "1", "memory": "1G"},
            },
        },
        {
            "resources": {
                "requests": {"cpu": "50mi"},
                "limits": {"nvidia.com/gpu": "1"},
            },
            "default_resources": {
                "requests": {"cpu": "25mi", "memory": "1M", "ephemeral_storage": None},
                "limits": {"cpu": "1", "memory": "1G", "ephemeral_storage": None},
            },
            "expected_resources": {
                "requests": {"cpu": "50mi", "memory": "1M"},
                "limits": {"cpu": "1", "memory": "1G", "nvidia.com/gpu": "1"},
            },
        },
    ]:
        expected_runtime = copy.deepcopy(runtime)
        expected_runtime["spec"]["resources"] = test_case.get("expected_resources")
        expected_runtime["spec"][
            "preemption_mode"
        ] = mlrun.mlconf.function_defaults.preemption_mode
        runtime["spec"]["resources"] = test_case.get("resources", None)
        mlrun.mlconf.default_function_pod_resources = test_case.get("default_resources")
        function = mlrun.new_function(runtime=runtime)
        assert (
            DeepDiff(
                function.to_dict(),
                expected_runtime,
                ignore_order=True,
            )
            == {}
        )


def test_with_requests():
    runtime = _get_runtime()
    runtime["spec"]["resources"] = {"limits": {"cpu": "20", "memory": "10G"}}
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {
            "cpu": "25mi",
            "memory": "1M",
            "gpu": None,
            "ephemeral_storage": None,
        },
        "limits": {"cpu": "1", "memory": "1G", "gpu": None, "ephemeral_storage": None},
    }
    function = mlrun.new_function(runtime=runtime)
    function.with_requests(mem="9G", cpu="15")
    expected = {
        "requests": {"cpu": "15", "memory": "9G"},
        "limits": {"cpu": "20", "memory": "10G"},
    }
    assert (
        DeepDiff(
            function.spec.resources,
            expected,
            ignore_order=True,
        )
        == {}
    )


def test_with_request_patch():
    runtime = _get_runtime()
    runtime["spec"]["resources"] = {"requests": {"cpu": "50mi"}}
    function = mlrun.new_function(runtime=runtime)
    function.with_requests(mem="9G", patch=True)
    expected = {
        "requests": {
            "cpu": "50mi",
            "memory": "9G",
        },
        "limits": {},
    }
    assert (
        DeepDiff(
            function.spec.resources,
            expected,
            ignore_order=True,
        )
        == {}
    )
    function.with_requests(cpu="15")  # default patch = False
    expected = {
        "requests": {"cpu": "15"},
        "limits": {},
    }
    assert (
        DeepDiff(
            function.spec.resources,
            expected,
            ignore_order=True,
        )
        == {}
    )


def test_with_limits_patch():
    runtime = _get_runtime()
    runtime["spec"]["resources"] = {"requests": {"cpu": "50mi"}}
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {
            "cpu": "25mi",
            "memory": "1M",
            "gpu": None,
            "ephemeral_storage": None,
        },
        "limits": {"cpu": "1", "memory": "1G", "gpu": None, "ephemeral_storage": None},
    }
    function = mlrun.new_function(runtime=runtime)
    function.with_limits(mem="9G", patch=True)
    expected = {
        "requests": {"cpu": "50mi", "memory": "1M"},
        "limits": {"cpu": "1", "memory": "9G"},
    }
    assert (
        DeepDiff(
            function.spec.resources,
            expected,
            ignore_order=True,
        )
        == {}
    )

    function.with_limits(mem="9G")  # default patch = False
    expected = {
        "requests": {"cpu": "50mi", "memory": "1M"},
        "limits": {"memory": "9G"},
    }
    assert (
        DeepDiff(
            function.spec.resources,
            expected,
            ignore_order=True,
        )
        == {}
    )


def test_with_limits():
    runtime = _get_runtime()
    runtime["spec"]["resources"] = {"requests": {"cpu": "50mi"}}
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {
            "cpu": "25mi",
            "memory": "1M",
            "gpu": None,
            "ephemeral_storage": None,
        },
        "limits": {"cpu": "1", "memory": "1G", "gpu": None, "ephemeral_storage": None},
    }
    function = mlrun.new_function(runtime=runtime)
    function.with_limits(mem="9G", cpu="15")
    expected = {
        "requests": {"cpu": "50mi", "memory": "1M"},
        "limits": {"cpu": "15", "memory": "9G"},
    }
    assert (
        DeepDiff(
            function.spec.resources,
            expected,
            ignore_order=True,
        )
        == {}
    )


def test_new_function_args_with_default_image_pull_secret():
    mlrun.mlconf.function.spec.image_pull_secret.default = "my_secret"
    runtime = _get_runtime()
    function = mlrun.new_function(runtime=runtime)
    expected_runtime = runtime
    expected_runtime["spec"][
        "image_pull_secret"
    ] = mlrun.mlconf.function.spec.image_pull_secret.default
    expected_runtime["spec"][
        "preemption_mode"
    ] = mlrun.mlconf.function_defaults.preemption_mode
    assert (
        DeepDiff(
            function.to_dict(),
            expected_runtime,
            ignore_order=True,
        )
        == {}
    )


def test_new_function_override_default_image_pull_secret():
    mlrun.mlconf.function.spec.image_pull_secret.default = "my_secret"
    runtime = _get_runtime()
    new_secret = "another_secret"
    runtime["spec"]["image_pull_secret"] = new_secret
    function = mlrun.new_function(runtime=runtime)
    expected_runtime = runtime
    expected_runtime["spec"]["image_pull_secret"] = new_secret
    expected_runtime["spec"][
        "preemption_mode"
    ] = mlrun.mlconf.function_defaults.preemption_mode
    assert (
        DeepDiff(
            function.to_dict(),
            expected_runtime,
            ignore_order=True,
        )
        == {}
    )
