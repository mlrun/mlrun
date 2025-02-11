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
import base64
import inspect
import json
import warnings

import kubernetes.client as k8s_client
import pytest
from deepdiff import DeepDiff

import mlrun
import mlrun.runtimes.databricks_job.databricks_runtime
import mlrun.runtimes.mpijob.abstract
import mlrun.runtimes.mpijob.v1
import mlrun.runtimes.pod


@pytest.mark.parametrize(
    "method, base_classes",
    [
        ("__init__", []),
        ("run", [mlrun.runtimes.base.BaseRuntime]),
    ],
)
def test_runtimes_inheritance(method, base_classes):
    classes_map = {
        mlrun.runtimes.base.FunctionSpec: [
            mlrun.runtimes.daskjob.DaskSpec,
            mlrun.runtimes.nuclio.function.NuclioSpec,
            mlrun.runtimes.nuclio.serving.ServingSpec,
            mlrun.runtimes.mpijob.abstract.MPIResourceSpec,
            mlrun.runtimes.mpijob.v1.MPIV1ResourceSpec,
            mlrun.runtimes.remotesparkjob.RemoteSparkSpec,
            mlrun.runtimes.sparkjob.spark3job.Spark3JobSpec,
        ],
        mlrun.runtimes.pod.KubeResourceSpec: [
            mlrun.runtimes.daskjob.DaskSpec,
            mlrun.runtimes.nuclio.function.NuclioSpec,
            mlrun.runtimes.nuclio.serving.ServingSpec,
            mlrun.runtimes.mpijob.abstract.MPIResourceSpec,
            mlrun.runtimes.mpijob.v1.MPIV1ResourceSpec,
            mlrun.runtimes.remotesparkjob.RemoteSparkSpec,
            mlrun.runtimes.sparkjob.spark3job.Spark3JobSpec,
        ],
        mlrun.runtimes.nuclio.function.NuclioSpec: [
            mlrun.runtimes.nuclio.serving.ServingSpec,
        ],
        mlrun.runtimes.base.FunctionStatus: [
            mlrun.runtimes.daskjob.DaskStatus,
            mlrun.runtimes.nuclio.function.NuclioStatus,
        ],
        mlrun.runtimes.base.BaseRuntime: [
            mlrun.runtimes.local.HandlerRuntime,
            mlrun.runtimes.local.BaseRuntime,
            mlrun.runtimes.nuclio.function.RemoteRuntime,
            mlrun.runtimes.nuclio.serving.ServingRuntime,
            mlrun.runtimes.kubejob.KubejobRuntime,
            mlrun.runtimes.daskjob.DaskCluster,
            mlrun.runtimes.mpijob.v1.MpiRuntimeV1,
            mlrun.runtimes.remotesparkjob.RemoteSparkRuntime,
            mlrun.runtimes.sparkjob.spark3job.Spark3Runtime,
            mlrun.runtimes.databricks_job.databricks_runtime.DatabricksRuntime,
        ],
    }
    if base_classes:
        # filter classes_map entries by base_classes
        classes_map = dict(
            filter(lambda pair: pair[0] in base_classes, classes_map.items())
        )

    invalid_classes = {}
    for base_class, inheriting_classes in classes_map.items():
        checked_classes = set()
        for inheriting_class in inheriting_classes:
            for class_ in inspect.getmro(inheriting_class):
                if base_class == class_:
                    break
                if class_ in checked_classes:
                    continue
                class_kwargs = list(
                    inspect.signature(getattr(class_, method)).parameters.keys()
                )
                base_class_kwargs = list(
                    inspect.signature(getattr(base_class, method)).parameters.keys()
                )
                if not set(base_class_kwargs).issubset(class_kwargs):
                    invalid_classes[inheriting_class] = list(
                        set(base_class_kwargs)
                        - set(base_class_kwargs).intersection(class_kwargs)
                    )
                checked_classes.add(inheriting_class)
    if invalid_classes:
        pytest.fail(
            f"Found classes that are not accepting all of their parent classes kwargs: {invalid_classes}"
        )


def test_resource_enrichment_in_resource_spec_initialization():
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {"cpu": "25m", "memory": "1M"},
        "limits": {"cpu": "2", "memory": "1G"},
    }
    expected_resources = {
        "requests": {"cpu": "25m", "memory": "1M"},
        "limits": {"cpu": "2", "memory": "1G"},
    }

    # without setting resources
    spec = mlrun.runtimes.pod.KubeResourceSpec()
    assert (
        DeepDiff(
            spec.resources,
            expected_resources,
            ignore_order=True,
        )
        == {}
    )

    # setting partial requests
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {"cpu": "25m", "memory": "1M"},
        "limits": {"cpu": "2", "memory": "1G"},
    }
    expected_resources = {
        "requests": {"cpu": "1", "memory": "1M"},
        "limits": {"cpu": "2", "memory": "1G"},
    }
    spec_requests = {"cpu": "1"}
    spec = mlrun.runtimes.pod.KubeResourceSpec(resources={"requests": spec_requests})
    assert (
        DeepDiff(
            spec.resources,
            expected_resources,
            ignore_order=True,
        )
        == {}
    )

    # setting partial requests and limits
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {"cpu": "25m", "memory": "1M"},
        "limits": {"cpu": "2", "memory": "1G"},
    }
    expected_resources = {
        "requests": {"cpu": "1", "memory": "500M"},
        "limits": {"cpu": "2", "memory": "2G"},
    }

    spec_requests = {"cpu": "1", "memory": "500M"}
    spec_limits = {"memory": "2G"}
    spec = mlrun.runtimes.pod.KubeResourceSpec(
        resources={"requests": spec_requests, "limits": spec_limits}
    )
    assert (
        DeepDiff(
            spec.resources,
            expected_resources,
            ignore_order=True,
        )
        == {}
    )

    # setting resource not in the k8s resources patterns
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        spec_requests = {"cpu": "1wrong"}
        mlrun.runtimes.pod.KubeResourceSpec(
            resources={"requests": spec_requests, "limits": spec_limits}
        )

    # setting partial requests and limits with equal gpus
    mlrun.mlconf.default_function_pod_resources = {
        "requests": {"cpu": "25m", "memory": "1M"},
        "limits": {"cpu": "2", "memory": "1G"},
    }
    expected_resources = {
        "requests": {"cpu": "25m", "memory": "1M"},
        "limits": {"cpu": "2", "memory": "1G", "nvidia.com/gpu": "2"},
    }
    spec_requests = {"nvidia.com/gpu": "2"}
    spec_limits = {"nvidia.com/gpu": "2"}

    spec = mlrun.runtimes.pod.KubeResourceSpec(
        resources={"requests": spec_requests, "limits": spec_limits}
    )

    assert (
        DeepDiff(
            spec.resources,
            expected_resources,
            ignore_order=True,
        )
        == {}
    )


def test_to_dict():
    volume_mount = k8s_client.V1VolumeMount(mount_path="some-path", name="volume-name")
    function = mlrun.new_function(kind=mlrun.runtimes.RuntimeKinds.job)
    # For sanitization
    function.spec.volume_mounts = [volume_mount]
    # For enrichment
    function.set_env(name="V3IO_ACCESS_KEY", value="123")
    # For apply enrichment before to_dict completion
    function.spec.disable_auto_mount = True

    function_dict = function.to_dict()
    assert function_dict["spec"]["volume_mounts"][0]["mountPath"] == "some-path"
    assert function_dict["spec"]["env"][0]["name"] == "V3IO_ACCESS_KEY"
    assert function_dict["spec"]["env"][0]["value"] == "123"
    assert function_dict["spec"]["disable_auto_mount"] is True

    stripped_function_dict = function.to_dict(strip=True)
    assert "volume_mounts" not in stripped_function_dict["spec"]
    assert stripped_function_dict["spec"]["env"][0]["name"] == "V3IO_ACCESS_KEY"
    assert stripped_function_dict["spec"]["env"][0]["value"] == ""
    assert stripped_function_dict["spec"]["disable_auto_mount"] is False

    excluded_function_dict = function.to_dict(exclude=["spec"])
    assert "spec" not in excluded_function_dict


def test_volume_mounts_addition():
    volume_mount = k8s_client.V1VolumeMount(mount_path="some-path", name="volume-name")
    dict_volume_mount = volume_mount.to_dict()
    sanitized_dict_volume_mount = k8s_client.ApiClient().sanitize_for_serialization(
        volume_mount
    )
    function = mlrun.new_function(kind=mlrun.runtimes.RuntimeKinds.job)
    function.spec.volume_mounts = [
        volume_mount,
        dict_volume_mount,
        sanitized_dict_volume_mount,
    ]
    assert len(function.spec.volume_mounts) == 1


def test_build_config_with_multiple_commands():
    image = "mlrun/mlrun"
    fn = mlrun.new_function(
        "some-function", "some-project", "some-tag", image=image, kind="job"
    )
    fn.build_config(commands=["pip install pandas", "pip install numpy"])
    assert len(fn.spec.build.commands) == 2

    fn.build_config(commands=["pip install pandas"])
    assert len(fn.spec.build.commands) == 2


def test_build_config_preserve_order():
    function = mlrun.new_function("some-function", kind="job")
    # run a lot of times as order change
    commands = []
    for index in range(10):
        commands.append(str(index))
    # when using un-stable (doesn't preserve order) methods to make a list unique (like list(set(x))) it's random
    # whether the order will be preserved, therefore run in a loop
    for _ in range(100):
        function.spec.build.commands = []
        function.build_config(commands=commands)
        assert function.spec.build.commands == commands


# Common Preemptible Affinity Terms
preemptible_affinity_iguazio = [
    [
        k8s_client.V1NodeSelectorRequirement(
            key="app.iguazio.com/lifecycle", operator="In", values=["preemptible"]
        )
    ]
]

preemptible_affinity_cloud_provider = [
    [
        k8s_client.V1NodeSelectorRequirement(
            key="cloud.google.com/gke-spot", operator="In", values=["true"]
        )
    ]
]


def create_node_affinity_with_terms(terms):
    """Helper function to create a V1Affinity with specific node selector terms."""
    return k8s_client.V1Affinity(
        node_affinity=k8s_client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=k8s_client.V1NodeSelector(
                node_selector_terms=[
                    k8s_client.V1NodeSelectorTerm(match_expressions=term)
                    for term in terms
                ]
            )
        )
    )


def mock_preemptible_config():
    """Fixture to set up mock preemptible configurations before each test."""
    mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
        json.dumps(
            {
                "app.iguazio.com/lifecycle": "preemptible",
                "cloud.google.com/gke-spot": "true",
            }
        ).encode("utf-8")
    )
    mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
        json.dumps(
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "effect": "NoSchedule",
                }
            ]
        ).encode("utf-8")
    )


@pytest.mark.parametrize(
    "mode, tolerations, node_selector, affinity, expected_tolerations, expected_node_selector, expected_affinity",
    [
        # Mode "none" – no change.
        (
            "none",
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "effect": "NoSchedule",
                },
                {
                    "key": "some-key",
                    "value": "true",
                    "operator": "Equal",
                    "effect": "NoSchedule",
                },
            ],
            {"user-node-selector": "some-value", "cloud.google.com/gke-spot": "true"},
            k8s_client.V1Affinity(),
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "toleration_seconds": None,
                    "effect": "NoSchedule",
                },
                {
                    "key": "some-key",
                    "value": "true",
                    "operator": "Equal",
                    "toleration_seconds": None,
                    "effect": "NoSchedule",
                },
            ],
            {"user-node-selector": "some-value", "cloud.google.com/gke-spot": "true"},
            k8s_client.V1Affinity(),
        ),
        # Mode "prevent" – preemptible settings are removed, affinity is cleared.
        (
            "prevent",
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "effect": "NoSchedule",
                },
                {
                    "key": "some-key",
                    "value": "true",
                    "operator": "Equal",
                    "effect": "NoSchedule",
                },
            ],
            {
                "app.iguazio.com/lifecycle": "preemptible",
                "some-node-selector": "some-value",
            },
            create_node_affinity_with_terms(preemptible_affinity_iguazio),
            [
                {
                    "key": "some-key",
                    "value": "true",
                    "operator": "Equal",
                    "toleration_seconds": None,
                    "effect": "NoSchedule",
                },
            ],
            {"some-node-selector": "some-value"},
            None,
        ),
        # Mode "constrain" – preemptible toleration is merged, affinity is set to preemptible.
        (
            "constrain",
            [],
            {"user-node-selector": "some-value"},
            None,
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "toleration_seconds": None,
                    "effect": "NoSchedule",
                },
            ],
            {"user-node-selector": "some-value"},
            create_node_affinity_with_terms(
                preemptible_affinity_iguazio + preemptible_affinity_cloud_provider
            ),
        ),
        # Mode "allow" – conflicting node selector settings are purged.
        (
            "allow",
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "effect": "NoSchedule",
                }
            ],
            {
                "user-node-selector": "some-value",
                "app.iguazio.com/lifecycle": "preemptible",
            },
            create_node_affinity_with_terms(
                preemptible_affinity_iguazio + preemptible_affinity_cloud_provider
            ),
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "toleration_seconds": None,
                    "effect": "NoSchedule",
                }
            ],
            {
                "user-node-selector": "some-value",
                "app.iguazio.com/lifecycle": "preemptible",
            },
            create_node_affinity_with_terms(
                preemptible_affinity_iguazio + preemptible_affinity_cloud_provider
            ),
        ),
        # Mode "allow" with no preemptible toleration.
        (
            "allow",
            [],
            {"user-node-selector": "some-value"},
            None,
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "toleration_seconds": None,
                    "effect": "NoSchedule",
                }
            ],
            {"user-node-selector": "some-value"},
            None,
        ),
        # Mode "prevent" without initial tolerations.
        (
            "prevent",
            [],
            {
                "user-node-selector": "some-value",
                "app.iguazio.com/lifecycle": "preemptible",
            },
            None,
            [],
            {"user-node-selector": "some-value"},
            None,
        ),
        # Mode "constrain" with tolerations already set.
        (
            "constrain",
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "effect": "NoSchedule",
                }
            ],
            {"user-node-selector": "some-value"},
            None,
            [
                {
                    "key": "cloud.google.com/gke-spot",
                    "value": "true",
                    "operator": "Equal",
                    "toleration_seconds": None,
                    "effect": "NoSchedule",
                }
            ],
            {"user-node-selector": "some-value"},
            create_node_affinity_with_terms(
                preemptible_affinity_iguazio + preemptible_affinity_cloud_provider
            ),
        ),
        # Mode "none" with no tolerations or selectors provided.
        (
            "none",
            [],
            {},
            None,
            [],
            {},
            None,
        ),
    ],
)
def test_enrich_function_preemption_spec(
    mode,
    tolerations,
    node_selector,
    affinity,
    expected_tolerations,
    expected_node_selector,
    expected_affinity,
):
    function = mlrun.new_function("some-function", kind="job")

    # Set the preemption mode.
    function.spec.preemption_mode = mode

    # Set initial values to simulate user-provided configuration.
    function.with_node_selection(
        node_selector=node_selector,
        tolerations=tolerations,
        affinity=affinity,
    )

    # Simulated mlconf preemptible settings
    mock_preemptible_config()

    function.spec.enrich_function_preemption_spec()

    assert (
        [t.to_dict() for t in function.spec.tolerations]
        if function.spec.tolerations
        else [] == expected_tolerations
    )
    assert function.spec.node_selector == expected_node_selector
    assert function.spec.affinity == expected_affinity


@pytest.mark.parametrize(
    "node_selector, tolerations, affinity, expected_warning_substrings",
    [
        # Only node_selector matches the preemptible configuration.
        (
            {"app.iguazio.com/lifecycle": "preemptible", "other": "value"},
            None,
            None,
            ["Node selectors: 'app.iguazio.com/lifecycle': 'preemptible'"],
        ),
        # Only tolerations match the preemptible configuration.
        (
            None,
            [
                k8s_client.V1Toleration(
                    key="cloud.google.com/gke-spot", value="true", effect="NoSchedule"
                )
            ],
            None,
            ["Tolerations: 'cloud.google.com/gke-spot'='true' (effect: 'NoSchedule')"],
        ),
        # Only affinity matches the preemptible configuration.
        (
            None,
            None,
            create_node_affinity_with_terms(preemptible_affinity_iguazio),
            ["Affinity: 'app.iguazio.com/lifecycle  In  ['preemptible']'"],
        ),
        # All three match.
        (
            {"app.iguazio.com/lifecycle": "preemptible", "other": "value"},
            [
                k8s_client.V1Toleration(
                    key="cloud.google.com/gke-spot", value="true", effect="NoSchedule"
                ),
                k8s_client.V1Toleration(key="custom", value="yes", effect="NoSchedule"),
            ],
            create_node_affinity_with_terms(preemptible_affinity_iguazio),
            [
                "Node selectors: 'app.iguazio.com/lifecycle': 'preemptible'",
                "Tolerations: 'cloud.google.com/gke-spot'='true' (effect: 'NoSchedule')",
                "Affinity: 'app.iguazio.com/lifecycle  In  ['preemptible']'",
            ],
        ),
        # No matching values.
        (
            {"custom": "value"},
            [k8s_client.V1Toleration(key="custom", value="yes", effect="NoSchedule")],
            create_node_affinity_with_terms(
                [
                    [
                        k8s_client.V1NodeSelectorRequirement(
                            key="custom-key", operator="In", values=["non-match"]
                        )
                    ]
                ],
            ),
            [],
        ),
    ],
)
def test_with_node_selection_warnings(
    node_selector,
    tolerations,
    affinity,
    expected_warning_substrings,
):
    """
    This test verifies that mlrun.Function.with_node_selection logs the expected warnings when
    user-provided configuration (node_selector, tolerations, affinity) matches the preemptible settings.
    """
    mock_preemptible_config()

    function = mlrun.new_function("test-func", kind="job")

    # Capture warnings raised during with_node_selection.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function.with_node_selection(
            node_selector=node_selector,
            tolerations=tolerations,
            affinity=affinity,
        )

    warning_messages = [str(w.message) for w in caught]

    # Assert that each expected warning substring is found in the warnings.
    for expected in expected_warning_substrings:
        assert any(
            expected in message for message in warning_messages
        ), f"Expected warning substring '{expected}' not found in warnings: {warning_messages}"
    # If no warnings are expected, assert that none were raised.
    if not expected_warning_substrings:
        assert len(warning_messages) == 0, (
            f"Expected no warnings, but found: {warning_messages}"
            "Expected no warnings, but found: {warning_messages}"
        )
