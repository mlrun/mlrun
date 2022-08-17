import inspect

import kubernetes.client
import pytest
from deepdiff import DeepDiff

import mlrun
import mlrun.runtimes.mpijob.abstract
import mlrun.runtimes.mpijob.v1
import mlrun.runtimes.pod


def test_runtimes_inheritance():
    classes_map = {
        mlrun.runtimes.base.FunctionSpec: [
            mlrun.runtimes.daskjob.DaskSpec,
            mlrun.runtimes.function.NuclioSpec,
            mlrun.runtimes.serving.ServingSpec,
            mlrun.runtimes.mpijob.abstract.MPIResourceSpec,
            mlrun.runtimes.mpijob.v1.MPIV1ResourceSpec,
            mlrun.runtimes.remotesparkjob.RemoteSparkSpec,
            mlrun.runtimes.sparkjob.spark2job.Spark2JobSpec,
            mlrun.runtimes.sparkjob.spark3job.Spark3JobSpec,
        ],
        mlrun.runtimes.pod.KubeResourceSpec: [
            mlrun.runtimes.daskjob.DaskSpec,
            mlrun.runtimes.function.NuclioSpec,
            mlrun.runtimes.serving.ServingSpec,
            mlrun.runtimes.mpijob.abstract.MPIResourceSpec,
            mlrun.runtimes.mpijob.v1.MPIV1ResourceSpec,
            mlrun.runtimes.remotesparkjob.RemoteSparkSpec,
            mlrun.runtimes.sparkjob.abstract.AbstractSparkJobSpec,
            mlrun.runtimes.sparkjob.spark2job.Spark2JobSpec,
            mlrun.runtimes.sparkjob.spark3job.Spark3JobSpec,
        ],
        mlrun.runtimes.function.NuclioSpec: [
            mlrun.runtimes.serving.ServingSpec,
        ],
        mlrun.runtimes.base.FunctionStatus: [
            mlrun.runtimes.daskjob.DaskStatus,
            mlrun.runtimes.function.NuclioStatus,
        ],
        mlrun.runtimes.base.BaseRuntime: [
            mlrun.runtimes.local.HandlerRuntime,
            mlrun.runtimes.local.BaseRuntime,
            mlrun.runtimes.function.RemoteRuntime,
            mlrun.runtimes.serving.ServingRuntime,
            mlrun.runtimes.kubejob.KubejobRuntime,
            mlrun.runtimes.daskjob.DaskCluster,
            mlrun.runtimes.mpijob.v1.MpiRuntimeV1,
            mlrun.runtimes.mpijob.v1alpha1.MpiRuntimeV1Alpha1,
            mlrun.runtimes.remotesparkjob.RemoteSparkRuntime,
            mlrun.runtimes.sparkjob.spark2job.Spark2Runtime,
            mlrun.runtimes.sparkjob.spark3job.Spark3Runtime,
        ],
    }
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
                    inspect.signature(class_.__init__).parameters.keys()
                )
                base_class_kwargs = list(
                    inspect.signature(base_class.__init__).parameters.keys()
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


def test_volume_mounts_addition():
    volume_mount = kubernetes.client.V1VolumeMount(
        mount_path="some-path", name="volume-name"
    )
    dict_volume_mount = volume_mount.to_dict()
    sanitized_dict_volume_mount = (
        kubernetes.client.ApiClient().sanitize_for_serialization(volume_mount)
    )
    function = mlrun.new_function(kind=mlrun.runtimes.RuntimeKinds.job)
    function.spec.volume_mounts = [
        volume_mount,
        dict_volume_mount,
        sanitized_dict_volume_mount,
    ]
    assert len(function.spec.volume_mounts) == 1
