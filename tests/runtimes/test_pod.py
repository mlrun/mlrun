import copy
import inspect

import pytest

import mlrun
import mlrun.runtimes.mpijob.abstract
import mlrun.runtimes.mpijob.v1
import mlrun.runtimes.pod
from mlrun.config import config


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
    checked_classes = set()
    invalid_classes = {}
    for base_class, inheriting_classes in classes_map.items():
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


def test_enrich_resources_with_default_pod_resources():
    function_resources = {}
    output = mlrun.runtimes.pod.enrich_resources_with_default_pod_resources(
        function_resources
    )
    assert output == config.default_function_pod_resources.to_dict()

    mlrun.mlconf.default_function_pod_resources = {
        "requests": {"cpu": "25m", "memory": "1M", "gpu": ""},
        "limits": {"cpu": "2", "memory": "1G", "gpu": ""},
    }
    function_resources = {"requests": {"cpu": "1"}}
    output = mlrun.runtimes.pod.enrich_resources_with_default_pod_resources(
        function_resources
    )

    assert output == mlrun.mlconf.default_function_pod_resources.to_dict()[
        "requests"
    ].update({"cpu": "1"})


def test_resource_enrichment_in_resource_spec_initialization():
    resources = {
        "requests": {"cpu": "25m", "memory": "1M", "gpu": ""},
        "limits": {"cpu": "2", "memory": "1G", "gpu": ""},
    }
    mlrun.mlconf.default_function_pod_resources = copy.copy(resources)

    spec = mlrun.runtimes.pod.KubeResourceSpec()
    assert spec.resources == mlrun.mlconf.default_function_pod_resources.to_dict()

    spec_requests = {"cpu": "1"}
    expected_resources = copy.copy(resources)
    expected_resources["requests"].update(spec_requests)
    spec = mlrun.runtimes.pod.KubeResourceSpec(resources={"requests": spec_requests})
    assert spec.resources == expected_resources

    spec_requests = {"cpu": "1", "memory": "500M"}
    spec_limits = {"memory": "2G"}
    expected_resources["requests"].update(spec_requests)
    expected_resources["limits"].update(spec_limits)
    spec = mlrun.runtimes.pod.KubeResourceSpec(
        resources={"requests": spec_requests, "limits": spec_limits}
    )
    assert spec.resources == expected_resources
