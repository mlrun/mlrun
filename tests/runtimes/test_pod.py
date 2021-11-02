import inspect
import typing
import pytest

import mlrun
import mlrun.runtimes.pod
import mlrun.runtimes.mpijob.abstract
import mlrun.runtimes.mpijob.v1


def test_runtimes_inheritance():
    classes_map = {
        mlrun.runtimes.pod.KubeResourceSpec: [
            mlrun.runtimes.daskjob.DaskSpec,
            mlrun.runtimes.function.NuclioSpec,
            mlrun.runtimes.serving.ServingSpec,
            mlrun.runtimes.mpijob.abstract.MPIResourceSpec,
            mlrun.runtimes.mpijob.v1.MPIV1ResourceSpec,
            mlrun.runtimes.remotesparkjob.RemoteSparkSpec,
            mlrun.runtimes.sparkjob.spark2job.Spark2JobSpec,
            mlrun.runtimes.sparkjob.spark3job.Spark3JobSpec,
        ]
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
                class_kwargs = list(inspect.signature(class_.__init__).parameters.keys())
                base_class_kwargs = list(inspect.signature(base_class.__init__).parameters.keys())
                if not set(base_class_kwargs).issubset(class_kwargs):
                    invalid_classes[inheriting_class] = list(set(base_class_kwargs) - set(base_class_kwargs).intersection(class_kwargs))
                checked_classes.add(inheriting_class)
    if invalid_classes:
        pytest.fail(f"Found classes that are not accepting all of their parent classes kwargs: {invalid_classes}")
