import deepdiff
import pytest

import mlrun
import mlrun.errors
import mlrun.runtimes.pod
from tests.api.runtimes.base import TestRuntimeBase


class TestKubeResource(TestRuntimeBase):
    def test_affinity_serialization(self):
        affinity = self._generate_affinity()
        kube_resource = mlrun.runtimes.pod.KubeResource()
        # simulating the client - setting from a class instance
        kube_resource.with_node_selection(affinity=affinity)
        # simulating sending to API - serialization through dict
        kube_resource_dict = kube_resource.to_dict()
        kube_resource = kube_resource.from_dict(kube_resource_dict)
        assert (
            deepdiff.DeepDiff(
                kube_resource.spec.affinity.to_dict(),
                affinity.to_dict(),
                ignore_order=True,
            )
            == {}
        )

    def test_with_limits_regex_validation(self):
        cases = [
            {"cpu": "no-number", "expected_failure": True},
            {"memory": "1g", "expected_failure": True},  # common mistake
            {"gpus": "1GPU", "expected_failure": True},
            {"cpu": "12e6"},
            {"memory": "12Mi"},
            {"memory": "12M"},
        ]
        for case in cases:
            kube_resource = mlrun.runtimes.pod.KubeResource()
            if case.get("expected_failure"):
                with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                    kube_resource.with_limits(
                        case.get("memory"), case.get("cpu"), case.get("gpus")
                    )
                if not case.get("gpus"):
                    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                        kube_resource.with_requests(case.get("memory"), case.get("cpu"))
            else:
                kube_resource.with_limits(
                    case.get("memory"), case.get("cpu"), case.get("gpus")
                )
                if not case.get("gpus"):
                    kube_resource.with_requests(case.get("memory"), case.get("cpu"))

    def test_function_with_handler_function(self):
        """
        Some users write their code in a notebook, then do code_to_function, and then mistakenly give the function
        itself to the handler kwarg, instead of giving the function name (string)
        This test is here to verify that the failure is fast and clear
        """
        def some_func():
            pass
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match=r"Parameter default_handler must be a string. got <class 'function'>"):
            mlrun.code_to_function("some-name",
                                   filename=str(self.assets_path / "dummy_function.py"),
                                   kind=mlrun.runtimes.RuntimeKinds.job,
                                   handler=some_func)
