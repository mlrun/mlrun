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

    # def test_get_sanitized_attribute(self):
    #     for test_case in [
    #         {
    #             "attribute_name":"affinity",
    #             "expected_spec_attribute": {
    #
    #             }
    #         }
    #     ]
    #     assert (
    #             DeepDiff(
    #                 parsed_function_object.spec._get_sanitized_attribute("affinity"),
    #                 submit_job_body["function"]["spec"]["affinity"],
    #                 ignore_order=True,
    #             )
    #             == {}
    #     )
