import deepdiff
import pytest
from kubernetes import client

import mlrun
import mlrun.errors
import mlrun.runtimes.pod
from tests.api.runtimes.base import TestRuntimeBase


class TestKubeResource(TestRuntimeBase):
    def test_attribute_serializations(self):
        def set_with_node_selection(resource, attr_name, attr):
            if attr_name == "tolerations":
                resource.with_node_selection(tolerations=attr)
            if attr_name == "affinity":
                resource.with_node_selection(affinity=attr)

        api = client.ApiClient()
        for test_case in [
            {
                "attribute_name": "affinity",
                "attribute": self._generate_affinity(),
            },
            {
                "attribute_name": "tolerations",
                "attribute": self._generate_tolerations(),
            },
        ]:
            attribute_name = test_case.get("attribute_name")
            attribute = test_case.get("attribute")
            kube_resource = mlrun.runtimes.pod.KubeResource()

            # simulating the client - setting from a class instance
            set_with_node_selection(kube_resource, attribute_name, attribute)
            # simulating sending to API - serialization through dict
            kube_resource_dict = kube_resource.to_dict()
            kube_resource = kube_resource.from_dict(kube_resource_dict)

            assert (
                deepdiff.DeepDiff(
                    kube_resource.spec.to_dict()[attribute_name],
                    api.sanitize_for_serialization(attribute),
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
