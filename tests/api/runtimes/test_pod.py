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
import deepdiff
import kubernetes
import pytest

import mlrun
import mlrun.errors
import mlrun.runtimes.pod
from tests.api.runtimes.base import TestRuntimeBase


class TestKubeResource(TestRuntimeBase):
    api = kubernetes.client.ApiClient()

    def test_attribute_serializations(self):
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
            self._set_with_node_selection(kube_resource, attribute_name, attribute)
            # simulating sending to API - serialization through dict
            kube_resource_dict = kube_resource.to_dict()
            kube_resource = kube_resource.from_dict(kube_resource_dict)
            assert (
                deepdiff.DeepDiff(
                    kube_resource.spec.to_dict()[attribute_name],
                    self.api.sanitize_for_serialization(attribute),
                    ignore_order=True,
                )
                == {}
            )

    def test_transform_attribute_to_k8s_class_instance(self):
        for test_case in [
            {
                "attribute_name": "tolerations",
                "expect_failure": True,
                "attribute": [
                    {
                        "key": "test1",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                        "toleration_seconds": 3600,
                    },
                ],
            },
            {
                "attribute_name": "tolerations",
                "expect_failure": False,
                "attribute": {
                    "key": "test1",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                    "tolerationSeconds": 3600,
                },
                "expected_attribute": self._generate_tolerations(),
            },
            {
                "attribute_name": "tolerations",
                "expect_failure": False,
                "attribute": self._generate_toleration(),
                "expected_attribute": self._generate_tolerations(),
            },
            {
                "attribute_name": "tolerations",
                "expect_failure": False,
                "attribute": self._generate_tolerations(),
                "expected_attribute": self._generate_tolerations(),
            },
            {
                "attribute_name": "affinity",
                "expect_failure": False,
                "attribute": self._generate_affinity(),
                "expected_attribute": self._generate_affinity(),
            },
            {
                "attribute_name": "affinity",
                "expect_failure": True,
                "attribute": self._generate_affinity().to_dict(),
            },
            {
                "attribute_name": "affinity",
                "expect_failure": False,
                "attribute": self.api.sanitize_for_serialization(
                    self._generate_affinity()
                ),
                "expected_attribute": self._generate_affinity(),
            },
            {
                "attribute_name": "affinity",
                "expect_failure": False,
                "attribute": None,
                "expected_attribute": None,
            },
        ]:
            attribute_name = test_case.get("attribute_name")
            attribute = test_case.get("attribute")
            expect_failure = test_case.get("expect_failure", False)
            expected_attribute = test_case.get("expected_attribute")

            kube_resource = mlrun.runtimes.pod.KubeResource()
            if expect_failure:
                with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError):
                    self._set_with_node_selection(
                        kube_resource, attribute_name, attribute
                    )
            else:
                self._set_with_node_selection(kube_resource, attribute_name, attribute)
                assert (
                    deepdiff.DeepDiff(
                        getattr(kube_resource.spec, attribute_name),
                        expected_attribute,
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

    @staticmethod
    def _set_with_node_selection(
        resource: mlrun.runtimes.pod.KubeResource, attr_name: str, attr
    ):
        if attr_name == "tolerations":
            resource.with_node_selection(tolerations=attr)
        if attr_name == "affinity":
            resource.with_node_selection(affinity=attr)
