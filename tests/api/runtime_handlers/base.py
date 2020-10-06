import unittest.mock
from typing import List, Dict

from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import get_runtime_handler
from mlrun.utils import create_logger
from tests.conftest import DictToK8sObjectWrapper

logger = create_logger(level="debug", name="test-runtime-handlers")


class TestRuntimeHandlerBase:
    def setup_method(self, method):
        self._logger = logger
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    def teardown_method(self, method):
        self._logger.info(
            f"Tearing down test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_teardown()

        self._logger.info(
            f"Finished tearing down test {self.__class__.__name__}::{method.__name__}"
        )

    def custom_setup(self):
        pass

    def custom_teardown(self):
        pass

    @staticmethod
    def _assert_runtime_handler_list_resources(
        runtime_kind, expected_crds=None, expected_pods=None, expected_services=None,
    ):
        runtime_handler = get_runtime_handler(runtime_kind)
        resources = runtime_handler.list_resources()
        crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
        get_k8s().list_pods.assert_called_once_with(
            get_k8s().resolve_namespace(),
            selector=runtime_handler._get_default_label_selector(),
        )
        if expected_crds:
            get_k8s().crdapi.list_namespaced_custom_object.assert_called_once_with(
                crd_group,
                crd_version,
                get_k8s().resolve_namespace(),
                crd_plural,
                label_selector=runtime_handler._get_default_label_selector(),
            )
        if expected_services:
            get_k8s().v1api.list_namespaced_service.assert_called_once_with(
                get_k8s().resolve_namespace(),
                label_selector=runtime_handler._get_default_label_selector(),
            )
        TestRuntimeHandlerBase._assert_list_resources_response(
            resources,
            expected_crds=expected_crds,
            expected_pods=expected_pods,
            expected_services=expected_services,
        )

    @staticmethod
    def _assert_list_resources_response(
        resources, expected_crds=None, expected_pods=None, expected_services=None
    ):
        expected_crds = expected_crds or []
        expected_pods = expected_pods or []
        expected_services = expected_services or []
        assert len(resources["crd_resources"]) == len(expected_crds)
        for index, crd in enumerate(expected_crds):
            assert resources["crd_resources"][index]["name"] == crd["metadata"]["name"]
            assert (
                resources["crd_resources"][index]["labels"] == crd["metadata"]["labels"]
            )
            assert resources["crd_resources"][index]["status"] == crd["status"]
        assert len(resources["pod_resources"]) == len(expected_pods)
        for index, pod in enumerate(expected_pods):
            pod_dict = pod.to_dict()
            assert (
                resources["pod_resources"][index]["name"]
                == pod_dict["metadata"]["name"]
            )
            assert (
                resources["pod_resources"][index]["labels"]
                == pod_dict["metadata"]["labels"]
            )
            assert resources["pod_resources"][index]["status"] == pod_dict["status"]
        if expected_services:
            assert len(resources["service_resources"]) == len(expected_services)
            for index, service in enumerate(expected_services):
                assert (
                    resources["service_resources"][index]["name"]
                    == service.metadata.name
                )
                assert (
                    resources["service_resources"][index]["labels"]
                    == service.metadata.labels
                )

    @staticmethod
    def _mock_list_namespaces_pods(pod_dicts_call_responses: List[List[Dict]]):
        calls = []
        for pod_dicts_call_response in pod_dicts_call_responses:
            pods = []
            for pod_dict in pod_dicts_call_response:
                pod = DictToK8sObjectWrapper(pod_dict)
                pods.append(pod)
            calls.append(DictToK8sObjectWrapper({
                'items': pods,
            }))
        get_k8s().v1api.list_namespaced_pod = unittest.mock.Mock(side_effect=calls)
        return calls

    @staticmethod
    def _mock_list_crds(crd_dicts):
        crds = {
            "items": crd_dicts,
        }
        get_k8s().crdapi.list_namespaced_custom_object = unittest.mock.Mock(
            return_value=crds
        )
        return crd_dicts

    @staticmethod
    def _mock_list_services(service_dicts):
        service_mocks = []
        for service_dict in service_dicts:
            service_mock = unittest.mock.Mock()
            service_mock.metadata.name.return_value = service_dict["metadata"]["name"]
            service_mock.metadata.labels.return_value = service_dict["metadata"][
                "labels"
            ]
            service_mocks.append(service_mock)
        services_mock = unittest.mock.Mock()
        services_mock.items = service_mocks
        get_k8s().v1api.list_namespaced_service = unittest.mock.Mock(
            return_value=services_mock
        )
        return service_mocks
