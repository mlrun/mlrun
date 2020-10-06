import unittest.mock

from mlrun import get_run_db
from mlrun.runtimes import get_runtime_handler
from mlrun.utils import create_logger

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
        runtime_kind,
        k8s_helper_mock,
        expected_crds=None,
        expected_pods=None,
        expected_services=None,
    ):
        runtime_handler = get_runtime_handler(runtime_kind)
        resources = runtime_handler.list_resources()
        crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
        k8s_helper_mock.list_pods.assert_called_once_with(
            k8s_helper_mock.resolve_namespace(),
            selector=runtime_handler._get_default_label_selector(),
        )
        if expected_crds:
            k8s_helper_mock.crdapi.list_namespaced_custom_object.assert_called_once_with(
                crd_group,
                crd_version,
                k8s_helper_mock.resolve_namespace(),
                crd_plural,
                label_selector=runtime_handler._get_default_label_selector(),
            )
        if expected_services:
            k8s_helper_mock.v1api.list_namespaced_service.assert_called_once_with(
                k8s_helper_mock.resolve_namespace(),
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
    def _mock_list_crds(k8s_helper_mock, crd_dicts):
        crds = {
            "items": crd_dicts,
        }
        k8s_helper_mock.crdapi.list_namespaced_custom_object.return_value = crds
        return crd_dicts

    @staticmethod
    def _mock_list_services(k8s_helper_mock, service_dicts):
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
        k8s_helper_mock.v1api.list_namespaced_service.return_value = services_mock
        return service_mocks
