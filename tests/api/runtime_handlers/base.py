import unittest.mock
from datetime import datetime, timezone
from typing import List, Dict

import pytest
from kubernetes import client
from sqlalchemy.orm import Session

import mlrun.api.crud as crud
from mlrun.api.constants import LogSources
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import get_runtime_handler
from mlrun.runtimes.constants import RunStates, PodPhases
from mlrun.utils import create_logger, now_date

logger = create_logger(level="debug", name="test-runtime-handlers")


class TestRuntimeHandlerBase:
    def setup_method(self, method):
        self._logger = logger
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )

        self.project = "test_project"
        self.run_uid = "test_run_uid"

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    @pytest.fixture(autouse=True)
    def _store_run_fixture(self, db: Session):
        self.run = {
            "status": {
                "state": RunStates.created,
                "last_update": now_date().isoformat(),
            },
            "metadata": {"project": self.project, "uid": self.run_uid},
        }
        get_db().store_run(db, self.run, self.run_uid, self.project)

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
    def _generate_pod(name, labels, phase=PodPhases.succeeded):
        terminated_container_state = client.V1ContainerStateTerminated(
            finished_at=datetime.now(timezone.utc), exit_code=0
        )
        container_state = client.V1ContainerState(terminated=terminated_container_state)
        container_status = client.V1ContainerStatus(
            state=container_state,
            image="must/provide:image",
            image_id="must-provide-image-id",
            name="must-provide-name",
            ready=True,
            restart_count=0,
        )
        status = client.V1PodStatus(phase=phase, container_statuses=[container_status])
        metadata = client.V1ObjectMeta(
            name=name, labels=labels, namespace=get_k8s().resolve_namespace()
        )
        pod = client.V1Pod(metadata=metadata, status=status)
        return pod

    @staticmethod
    def _assert_runtime_handler_list_resources(
        runtime_kind, expected_crds=None, expected_pods=None, expected_services=None,
    ):
        runtime_handler = get_runtime_handler(runtime_kind)
        resources = runtime_handler.list_resources()
        crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
        get_k8s().v1api.list_namespaced_pod.assert_called_once_with(
            get_k8s().resolve_namespace(),
            label_selector=runtime_handler._get_default_label_selector(),
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
    def _mock_list_namespaced_pods(list_pods_call_responses: List[List[client.V1Pod]]):
        calls = []
        for list_pods_call_response in list_pods_call_responses:
            pods = client.V1PodList(items=list_pods_call_response)
            calls.append(pods)
        get_k8s().v1api.list_namespaced_pod = unittest.mock.Mock(side_effect=calls)
        return calls

    @staticmethod
    def _assert_delete_namespaced_pods(
        expected_pod_names: List[str], expected_pod_namespace: str = None
    ):
        calls = [
            unittest.mock.call(expected_pod_name, expected_pod_namespace)
            for expected_pod_name in expected_pod_names
        ]
        if not expected_pod_names:
            assert get_k8s().v1api.delete_namespaced_pod.call_count == 0
        else:
            get_k8s().v1api.delete_namespaced_pod.assert_has_calls(calls)

    @staticmethod
    def _assert_delete_namespaced_services(
        expected_service_names: List[str], expected_service_namespace: str = None
    ):
        calls = [
            unittest.mock.call(expected_service_name, expected_service_namespace)
            for expected_service_name in expected_service_names
        ]
        if not expected_service_names:
            assert get_k8s().v1api.delete_namespaced_service.call_count == 0
        else:
            get_k8s().v1api.delete_namespaced_service.assert_has_calls(calls)

    @staticmethod
    def _assert_delete_namespaced_custom_objects(
        runtime_handler,
        expected_custom_object_names: List[str],
        expected_custom_object_namespace: str = None,
    ):
        crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
        calls = [
            unittest.mock.call(
                crd_group,
                crd_version,
                expected_custom_object_namespace,
                crd_plural,
                expected_custom_object_name,
                client.V1DeleteOptions(),
            )
            for expected_custom_object_name in expected_custom_object_names
        ]
        if not expected_custom_object_names:
            assert get_k8s().crdapi.delete_namespaced_custom_object.call_count == 0
        else:
            get_k8s().crdapi.delete_namespaced_custom_object.assert_has_calls(calls)

    @staticmethod
    def _mock_delete_namespaced_pods():
        get_k8s().v1api.delete_namespaced_pod = unittest.mock.Mock()

    @staticmethod
    def _mock_delete_namespaced_custom_objects():
        get_k8s().crdapi.delete_namespaced_custom_object = unittest.mock.Mock()

    @staticmethod
    def _mock_delete_namespaced_services():
        get_k8s().v1api.delete_namespaced_service = unittest.mock.Mock()

    @staticmethod
    def _mock_read_namespaced_pod_log():
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)
        return log

    @staticmethod
    def _mock_list_namespaced_crds(crd_dicts_call_responses: List[List[Dict]]):
        calls = []
        for crd_dicts_call_response in crd_dicts_call_responses:
            calls.append({"items": crd_dicts_call_response})
        get_k8s().crdapi.list_namespaced_custom_object = unittest.mock.Mock(
            side_effect=calls
        )
        return calls

    @staticmethod
    def _mock_list_services(services):
        services_list = client.V1ServiceList(items=services)
        get_k8s().v1api.list_namespaced_service = unittest.mock.Mock(
            return_value=services_list
        )
        return services

    @staticmethod
    def _assert_list_namespaced_pods_calls(
        runtime_handler,
        expected_number_of_calls: int,
        expected_label_selector: str = None,
    ):
        assert (
            get_k8s().v1api.list_namespaced_pod.call_count == expected_number_of_calls
        )
        expected_label_selector = (
            expected_label_selector or runtime_handler._get_default_label_selector()
        )
        get_k8s().v1api.list_namespaced_pod.assert_any_call(
            get_k8s().resolve_namespace(), label_selector=expected_label_selector
        )

    @staticmethod
    def _assert_list_namespaced_crds_calls(
        runtime_handler, expected_number_of_calls: int
    ):
        crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
        assert (
            get_k8s().crdapi.list_namespaced_custom_object.call_count
            == expected_number_of_calls
        )
        get_k8s().crdapi.list_namespaced_custom_object.assert_any_call(
            crd_group,
            crd_version,
            get_k8s().resolve_namespace(),
            crd_plural,
            label_selector=runtime_handler._get_default_label_selector(),
        )

    @staticmethod
    def _assert_run_logs(
        db: Session,
        project: str,
        uid: str,
        expected_log: str,
        logger_pod_name: str = None,
    ):
        if logger_pod_name is not None:
            get_k8s().v1api.read_namespaced_pod_log.assert_called_once_with(
                name=logger_pod_name, namespace=get_k8s().resolve_namespace(),
            )
        _, log = crud.Logs.get_logs(db, project, uid, source=LogSources.PERSISTENCY)
        assert log == expected_log.encode()

    @staticmethod
    def _assert_run_reached_state(
        db: Session, project: str, uid: str, expected_state: str
    ):
        run = get_db().read_run(db, uid, project)
        assert run["status"]["state"] == expected_state
