# Copyright 2023 Iguazio
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
import unittest.mock
import uuid
from datetime import datetime, timezone
from typing import Optional

import deepdiff
import fastapi.testclient
import pytest
from kubernetes import client
from sqlalchemy.orm import Session

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import server.api.crud
import server.api.utils.clients.chief
from mlrun.common.runtimes.constants import PodPhases, RunStates
from mlrun.utils import create_test_logger, now_date
from server.api.runtime_handlers import get_runtime_handler
from server.api.utils.singletons.db import get_db
from server.api.utils.singletons.k8s import get_k8s_helper

logger = create_test_logger(name="test-runtime-handlers")


class TestRuntimeHandlerBase:
    def setup_method(self, method):
        self._logger = logger
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )

        self.project = "test-project"
        self.run_uid = "test_run_uid"
        self.kind = "job"

        mlrun.mlconf.mpijob_crd_version = (
            mlrun.common.runtimes.constants.MPIJobCRDVersions.v1
        )
        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    @pytest.fixture(autouse=True)
    def _store_run_fixture(self, db: Session):
        self._store_run(db)

    def _store_run(
        self,
        db: Session,
        name: str = None,
        uid: str = None,
        start_time: datetime = None,
    ):
        self.run = {
            "status": {
                "state": RunStates.created,
                "last_update": now_date().isoformat(),
            },
            "metadata": {
                "project": self.project,
                "name": name or "some-run-name",
                "uid": uid or self.run_uid,
                "labels": {
                    "kind": self.kind,
                },
            },
            "spec": {
                "state_thresholds": mlrun.mlconf.function.spec.state_thresholds.default.to_dict(),
                "node_selector": {"test/host": "node1"},
            },
        }
        if start_time:
            self.run["status"]["start_time"] = start_time.isoformat()
        server.api.crud.Runs().store_run(
            db, self.run, self.run["metadata"]["uid"], project=self.project
        )

    @pytest.fixture(autouse=True)
    def setup_method_fixture(self, db: Session, client: fastapi.testclient.TestClient):
        # We want this mock for every test, ideally we would have simply put it in the setup_method
        # but it is happening before the fixtures initialization. We need the client fixture (which needs the db one)
        # in order to be able to mock k8s stuff
        get_k8s_helper().v1api = unittest.mock.Mock()
        get_k8s_helper().crdapi = unittest.mock.Mock()
        get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
            return_value=True
        )
        # enable inheriting classes to do the same
        self.custom_setup_after_fixtures()

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

    def custom_setup_after_fixtures(self):
        pass

    def custom_teardown(self):
        pass

    @staticmethod
    def _generate_pod(name, labels, phase=PodPhases.succeeded):
        terminated_container_state = client.V1ContainerStateTerminated(
            finished_at=datetime.now(timezone.utc),
            exit_code=0,
            reason="Some reason",
            message="Failed message",
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
        status = client.V1PodStatus(
            phase=phase,
            container_statuses=[container_status],
            start_time=datetime.now(timezone.utc),
        )
        metadata = client.V1ObjectMeta(
            name=name, labels=labels, namespace=get_k8s_helper().resolve_namespace()
        )
        pod = client.V1Pod(metadata=metadata, status=status)
        return pod

    @staticmethod
    def _generate_job_labels(run_name, uid=None, job_labels=None):
        labels = job_labels.copy() if job_labels else {}
        labels[mlrun_constants.MLRunInternalLabels.uid] = uid or str(uuid.uuid4())
        labels[mlrun_constants.MLRunInternalLabels.name] = run_name
        return labels

    @staticmethod
    def _generate_run_pod_label_selector(run_uid):
        return f"{mlrun_constants.MLRunInternalLabels.uid}={run_uid}"

    @staticmethod
    def _generate_config_map(name, labels, data=None):
        metadata = client.V1ObjectMeta(
            name=name, labels=labels, namespace=get_k8s_helper().resolve_namespace()
        )
        if data is None:
            data = {"key": "value"}
        return client.V1ConfigMap(metadata=metadata, data=data)

    def _assert_runtime_handler_list_resources(
        self,
        runtime_kind,
        expected_crds=None,
        expected_pods=None,
        expected_services=None,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ):
        runtime_handler = get_runtime_handler(runtime_kind)
        if group_by is None:
            project = "*"
            label_selector = runtime_handler._get_default_label_selector()
            assertion_func = TestRuntimeHandlerBase._assert_list_resources_response
        elif group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.job:
            project = self.project
            label_selector = ",".join(
                [
                    runtime_handler._get_default_label_selector(),
                    f"{mlrun_constants.MLRunInternalLabels.project}={self.project}",
                ]
            )
            assertion_func = (
                TestRuntimeHandlerBase._assert_list_resources_grouped_by_job_response
            )
        elif group_by == mlrun.common.schemas.ListRuntimeResourcesGroupByField.project:
            project = self.project
            label_selector = ",".join(
                [
                    runtime_handler._get_default_label_selector(),
                    f"{mlrun_constants.MLRunInternalLabels.project}={self.project}",
                ]
            )
            assertion_func = TestRuntimeHandlerBase._assert_list_resources_grouped_by_project_response
        else:
            raise NotImplementedError("Unsupported group by value")
        resources = runtime_handler.list_resources(project, group_by=group_by)
        crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
        get_k8s_helper().v1api.list_namespaced_pod.assert_called_once_with(
            get_k8s_helper().resolve_namespace(),
            label_selector=label_selector,
        )
        if expected_crds:
            get_k8s_helper().crdapi.list_namespaced_custom_object.assert_called_once_with(
                crd_group,
                crd_version,
                get_k8s_helper().resolve_namespace(),
                crd_plural,
                label_selector=label_selector,
            )
        if expected_services:
            get_k8s_helper().v1api.list_namespaced_service.assert_called_once_with(
                get_k8s_helper().resolve_namespace(),
                label_selector=label_selector,
            )
        assertion_func(
            self,
            resources,
            expected_crds=expected_crds,
            expected_pods=expected_pods,
            expected_services=expected_services,
            runtime_handler=runtime_handler,
        )

        return resources

    def _assert_list_resources_grouped_by_job_response(
        self,
        resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        expected_crds=None,
        expected_pods=None,
        expected_services=None,
        **kwargs,
    ):
        self._assert_list_resources_grouped_by_response(
            resources,
            lambda labels: (
                labels[mlrun_constants.MLRunInternalLabels.project],
                labels[mlrun_constants.MLRunInternalLabels.uid],
            ),
            expected_crds,
            expected_pods,
            expected_services,
        )

    def _assert_list_resources_grouped_by_project_response(
        self,
        resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        expected_crds=None,
        expected_pods=None,
        expected_services=None,
        runtime_handler=None,
    ):
        def _extract_project_and_kind_from_runtime_resources_labels(
            labels: dict,
        ) -> tuple[str, str]:
            project = labels.get(mlrun_constants.MLRunInternalLabels.project, "")
            class_ = labels[mlrun_constants.MLRunInternalLabels.mlrun_class]
            kind = runtime_handler._resolve_kind_from_class(class_)
            return project, kind

        self._assert_list_resources_grouped_by_response(
            resources,
            _extract_project_and_kind_from_runtime_resources_labels,
            expected_crds,
            expected_pods,
            expected_services,
        )

    def _assert_list_resources_grouped_by_response(
        self,
        resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        group_by_field_extractor,
        expected_crds=None,
        expected_pods=None,
        expected_services=None,
    ):
        expected_crds = expected_crds or []
        expected_pods = expected_pods or []
        expected_services = expected_services or []
        for index, crd in enumerate(expected_crds):
            self._assert_resource_in_response_resources(
                "crd", crd, resources, "crd_resources", group_by_field_extractor
            )
        for index, pod in enumerate(expected_pods):
            pod_dict = pod.to_dict()
            self._assert_resource_in_response_resources(
                "pod", pod_dict, resources, "pod_resources", group_by_field_extractor
            )
        for index, service in enumerate(expected_services):
            service_dict = service.to_dict()
            self._assert_resource_in_response_resources(
                "service",
                service_dict,
                resources,
                "service_resources",
                group_by_field_extractor,
            )

    @staticmethod
    def _assert_resource_in_response_resources(
        expected_resource_type: str,
        expected_resource: dict,
        resources: mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        resources_field_name: str,
        group_by_field_extractor,
    ):
        (
            first_group_by_field_value,
            second_group_by_field_value,
        ) = group_by_field_extractor(expected_resource["metadata"]["labels"])
        found = False
        for resource in getattr(
            resources[first_group_by_field_value][second_group_by_field_value],
            resources_field_name,
        ):
            if resource.name == expected_resource["metadata"]["name"]:
                found = True
                assert (
                    deepdiff.DeepDiff(
                        resource.labels,
                        expected_resource["metadata"]["labels"],
                        ignore_order=True,
                    )
                    == {}
                )
                assert (
                    deepdiff.DeepDiff(
                        resource.status,
                        expected_resource["status"],
                        ignore_order=True,
                    )
                    == {}
                )
        if not found:
            pytest.fail(
                f"Expected {expected_resource_type} was not found in response resources"
            )

    def _assert_list_resources_response(
        self,
        resources: mlrun.common.schemas.RuntimeResources,
        expected_crds=None,
        expected_pods=None,
        expected_services=None,
        **kwargs,
    ):
        expected_crds = expected_crds or []
        expected_pods = expected_pods or []
        expected_services = expected_services or []
        assert len(resources.crd_resources) == len(expected_crds)
        for index, crd in enumerate(expected_crds):
            assert resources.crd_resources[index].name == crd["metadata"]["name"]
            assert resources.crd_resources[index].labels == crd["metadata"]["labels"]
            assert resources.crd_resources[index].status == crd.get("status", {})
        assert len(resources.pod_resources) == len(expected_pods)
        for index, pod in enumerate(expected_pods):
            pod_dict = pod.to_dict()
            assert resources.pod_resources[index].name == pod_dict["metadata"]["name"]
            assert (
                resources.pod_resources[index].labels == pod_dict["metadata"]["labels"]
            )
            assert resources.pod_resources[index].status == pod_dict["status"]
        if expected_services:
            assert len(resources.service_resources) == len(expected_services)
            for index, service in enumerate(expected_services):
                assert resources.service_resources[index].name == service.metadata.name
                assert (
                    resources.service_resources[index].labels == service.metadata.labels
                )

    @staticmethod
    def _mock_list_namespaced_pods(list_pods_call_responses: list[list[client.V1Pod]]):
        calls = []
        for list_pods_call_response in list_pods_call_responses:
            pods = client.V1PodList(
                items=list_pods_call_response, metadata=client.V1ListMeta()
            )
            calls.append(pods)
        get_k8s_helper().v1api.list_namespaced_pod = unittest.mock.Mock(
            side_effect=calls
        )
        return calls

    @staticmethod
    def _assert_delete_namespaced_pods(
        expected_pod_names: list[str], expected_pod_namespace: str = None
    ):
        calls = [
            unittest.mock.call(
                expected_pod_name,
                expected_pod_namespace,
                grace_period_seconds=None,
                propagation_policy="Background",
            )
            for expected_pod_name in expected_pod_names
        ]
        if not expected_pod_names:
            assert get_k8s_helper().v1api.delete_namespaced_pod.call_count == 0
        else:
            get_k8s_helper().v1api.delete_namespaced_pod.assert_has_calls(calls)

    @staticmethod
    def _assert_delete_namespaced_services(
        expected_service_names: list[str], expected_service_namespace: str = None
    ):
        calls = [
            unittest.mock.call(
                expected_service_name,
                expected_service_namespace,
                grace_period_seconds=None,
            )
            for expected_service_name in expected_service_names
        ]
        if not expected_service_names:
            assert get_k8s_helper().v1api.delete_namespaced_service.call_count == 0
        else:
            get_k8s_helper().v1api.delete_namespaced_service.assert_has_calls(calls)

    @staticmethod
    def _assert_delete_namespaced_custom_objects(
        runtime_handler,
        expected_custom_object_names: list[str],
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
                grace_period_seconds=None,
            )
            for expected_custom_object_name in expected_custom_object_names
        ]
        if not expected_custom_object_names:
            assert (
                get_k8s_helper().crdapi.delete_namespaced_custom_object.call_count == 0
            )
        else:
            get_k8s_helper().crdapi.delete_namespaced_custom_object.assert_has_calls(
                calls
            )

    @staticmethod
    def _mock_delete_namespaced_pods():
        get_k8s_helper().v1api.delete_namespaced_pod = unittest.mock.Mock()

    @staticmethod
    def _mock_delete_namespaced_custom_objects():
        get_k8s_helper().crdapi.delete_namespaced_custom_object = unittest.mock.Mock()

    @staticmethod
    def _mock_delete_namespaced_services():
        get_k8s_helper().v1api.delete_namespaced_service = unittest.mock.Mock()

    @staticmethod
    def _mock_read_namespaced_pod_log():
        log = "Some log string"
        get_k8s_helper().v1api.read_namespaced_pod_log = unittest.mock.Mock(
            return_value=log
        )
        return log

    @staticmethod
    def _mock_list_namespaced_crds(crd_dicts_call_responses: list[list[dict]]):
        calls = []
        for crd_dicts_call_response in crd_dicts_call_responses:
            calls.append(
                {"items": crd_dicts_call_response, "metadata": {"continue": None}}
            )
        get_k8s_helper().crdapi.list_namespaced_custom_object = unittest.mock.Mock(
            side_effect=calls
        )
        return calls

    @staticmethod
    def _mock_list_namespaced_config_map(config_maps):
        config_maps_list = client.V1ConfigMapList(items=config_maps)
        get_k8s_helper().v1api.list_namespaced_config_map = unittest.mock.Mock(
            return_value=config_maps_list
        )
        return config_maps

    @staticmethod
    def _mock_list_services(services):
        services_list = client.V1ServiceList(items=services)
        get_k8s_helper().v1api.list_namespaced_service = unittest.mock.Mock(
            return_value=services_list
        )
        return services

    @staticmethod
    def _assert_list_namespaced_pods_calls(
        runtime_handler,
        expected_number_of_calls: int,
        expected_label_selector: str = None,
        paginated: bool = True,
    ):
        assert (
            get_k8s_helper().v1api.list_namespaced_pod.call_count
            == expected_number_of_calls
        ), (
            f"Unexpected number of calls to list_namespaced_pod "
            f"{get_k8s_helper().v1api.list_namespaced_pod.call_count}, expected {expected_number_of_calls}"
        )
        # if expected_label_selector and expected_label_selector != "":
        expected_label_selector = (
            expected_label_selector or runtime_handler._get_default_label_selector()
        )
        kwargs = {}
        if paginated:
            kwargs = {
                "watch": False,
                "limit": int(mlrun.mlconf.kubernetes.pagination.list_pods_limit),
                "_continue": None,
            }
        get_k8s_helper().v1api.list_namespaced_pod.assert_any_call(
            get_k8s_helper().resolve_namespace(),
            label_selector=expected_label_selector,
            **kwargs,
        )

    @staticmethod
    def _assert_list_namespaced_crds_calls(
        runtime_handler, expected_number_of_calls: int, paginated: bool = True
    ):
        crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
        assert (
            get_k8s_helper().crdapi.list_namespaced_custom_object.call_count
            == expected_number_of_calls
        )
        kwargs = {}
        if paginated:
            kwargs = {
                "watch": False,
                "limit": int(mlrun.mlconf.kubernetes.pagination.list_crd_objects_limit),
                "_continue": None,
            }
        get_k8s_helper().crdapi.list_namespaced_custom_object.assert_any_call(
            crd_group,
            crd_version,
            get_k8s_helper().resolve_namespace(),
            crd_plural,
            label_selector=runtime_handler._get_default_label_selector(),
            **kwargs,
        )

    @staticmethod
    def _assert_run_reached_state(
        db: Session,
        project: str,
        uid: str,
        expected_state: str,
        expected_status_attrs: dict = None,
    ):
        expected_status_attrs = expected_status_attrs or {}
        run = get_db().read_run(db, uid, project)
        assert run["status"]["state"] == expected_state

        for key, val in expected_status_attrs.items():
            assert run["status"][key] == val
