from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
from mlrun.runtimes.constants import MPIJobCRDVersions
from mlrun.runtimes.constants import RunStates
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestMPIjobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        config.mpijob_crd_version = MPIJobCRDVersions.v1
        self.runtime_handler = get_runtime_handler(RuntimeKinds.mpijob)

        # initializing them here to save space in tests
        self.active_crd_dict = self._generate_mpijob_crd(
            self.project, self.run_uid, self._get_active_crd_status(),
        )
        self.succeeded_crd_dict = self._generate_mpijob_crd(
            self.project, self.run_uid, self._get_succeeded_crd_status(),
        )
        self.failed_crd_dict = self._generate_mpijob_crd(
            self.project, self.run_uid, self._get_failed_crd_status(),
        )

        # there's currently a bug (fix was merged but not released https://github.com/kubeflow/mpi-operator/pull/271)
        # that causes mpijob's pods to not being labels with the given (MLRun's) labels - this prevents list resources
        # from finding the pods, so we're simulating the same thing here
        self._mock_list_namespaced_pods([[]])

    def test_list_mpijob_resources(self):
        mocked_responses = self._mock_list_namespaced_crds([[self.succeeded_crd_dict]])
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.mpijob, expected_crds=mocked_responses[0]["items"]
        )

    def test_delete_resources_succeeded_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.succeeded_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.succeeded_crd_dict["metadata"]["name"]],
            self.succeeded_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(db, self.project, self.run_uid, "")

    def test_delete_resources_running_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)

        # nothing removed cause crd is active
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler, [],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
        )

    def test_delete_resources_with_grace_period(self, db: Session, client: TestClient):
        recently_completed_crd_dict = self._generate_mpijob_crd(
            self.project,
            self.run_uid,
            self._get_succeeded_crd_status(datetime.now(timezone.utc).isoformat()),
        )
        list_namespaced_crds_calls = [
            [recently_completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10)

        # nothing removed cause grace period didn't pass
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler, [],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
        )

    def test_delete_resources_with_force(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10, force=True)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.active_crd_dict["metadata"]["name"]],
            self.active_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )
        self._assert_run_logs(db, self.project, self.run_uid, "")

    def test_monitor_run_succeeded_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
            [self.succeeded_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_crds_calls
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, expected_number_of_list_crds_calls,
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(db, self.project, self.run_uid, "")

    def test_monitor_run_failed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
            [self.failed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_crds_calls
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, expected_number_of_list_crds_calls,
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        self._assert_run_logs(db, self.project, self.run_uid, "")

    @staticmethod
    def _generate_mpijob_crd(project, uid, status=None):
        if status is None:
            status = TestMPIjobRuntimeHandler._get_succeeded_crd_status()
        crd_dict = {
            "metadata": {
                "name": "train-eaf63df8",
                "namespace": get_k8s().resolve_namespace(),
                "labels": {
                    "mlrun/class": "mpijob",
                    "mlrun/function": "trainer",
                    "mlrun/name": "train",
                    "mlrun/project": project,
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": uid,
                },
            },
            "status": status,
        }
        return crd_dict

    @staticmethod
    def _get_active_crd_status():
        return {
            "replicaStatuses": {"Launcher": {"active": 1}, "Worker": {"active": 4}},
        }

    @staticmethod
    def _get_succeeded_crd_status(timestamp=None):
        return {
            "completionTime": timestamp or "2020-10-06T00:36:41Z",
            "replicaStatuses": {"Launcher": {"succeeded": 1}, "Worker": {}},
        }

    @staticmethod
    def _get_failed_crd_status():
        return {
            "completionTime": "2020-10-06T00:36:41Z",
            "replicaStatuses": {"Launcher": {"failed": 1}, "Worker": {}},
        }
