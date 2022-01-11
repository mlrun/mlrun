from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import tests.conftest
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds, get_runtime_handler
from mlrun.runtimes.constants import PodPhases, RunStates
from mlrun.utils import now_date
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.runtime_handler = get_runtime_handler(RuntimeKinds.job)
        self.runtime_handler.wait_for_deletion_interval = 0

        job_labels = {
            "mlrun/class": self._get_class_name(),
            "mlrun/function": "my-trainer",
            "mlrun/name": "my-training",
            "mlrun/project": self.project,
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "mlrun/uid": self.run_uid,
        }
        job_pod_name = "my-training-j7dtf"

        # initializing them here to save space in tests
        self.pending_job_pod = self._generate_pod(
            job_pod_name, job_labels, PodPhases.pending
        )
        self.running_job_pod = self._generate_pod(
            job_pod_name, job_labels, PodPhases.running
        )
        self.completed_job_pod = self._generate_pod(
            job_pod_name, job_labels, PodPhases.succeeded
        )
        self.failed_job_pod = self._generate_pod(
            job_pod_name, job_labels, PodPhases.failed
        )

        builder_legacy_labels = {
            "mlrun/class": "build",
            "mlrun/task-name": "mlrun-build-hedi-simple-func-legacy",
        }
        builder_legacy_pod_name = "mlrun-build-hedi-simple-legacy-func-8qwrd"
        self.completed_legacy_builder_pod = self._generate_pod(
            builder_legacy_pod_name, builder_legacy_labels, PodPhases.succeeded
        )

    def _get_class_name(self):
        return "job"

    def test_list_resources(self, db: Session, client: TestClient):
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, expected_pods=pods
        )

    def test_list_resources_grouped_by(self, db: Session, client: TestClient):
        for group_by in [
            mlrun.api.schemas.ListRuntimeResourcesGroupByField.job,
            mlrun.api.schemas.ListRuntimeResourcesGroupByField.project,
        ]:
            pods = self._mock_list_resources_pods()
            self._assert_runtime_handler_list_resources(
                RuntimeKinds.job, expected_pods=pods, group_by=group_by,
            )

    def test_list_resources_grouped_by_project_with_non_project_resources(
        self, db: Session, client: TestClient
    ):
        pods = self._mock_list_resources_pods(self.completed_legacy_builder_pod)
        resources = self._assert_runtime_handler_list_resources(
            RuntimeKinds.job,
            expected_pods=pods,
            group_by=mlrun.api.schemas.ListRuntimeResourcesGroupByField.project,
        )
        # the legacy builder pod does not have a project label, verify it is listed under the empty key
        # so it will be removed on cleanup
        assert "" in resources

    def test_delete_resources_completed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.completed_job_pod],
            # additional time for the get_logger_pods
            [self.completed_job_pod],
            # additional time for wait for pods deletion - simulate pod not removed yet
            [self.completed_job_pod],
            # additional time for wait for pods deletion - simulate pod gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_pods(
            [self.completed_job_pod.metadata.name],
            self.completed_job_pod.metadata.namespace,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_job_pod.metadata.name,
        )

    def test_delete_resources_completed_builder_pod(
        self, db: Session, client: TestClient
    ):
        """
        Test mainly used to verify that we're not spamming errors in logs in this specific scenario
        """
        list_namespaced_pods_calls = [
            [self.completed_legacy_builder_pod],
            # additional time for the get_logger_pods
            [self.completed_legacy_builder_pod],
            # additional time for wait for pods deletion - simulate pod not removed yet
            [self.completed_legacy_builder_pod],
            # additional time for wait for pods deletion - simulate pod gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_pods(
            [self.completed_legacy_builder_pod.metadata.name],
            self.completed_legacy_builder_pod.metadata.namespace,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_delete_resources_running_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.running_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)

        # nothing removed cause pod is running
        self._assert_delete_namespaced_pods([])
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_delete_resources_with_grace_period(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.completed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10)

        # nothing removed cause pod grace period didn't pass
        self._assert_delete_namespaced_pods([])
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_delete_resources_with_force(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.running_job_pod],
            # additional time for the get_logger_pods
            [self.running_job_pod],
            # additional time for wait for pods deletion - simulate pod gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10, force=True)
        self._assert_delete_namespaced_pods(
            [self.running_job_pod.metadata.name],
            self.running_job_pod.metadata.namespace,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.running_job_pod.metadata.name,
        )

    def test_monitor_run_completed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_job_pod],
            [self.running_job_pod],
            [self.completed_job_pod],
            # additional time for the get_logger_pods
            [self.completed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_job_pod.metadata.name,
        )

    def test_monitor_run_failed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_job_pod],
            [self.running_job_pod],
            [self.failed_job_pod],
            # additional time for the get_logger_pods
            [self.failed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.failed_job_pod.metadata.name,
        )

    def test_monitor_run_no_pods(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [],
            [],
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.created
        )
        self._assert_run_logs(db, self.project, self.run_uid, "")

    def test_monitor_run_overriding_terminal_state(
        self, db: Session, client: TestClient
    ):
        list_namespaced_pods_calls = [
            [self.failed_job_pod],
            # additional time for the get_logger_pods
            [self.failed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
        self.run["status"]["state"] = RunStates.completed
        mlrun.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_job_pod.metadata.name,
        )

    def test_monitor_run_debouncing_non_terminal_state(
        self, db: Session, client: TestClient
    ):
        # set monitoring interval so debouncing will be active
        config.runs_monitoring_interval = 100

        # Mocking the SDK updating the Run's state to terminal state
        self.run["status"]["state"] = RunStates.completed
        original_update_run_updated_time = (
            mlrun.api.utils.singletons.db.get_db()._update_run_updated_time
        )
        mlrun.api.utils.singletons.db.get_db()._update_run_updated_time = tests.conftest.freeze(
            original_update_run_updated_time, now=now_date()
        )
        mlrun.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        mlrun.api.utils.singletons.db.get_db()._update_run_updated_time = (
            original_update_run_updated_time
        )

        # Mocking pod that is still in non-terminal state
        self._mock_list_namespaced_pods([[self.running_job_pod]])

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )

        # Mocking that update occurred before debounced period
        debounce_period = config.runs_monitoring_interval
        mlrun.api.utils.singletons.db.get_db()._update_run_updated_time = tests.conftest.freeze(
            original_update_run_updated_time,
            now=now_date() - timedelta(seconds=float(2 * debounce_period)),
        )
        mlrun.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        mlrun.api.utils.singletons.db.get_db()._update_run_updated_time = (
            original_update_run_updated_time
        )

        # Mocking pod that is still in non-terminal state
        self._mock_list_namespaced_pods([[self.running_job_pod]])

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was not debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )

        # Mocking pod that is in terminal state (extra one for the log collection)
        self._mock_list_namespaced_pods(
            [[self.completed_job_pod], [self.completed_job_pod]]
        )

        # Mocking read log calls
        log = self._mock_read_namespaced_pod_log()

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was not debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )

        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_job_pod.metadata.name,
        )

    def test_monitor_run_run_does_not_exists(self, db: Session, client: TestClient):
        get_db().del_run(db, self.run_uid, self.project)
        list_namespaced_pods_calls = [
            [self.completed_job_pod],
            # additional time for the get_logger_pods
            [self.completed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_job_pod.metadata.name,
        )

    def _mock_list_resources_pods(self, pod=None):
        pod = pod or self.completed_job_pod
        mocked_responses = self._mock_list_namespaced_pods([[pod]])
        return mocked_responses[0].items
