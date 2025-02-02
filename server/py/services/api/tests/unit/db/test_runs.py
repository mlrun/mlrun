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
from datetime import datetime, timedelta, timezone

import pytest

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.model
from tests.conftest import new_run

import framework.db.sqldb.helpers
import services.api.initial_data
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestRuns(TestDatabaseBase):
    def test_list_runs_name_filter(self):
        project = "project"
        run_name_1 = "run_name_1"
        run_name_2 = "run_name_2"
        run_1 = {"metadata": {"name": run_name_1}, "status": {"bla": "blabla"}}
        run_2 = {"metadata": {"name": run_name_2}, "status": {"bla": "blabla"}}
        # run with no name (had name but filled with no-name after version 2 data migration)
        run_3 = {"metadata": {"name": "no-name"}, "status": {"bla": "blabla"}}
        run_uid_1 = "run_uid_1"
        run_uid_2 = "run_uid_2"
        run_uid_3 = "run_uid_3"

        self._db.store_run(self._db_session, run_1, run_uid_1, project)
        self._db.store_run(self._db_session, run_2, run_uid_2, project)
        self._db.store_run(self._db_session, run_3, run_uid_3, project)
        runs = self._db.list_runs(self._db_session, project=project)
        assert len(runs) == 3

        runs = self._db.list_runs(self._db_session, name=run_name_1, project=project)
        assert len(runs) == 1
        assert runs[0]["metadata"]["name"] == run_name_1

        runs = self._db.list_runs(self._db_session, name=run_name_2, project=project)
        assert len(runs) == 1
        assert runs[0]["metadata"]["name"] == run_name_2

        runs = self._db.list_runs(self._db_session, name="~RUN_naMe", project=project)
        assert len(runs) == 2

    def test_runs_with_notifications(self):
        project_name = "project"
        run_uids = ["uid1", "uid2", "uid3"]
        num_runs = len(run_uids)
        # create several runs with different uids, each with a notification
        for run_uid in run_uids:
            self._create_new_run(project=project_name, uid=run_uid)
            notification = mlrun.model.Notification(
                kind="slack",
                when=["completed", "error"],
                name=f"test-notification-{run_uid}",
                message="test-message",
                condition="blabla",
                severity="info",
                params={"some-param": "some-value"},
            )
            self._db.store_run_notifications(
                self._db_session, [notification], run_uid, project_name
            )

        runs = self._db.list_runs(
            self._db_session, project=project_name, with_notifications=True
        )
        assert len(runs) == num_runs
        for run in runs:
            run_notifications = run["spec"]["notifications"]
            assert len(run_notifications) == 1
            assert (
                run_notifications[0]["name"]
                == f"test-notification-{run['metadata']['uid']}"
            )

        self._db.delete_run_notifications(
            self._db_session, run_uid=run_uids[0], project=project_name
        )
        runs = self._db.list_runs(
            self._db_session, project=project_name, with_notifications=True
        )
        assert len(runs) == num_runs - 1

        self._db.delete_run_notifications(self._db_session, project=project_name)
        runs = self._db.list_runs(
            self._db_session, project=project_name, with_notifications=False
        )
        assert len(runs) == num_runs
        runs = self._db.list_runs(
            self._db_session, project=project_name, with_notifications=True
        )
        assert len(runs) == 0

        self._db.del_runs(self._db_session, project=project_name)
        self._db.verify_project_has_no_related_resources(self._db_session, project_name)

    def test_list_runs_with_notifications_identical_run_names(self):
        project_name = "project"

        self._create_new_run(project=project_name, name="test-run", uid="uid1")
        notification = mlrun.model.Notification(
            kind="slack",
            when=["completed", "error"],
            name="test-notification",
            message="test-message",
            condition="blabla",
            severity="info",
            params={"some-param": "some-value"},
        )
        self._db.store_run_notifications(
            self._db_session, [notification], "uid1", project_name
        )

        # same name, different uid
        self._create_new_run(project=project_name, name="test-run", uid="uid2")

        # default query with partition should only return the last run of the same name. this is done in the endpoint
        # and in the httpdb client, so we'll implement it here manually as this db instance goes directly to the sql db
        # implementation.
        partition_by = mlrun.common.schemas.RunPartitionByField.project_and_name
        partition_sort_by = mlrun.common.schemas.SortField.updated

        runs = self._db.list_runs(
            self._db_session,
            project=project_name,
            with_notifications=True,
            partition_by=partition_by,
            partition_sort_by=partition_sort_by,
        )
        assert len(runs) == 1

        runs = self._db.list_runs(
            self._db_session,
            project=project_name,
            with_notifications=False,
            partition_by=partition_by,
            partition_sort_by=partition_sort_by,
        )
        assert len(runs) == 1

        # without partitioning, we should get all runs when querying without notifications and only the first run
        # when querying with notifications
        runs = self._db.list_runs(
            self._db_session, project=project_name, with_notifications=True
        )
        assert len(runs) == 1

        runs = self._db.list_runs(
            self._db_session, project=project_name, with_notifications=False
        )
        assert len(runs) == 2

        self._db.del_runs(self._db_session, project=project_name)
        self._db.verify_project_has_no_related_resources(self._db_session, project_name)

    def test_list_distinct_runs_uids(self):
        project_name = "project"
        uid = "run-uid"
        # create 3 runs with same uid but different iterations
        for i in range(3):
            self._create_new_run(project=project_name, iteration=i, uid=uid)

        runs = self._db.list_runs(self._db_session, project=project_name, iter=True)
        assert len(runs) == 3

        distinct_runs = self._db.list_distinct_runs_uids(
            self._db_session, project=project_name, only_uids=False
        )
        assert len(distinct_runs) == 1
        assert isinstance(distinct_runs[0], dict)
        assert distinct_runs[0]["metadata"]["uid"] == uid

        only_uids = self._db.list_distinct_runs_uids(
            self._db_session, project=project_name, only_uids=True
        )
        assert len(only_uids) == 1
        assert isinstance(only_uids[0], str)
        assert only_uids[0] == uid

        only_uids_requested_true = self._db.list_distinct_runs_uids(
            self._db_session,
            project=project_name,
            only_uids=True,
            requested_logs_modes=[True],
        )
        assert len(only_uids_requested_true) == 0

        only_uids_requested_false = self._db.list_distinct_runs_uids(
            self._db_session,
            project=project_name,
            only_uids=True,
            requested_logs_modes=[False],
        )
        assert len(only_uids_requested_false) == 1
        assert isinstance(only_uids[0], str)

        distinct_runs_requested_true = self._db.list_distinct_runs_uids(
            self._db_session, project=project_name, requested_logs_modes=[True]
        )
        assert len(distinct_runs_requested_true) == 0

        distinct_runs_requested_false = self._db.list_distinct_runs_uids(
            self._db_session, project=project_name, requested_logs_modes=[False]
        )
        assert len(distinct_runs_requested_false) == 1
        assert isinstance(distinct_runs[0], dict)

    def test_list_runs_state_filter(self):
        project = "project"
        run_uid_running = "run-running"
        run_uid_completed = "run-completed"
        self._create_new_run(
            project,
            uid=run_uid_running,
            state=mlrun.common.runtimes.constants.RunStates.running,
        )
        self._create_new_run(
            project,
            uid=run_uid_completed,
            state=mlrun.common.runtimes.constants.RunStates.completed,
        )
        runs = self._db.list_runs(self._db_session, project=project)
        assert len(runs) == 2

        runs = self._db.list_runs(
            self._db_session,
            project=project,
            states=[mlrun.common.runtimes.constants.RunStates.running],
        )
        assert len(runs) == 1
        assert runs[0]["metadata"]["uid"] == run_uid_running

        runs = self._db.list_runs(
            self._db_session,
            project=project,
            states=[mlrun.common.runtimes.constants.RunStates.completed],
        )
        assert len(runs) == 1
        assert runs[0]["metadata"]["uid"] == run_uid_completed

    def test_store_run_overriding_start_time(self):
        # First store - fills the start_time
        project, name, uid, iteration, run = self._create_new_run()

        # use to internal function to get the record itself to be able to assert the column itself
        runs = self._db._find_runs(
            self._db_session, uid=None, project=project, labels=None
        ).all()
        assert len(runs) == 1
        assert (
            self._db._add_utc_timezone(runs[0].start_time).isoformat()
            == runs[0].struct["status"]["start_time"]
        )

        # Second store - should allow to override the start time
        run["status"]["start_time"] = datetime.now(timezone.utc).isoformat()
        self._db.store_run(self._db_session, run, uid, project)

        # get the start time and verify
        runs = self._db._find_runs(
            self._db_session, uid=None, project=project, labels=None
        ).all()
        assert len(runs) == 1
        assert (
            self._db._add_utc_timezone(runs[0].start_time).isoformat()
            == runs[0].struct["status"]["start_time"]
        )
        assert runs[0].struct["status"]["start_time"] == run["status"]["start_time"]

    def test_data_migration_align_runs_table(self):
        time_before_creation = datetime.now(tz=timezone.utc)
        # Create runs
        for project in ["run-project-1", "run-project-2", "run-project-3"]:
            for name in ["run-name-1", "run-name-2", "run-name-3"]:
                for index in range(3):
                    uid = f"{name}-uid-{index}"
                    for iteration in range(3):
                        self._create_new_run(
                            project,
                            name,
                            uid,
                            iteration,
                            state=mlrun.common.runtimes.constants.RunStates.completed,
                        )
        # get all run records, and change to be as they will be in field (before the migration)
        runs = self._db._find_runs(self._db_session, None, "*", None).all()
        for run in runs:
            self._change_run_record_to_before_align_runs_migration(
                run, time_before_creation
            )
            self._db._upsert(self._db_session, [run], ignore=True)

        # run the migration
        services.api.initial_data._align_runs_table(self._db, self._db_session)

        # assert after migration column start time aligned to the body start time
        runs = self._db._find_runs(self._db_session, None, "*", None).all()
        for run in runs:
            self._ensure_run_after_align_runs_migration(run, time_before_creation)

    def test_data_migration_align_runs_table_with_empty_run_body(self):
        time_before_creation = datetime.now(tz=timezone.utc)
        # First store - fills the start_time
        project, name, uid, iteration, run = self._create_new_run(
            state=mlrun.common.runtimes.constants.RunStates.completed
        )
        # get all run records, and change to be as they will be in field (before the migration)
        runs = self._db._find_runs(self._db_session, None, "*", None).all()
        assert len(runs) == 1
        run = runs[0]
        # change to be as it will be in field (before the migration) and then empty the body
        self._change_run_record_to_before_align_runs_migration(
            run, time_before_creation
        )
        run.struct = {}
        self._db._upsert(self._db_session, [run], ignore=True)

        # run the migration
        services.api.initial_data._align_runs_table(self._db, self._db_session)

        runs = self._db._find_runs(self._db_session, None, "*", None).all()
        assert len(runs) == 1
        run = runs[0]
        self._ensure_run_after_align_runs_migration(run)

    def test_store_run_success(self):
        project, name, uid, iteration, run = self._create_new_run()

        # use to internal function to get the record itself to be able to assert columns
        runs = self._db._find_runs(
            self._db_session, uid=None, project=project, labels=None
        ).all()
        assert len(runs) == 1
        run = runs[0]
        assert run.project == project
        assert run.name == name
        assert run.uid == uid
        assert run.iteration == iteration
        assert run.state == mlrun.common.runtimes.constants.RunStates.created
        assert run.state == run.struct["status"]["state"]
        assert (
            self._db._add_utc_timezone(run.start_time).isoformat()
            == run.struct["status"]["start_time"]
        )

        assert (
            self._db._add_utc_timezone(run.updated).isoformat()
            == run.struct["status"]["last_update"]
        )

    def test_update_runs_requested_logs(self):
        project, name, uid, iteration, run = self._create_new_run()

        runs_before = self._db.list_runs(
            self._db_session, project=project, uid=uid, return_as_run_structs=False
        )
        assert runs_before[0].requested_logs is False
        run_updated_time = runs_before[0].updated

        self._db.update_runs_requested_logs(self._db_session, [uid], True)

        runs_after = self._db.list_runs(
            self._db_session, project=project, uid=uid, return_as_run_structs=False
        )
        assert runs_after[0].requested_logs is True
        assert runs_after[0].updated > run_updated_time

    def test_update_run_success(self):
        project, name, uid, iteration, run = self._create_new_run()

        with unittest.mock.patch(
            "framework.db.sqldb.helpers.update_labels", return_value=None
        ) as update_labels_mock:
            self._db.update_run(
                self._db_session,
                {"metadata.some-new-field": "value", "spec.another-new-field": "value"},
                uid,
                project,
                iteration,
            )
            run = self._db.read_run(self._db_session, uid, project, iteration)
            assert run["metadata"]["project"] == project
            assert run["metadata"]["name"] == name
            assert run["metadata"]["some-new-field"] == "value"
            assert run["spec"]["another-new-field"] == "value"
            assert update_labels_mock.call_count == 0

    def test_run_iter(self):
        uid, prj = "uid39", "lemon"
        run = new_run("s1", {"l1": "v1", "l2": "v2"}, x=1)
        for i in range(7):
            self._db.store_run(self._db_session, run, uid, prj, i)
        self._db._get_run(self._db_session, uid, prj, 0)  # See issue 140

    def test_update_run_labels(self):
        project, name, uid, iteration, run = self._create_new_run()

        self._db.update_run(
            self._db_session,
            {"metadata.labels": {"a": "b"}},
            uid,
            project,
            iteration,
        )
        run = self._db.read_run(self._db_session, uid, project, iteration)
        assert run["metadata"]["labels"] == {"a": "b"}

        run["metadata"]["labels"] = {"a": "b" * 256}
        # too long value
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Value of `a` label is too long. "
            "Maximum allowed length is 255 characters.",
        ):
            self._db.update_run(
                self._db_session,
                run,
                uid,
                project,
                iteration,
            )

    def test_store_and_update_run_update_name_failure(self):
        project, name, uid, iteration, run = self._create_new_run()

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Changing name for an existing run is invalid",
        ):
            run["metadata"]["name"] = "new-name"
            self._db.store_run(
                self._db_session,
                run,
                uid,
                project,
                iteration,
            )

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Changing name for an existing run is invalid",
        ):
            self._db.update_run(
                self._db_session,
                {"metadata.name": "new-name"},
                uid,
                project,
                iteration,
            )

    def test_list_runs_limited_unsorted_failure(self):
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Limiting the number of returned records without sorting will provide non-deterministic results",
        ):
            self._db.list_runs(
                self._db_session,
                sort=False,
                last=1,
            )

    def test_list_runs_with_same_names(self):
        run_names = ["run_name_1", "run_name_2"]
        project_names = ["project1", "project2"]
        for run_name in run_names:
            for project_name in project_names:
                run = {"metadata": {"name": run_name}, "status": {"bla": "blabla"}}
                run_uid = f"{run_name}-{project_name}"
                self._db.store_run(self._db_session, run, run_uid, project_name)

        runs = self._db.list_runs(
            self._db_session,
            project="*",
            partition_sort_by=mlrun.common.schemas.SortField.created,
            partition_by=mlrun.common.schemas.RunPartitionByField.name,
        )
        assert len(runs) == 2

        runs = self._db.list_runs(
            self._db_session,
            project="*",
            partition_sort_by=mlrun.common.schemas.SortField.created,
            partition_by=mlrun.common.schemas.RunPartitionByField.project_and_name,
        )
        assert len(runs) == 4

    def test_list_runs_with_end_time(self):
        project, name, run_uid, iteration, run = self._create_new_run()

        assert not run["status"].get("end_time")

        # update the run's end_time
        end_time = datetime.now(timezone.utc)
        end_time_iso = end_time.isoformat()
        updates = {"status.end_time": end_time_iso}
        self._db.update_run(self._db_session, updates, run_uid, project)

        # fetch the run and verify the end_time
        run = self._db.read_run(self._db_session, run_uid, project, iteration)
        assert run["status"].get("end_time")
        assert run["status"]["end_time"] == end_time_iso

        # list runs with end_time filter
        runs = self._db.list_runs(
            self._db_session,
            project=project,
            end_time_from=end_time - timedelta(milliseconds=100),
        )
        assert len(runs) == 1
        stored_run = runs[0]
        assert stored_run["metadata"]["uid"] == run_uid
        assert stored_run["status"]["end_time"] == end_time_iso
        assert stored_run["status"]["end_time"] > stored_run["status"]["start_time"]

    @staticmethod
    def _change_run_record_to_before_align_runs_migration(run, time_before_creation):
        run_dict = run.struct

        # change only the start_time column (and not the field in the body) to be earlier
        assert (
            framework.db.sqldb.helpers.run_start_time(run_dict) > time_before_creation
        )
        run.start_time = time_before_creation

        # change name column to be empty
        run.name = None

        # change state column to be empty created (should be completed)
        run.state = mlrun.common.runtimes.constants.RunStates.created

        # change updated column to be empty
        run.updated = None

    def _ensure_run_after_align_runs_migration(self, run, time_before_creation=None):
        run_dict = run.struct

        # ensure start time aligned
        assert framework.db.sqldb.helpers.run_start_time(
            run_dict
        ) == self._db._add_utc_timezone(run.start_time)
        if time_before_creation is not None:
            assert (
                framework.db.sqldb.helpers.run_start_time(run_dict)
                > time_before_creation
            )

        # ensure name column filled
        assert run_dict["metadata"]["name"] == run.name

        # ensure state column aligned
        assert run_dict["status"]["state"] == run.state

        # ensure updated column filled
        assert (
            run_dict["status"]["last_update"]
            == self._db._add_utc_timezone(run.updated).isoformat()
        )

    def _create_new_run(
        self,
        project="project",
        name="run-name-1",
        uid="run-uid",
        iteration=0,
        state=mlrun.common.runtimes.constants.RunStates.created,
    ):
        run = {
            "metadata": {
                "name": name,
                "uid": uid,
                "project": project,
                "iter": iteration,
            },
            "status": {"state": state},
        }

        self._db.store_run(self._db_session, run, uid, project, iter=iteration)
        return project, name, uid, iteration, run
