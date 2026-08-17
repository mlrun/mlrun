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

import time
import unittest.mock
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.model
from tests.conftest import new_run

import framework.db.sqldb.db
import framework.db.sqldb.helpers
import framework.db.sqldb.models
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

    def test_list_runs_multi_label_filter(self):
        # Filtering by 2+ labels routes through SQLDB._add_labels_filter's OR+GROUP BY+HAVING
        # branch, which must return only runs matching *all* requested labels.
        project = "project"
        run_matches_both = {
            "metadata": {
                "name": "run-matches-both",
                "labels": {"workflow-id": "wf-1", "job-type": "workflow-runner"},
            },
            "status": {},
        }
        run_matches_one = {
            "metadata": {
                "name": "run-matches-one",
                "labels": {"workflow-id": "wf-1"},
            },
            "status": {},
        }
        self._db.store_run(self._db_session, run_matches_both, "uid-both", project)
        self._db.store_run(self._db_session, run_matches_one, "uid-one", project)

        query = self._db._find_runs(
            self._db_session,
            None,
            project,
            ["workflow-id=wf-1", "job-type=workflow-runner"],
        )
        assert {run.uid for run in query.all()} == {"uid-both"}

    def test_list_runs_multi_label_filter_is_postgres_group_by_safe(self):
        # On PostgreSQL, every column in a GROUP BY query's SELECT list must be either
        # grouped or aggregated. The multi-label branch used to select the full Label
        # entity (id, name, value, parent) while grouping only by `parent`, which
        # PostgreSQL rejects with psycopg2.errors.GroupingError (see ML-12863) even
        # though MySQL/SQLite silently tolerate it.
        query = self._db._find_runs(
            self._db_session,
            None,
            "project",
            ["workflow-id=wf-1", "job-type=workflow-runner"],
        )
        compiled_sql = str(query.statement.compile(dialect=postgresql.dialect()))
        labels_subquery_select = compiled_sql.split("JOIN (")[1].split(" \n")[0]
        assert labels_subquery_select == "SELECT runs_labels.parent AS parent", (
            "the multi-label OR+GROUP BY subquery must select only the grouped "
            f"column on PostgreSQL, got: {labels_subquery_select!r}"
        )

    def test_terminal_end_time_uses_clock_timestamp_on_postgres(self):
        # ML-12865: the `now` helper stamps a run's terminal end_time. On PostgreSQL it must compile to
        # clock_timestamp(), not now()/transaction_timestamp() which is frozen at transaction start - the abort
        # flow holds one transaction open across the long runtime-resource deletion, so now() would record
        # end_time in the past, dropping the run out of the notification pusher's sliding end_time window.
        # MySQL/SQLite are unaffected - only the PostgreSQL compilation is overridden.
        now = framework.db.sqldb.db.now
        assert str(now(6).compile(dialect=postgresql.dialect())) == "clock_timestamp()"
        assert "clock_timestamp" not in str(now(6).compile(dialect=mysql.dialect()))
        assert "clock_timestamp" not in str(now(6).compile(dialect=sqlite.dialect()))

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
        # The full record per uid is the representative (highest-id) row, i.e. the last
        # iteration stored. This is resolved via a portable max(id)+IN query rather than a
        # MySQL-only loose GROUP BY, so PostgreSQL's strict GROUP BY does not reject it.
        assert distinct_runs[0]["metadata"]["iter"] == 2

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
        run["status"]["start_time"] = datetime.now(UTC).isoformat()
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

    def test_store_run_success(self):
        project, name, uid, iteration, run_dict = self._create_new_run()

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

        end_time = datetime.now(UTC)
        run_dict["status"]["state"] = (
            mlrun.common.runtimes.constants.RunStates.completed
        )
        run_dict["status"]["end_time"] = end_time.isoformat()
        self._db.store_run(self._db_session, run_dict, uid, project, iter=iteration)

        runs = self._db._find_runs(
            self._db_session, uid=None, project=project, labels=None
        ).all()
        assert len(runs) == 1
        run = runs[0]
        assert (
            self._db._add_utc_timezone(run.end_time).isoformat()
            == run.struct["status"]["end_time"]
            == end_time.isoformat()
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
                {
                    "metadata.some-new-field": "value",
                    "spec.another-new-field": "value",
                    "status.state": "completed",
                },
                uid,
                project,
                iteration,
            )
            run = self._db.read_run(self._db_session, uid, project, iteration)
            assert run["metadata"]["project"] == project
            assert run["metadata"]["name"] == name
            assert run["metadata"]["some-new-field"] == "value"
            assert run["spec"]["another-new-field"] == "value"
            assert run["status"]["state"] == "completed"
            assert run["status"]["end_time"] is not None
            assert update_labels_mock.call_count == 0

    def test_store_and_update_run_from_terminal_state_to_non_terminal_state(self):
        project, name, uid, iteration, run = self._create_new_run(
            state=mlrun.common.runtimes.constants.RunStates.completed
        )
        run = self._db.read_run(self._db_session, uid, project, iteration)

        # Store completed expected to fill end time
        initial_end_time = run["status"]["end_time"]
        assert initial_end_time is not None

        # Update the run using `store` to running state to test the store flow as well
        self._create_new_run(state=mlrun.common.runtimes.constants.RunStates.running)
        run = self._db.read_run(self._db_session, uid, project, iteration)

        # Store running expected to remove end time
        assert "end_time" not in run["status"]

        # Sleep 1 second to allow next end time to be different
        time.sleep(1)
        self._db.update_run(
            self._db_session,
            {"status.state": mlrun.common.runtimes.constants.RunStates.completed},
            uid,
            project,
            iteration,
        )
        run = self._db.read_run(self._db_session, uid, project, iteration)

        # Update completed expected to fill end time
        assert run["status"]["end_time"] > initial_end_time

        self._db.update_run(
            self._db_session,
            {"status.state": mlrun.common.runtimes.constants.RunStates.running},
            uid,
            project,
            iteration,
        )
        run = self._db.read_run(self._db_session, uid, project, iteration)

        # Update running expected to remove end time
        assert "end_time" not in run["status"]

    def test_consecutive_completed_update_requests(self):
        project, name, uid, iteration, run = self._create_new_run(
            state=mlrun.common.runtimes.constants.RunStates.completed
        )
        run = self._db.read_run(self._db_session, uid, project, iteration)

        # Store completed expected to fill end time
        initial_end_time = run["status"]["end_time"]
        assert initial_end_time is not None

        self._db.update_run(
            self._db_session,
            {"status.state": mlrun.common.runtimes.constants.RunStates.completed},
            uid,
            project,
            iteration,
        )
        run = self._db.read_run(self._db_session, uid, project, iteration)
        assert run["status"]["end_time"] == initial_end_time

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

        label_key = "a" * 256
        run["metadata"]["labels"] = {label_key: "b"}
        # too long name
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=f"Name of `{label_key}` label is too long. "
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

    def test_list_runs_orders_by_id_when_start_time_is_identical(self):
        # this test verifies that when start_time date is identical, runs should be ordered by run id
        project_name = "my-project"
        t1 = datetime.now()

        # Create runs
        number_of_runs = 10
        for counter in range(number_of_runs):
            run_name = f"run-{counter}"
            self._create_new_run(
                project=project_name, name=run_name, uid=f"uid-{counter}"
            )

            # Set the same `start_time` timestamp for all runs
            self._db.update_db_object(
                self._db_session,
                framework.db.sqldb.models.Run,
                filters={"name": run_name},
                start_time=t1,
            )

        runs = self._db.list_runs(
            self._db_session,
            project=project_name,
        )
        assert len(runs) == number_of_runs, (
            f"Expected {number_of_runs} results, got {len(runs)}"
        )

        expected_names = [f"run-{i}" for i in range(number_of_runs - 1, -1, -1)]

        for run, expected_name in zip(runs, expected_names):
            run_name = run["metadata"]["name"]
            assert run_name == expected_name, (
                f"Expected {expected_name}, got {run_name}"
            )

    def test_list_runs_partitioned_no_max_partitions_has_explicit_order_by(self):
        # Regression test for ML-13004: _create_partitioned_query's max_partitions=0
        # branch (the one every "no filter" list_runs call hits) joined the ranked
        # subquery onto the runs table with no outer ORDER BY, leaving the result
        # order up to the DB engine instead of following partition_sort_by/order.
        #
        # SQLite happens to preserve the window function's internal sort into the
        # final join for this query shape regardless of whether the fix is present,
        # so an end-to-end "is the returned row order correct" assertion can't
        # actually distinguish fixed from broken here (verified: it passes either
        # way). The real regression guard is inspecting the compiled SQL for an
        # explicit ORDER BY - this is what MySQL's optimizer needs to produce a
        # deterministic result, and it's what the fix actually adds.
        project_name = "my-project"
        query = self._db._find_runs(self._db_session, None, project_name)
        partitioned_query = self._db._create_partitioned_query(
            self._db_session,
            query,
            framework.db.sqldb.models.Run,
            mlrun.common.schemas.RunPartitionByField.project_and_name,
            rows_per_partition=5,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
        )
        compiled_sql = str(partitioned_query.statement.compile(dialect=mysql.dialect()))
        # The row_number() window function's own OVER(ORDER BY ...) clause always
        # contains the substring "ORDER BY" - that's not what's under test. A
        # second, outer-level ORDER BY (added by the fix) is what's needed for a
        # deterministic final result, so assert on the count, not just presence.
        assert compiled_sql.count("ORDER BY") >= 2, (
            "expected an explicit outer ORDER BY on the partitioned query in "
            f"addition to the window function's own OVER(ORDER BY ...), got: {compiled_sql}"
        )

        # End-to-end sanity check: still exercises the real list_runs() path and
        # documents the intended behaviour, even though it can't fail on SQLite.
        run_name = "run-same-name"
        number_of_runs = 5
        base_time = datetime(2026, 1, 1, tzinfo=UTC)
        for counter in range(number_of_runs):
            uid = f"uid-{counter}"
            self._create_new_run(project=project_name, name=run_name, uid=uid)
            self._db.update_db_object(
                self._db_session,
                framework.db.sqldb.models.Run,
                filters={"uid": uid},
                updated=base_time + timedelta(seconds=counter),
            )

        runs = self._db.list_runs(
            self._db_session,
            project=project_name,
            partition_by=mlrun.common.schemas.RunPartitionByField.project_and_name,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
            rows_per_partition=number_of_runs,
        )
        assert len(runs) == number_of_runs
        expected_uids = [f"uid-{i}" for i in range(number_of_runs - 1, -1, -1)]
        actual_uids = [run["metadata"]["uid"] for run in runs]
        assert actual_uids == expected_uids, (
            f"Expected runs ordered newest-updated-first {expected_uids}, "
            f"got {actual_uids}"
        )

    def test_list_runs_partitioned_with_max_partitions_has_explicit_order_by(self):
        # Regression test for ML-13004: the max_partitions>0 branch of
        # _create_partitioned_query has the same missing-ORDER-BY gap as the
        # max_partitions=0 branch. See the comment in
        # test_list_runs_partitioned_no_max_partitions_has_explicit_order_by for why
        # the compiled-SQL assertion, not row order, is the real regression guard.
        project_name = "my-project"
        query = self._db._find_runs(self._db_session, None, project_name)
        partitioned_query = self._db._create_partitioned_query(
            self._db_session,
            query,
            framework.db.sqldb.models.Run,
            mlrun.common.schemas.RunPartitionByField.project_and_name,
            rows_per_partition=2,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
            max_partitions=2,
        )
        compiled_sql = str(partitioned_query.statement.compile(dialect=mysql.dialect()))
        # This branch already has two window functions pre-fix - row_number() and
        # dense_rank() - each with its own OVER(ORDER BY ...), contributing 2
        # occurrences of "ORDER BY" regardless of the fix. A third, outer-level
        # ORDER BY (added by the fix) is what's needed for a deterministic final
        # result, so assert on the count, not just presence.
        assert compiled_sql.count("ORDER BY") >= 3, (
            "expected an explicit outer ORDER BY on the partitioned query in "
            "addition to the row_number()/dense_rank() window functions' own "
            f"OVER(ORDER BY ...) clauses, got: {compiled_sql}"
        )

        # End-to-end sanity check: still exercises the real list_runs() path and
        # documents the intended behaviour, even though it can't fail on SQLite.
        base_time = datetime(2026, 1, 1, tzinfo=UTC)
        partitions = [
            ("run-oldest", 0),
            ("run-newest", 20),
            ("run-middle", 10),
        ]
        for run_name, offset_seconds in partitions:
            for counter in range(2):
                uid = f"{run_name}-uid-{counter}"
                self._create_new_run(project=project_name, name=run_name, uid=uid)
                self._db.update_db_object(
                    self._db_session,
                    framework.db.sqldb.models.Run,
                    filters={"uid": uid},
                    updated=base_time + timedelta(seconds=offset_seconds + counter),
                )

        runs = self._db.list_runs(
            self._db_session,
            project=project_name,
            partition_by=mlrun.common.schemas.RunPartitionByField.project_and_name,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
            rows_per_partition=2,
            max_partitions=2,
        )
        run_names = [run["metadata"]["name"] for run in runs]
        assert set(run_names) == {"run-newest", "run-middle"}
        assert len(runs) == 4
        expected_order = [
            "run-newest-uid-1",
            "run-newest-uid-0",
            "run-middle-uid-1",
            "run-middle-uid-0",
        ]
        actual_order = [run["metadata"]["uid"] for run in runs]
        assert actual_order == expected_order, (
            f"Expected {expected_order}, got {actual_order}"
        )

    def test_list_runs_with_missing_milliseconds_in_timestamp(self):
        self._create_new_run(project="my-project")

        t1 = datetime.now().replace(microsecond=0)

        # Set the `start_time` and `end_time` timestamps without microseconds
        self._db.update_db_object(
            self._db_session, framework.db.sqldb.models.Run, start_time=t1, end_time=t1
        )

        runs = self._db.list_runs(self._db_session, project="my-project")
        assert len(runs) == 1

        assert runs[0]["status"]["start_time"].endswith(".000000+00:00")
        assert runs[0]["status"]["end_time"].endswith(".000000+00:00")

    def test_list_runs_empty_project_list_returns_empty(self):
        # Cross-project listing (project="*") for a user with no accessible projects
        # resolves to an empty project list. That must yield an empty result, not an error.
        self._create_new_run(project="some-project")

        runs = self._db.list_runs(self._db_session, project=[])
        assert len(runs) == 0

        # A populated project list still filters normally (the empty-list relaxation doesn't
        # weaken the list path).
        runs = self._db.list_runs(self._db_session, project=["some-project"])
        assert len(runs) == 1

    @pytest.mark.parametrize("project", [None, ""])
    def test_list_runs_missing_project_raises(self, project):
        # A truly missing project (None / "") applies no project filter, so it must keep
        # raising rather than silently listing across all projects.
        with pytest.raises(mlrun.errors.MLRunMissingProjectError):
            self._db.list_runs(self._db_session, project=project)

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
