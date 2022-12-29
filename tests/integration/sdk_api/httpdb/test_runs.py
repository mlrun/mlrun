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
import http
import json

import pytest

import mlrun
import mlrun.api.schemas
import tests.integration.sdk_api.base


class TestRuns(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_store_big_run(self):
        """
        Sometimes when the run has artifacts (inputs or outputs) their preview is pretty big (but it is limited to some
        size), when we moved to MySQL a run similar to the one this test is storing was failing to be read from the DB
        after insert on _pickle.UnpicklingError: pickle data was truncated
        So we fixed this by changing the BLOB fields to sqlalchemy.dialects.mysql.MEDIUMBLOB
        This test verifies it's working
        """
        project_name = "runs-project"
        project = mlrun.new_project(project_name)
        project.save()
        uid = "some-uid"
        run_body_path = str(self.assets_path / "big-run.json")
        with open(run_body_path) as run_body_file:
            run_body = json.load(run_body_file)
        mlrun.get_run_db().store_run(run_body, uid, project_name)
        mlrun.get_run_db().read_run(uid, project_name)

    def test_list_runs(self):
        # Create runs
        projects = ["run-project-1", "run-project-2", "run-project-3"]
        run_names = ["run-name-1", "run-name-2", "run-name-3"]
        suffixes = ["first", "second", "third"]
        for project in projects:
            project_obj = mlrun.new_project(project)
            project_obj.save()
            for name in run_names:
                for suffix in suffixes:
                    uid = f"{name}-uid-{suffix}"
                    for iteration in range(3):
                        run = {
                            "metadata": {
                                "name": name,
                                "uid": uid,
                                "project": project,
                                "iter": iteration,
                            },
                        }
                        mlrun.get_run_db().store_run(run, uid, project, iteration)

        # basic list, all projects, all iterations so 3 projects * 3 names * 3 uids * 3 iterations = 81
        runs = _list_and_assert_objects(81, project="*", iter=True)

        # basic list, specific project, all iterations, so 3 names * 3 uids * 3 iterations = 27
        runs = _list_and_assert_objects(27, project=projects[0], iter=True)

        # basic list, specific project, only iteration 0, so 3 names * 3 uids = 9
        runs = _list_and_assert_objects(9, project=projects[0], iter=False)

        # partioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
        runs = _list_and_assert_objects(
            3,
            project=projects[0],
            partition_by=mlrun.api.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.api.schemas.SortField.created,
            partition_order=mlrun.api.schemas.OrderType.asc,
        )
        # sorted by ascending created so only the first ones created
        for run in runs:
            assert "first" in run["metadata"]["uid"]

        # partioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
        runs = _list_and_assert_objects(
            3,
            project=projects[0],
            partition_by=mlrun.api.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.api.schemas.SortField.updated,
            partition_order=mlrun.api.schemas.OrderType.desc,
        )
        # sorted by descending updated so only the third ones created
        for run in runs:
            assert "third" in run["metadata"]["uid"]

        # partioned list, specific project, 5 row per partition, so 3 names * 5 row = 15
        runs = _list_and_assert_objects(
            15,
            project=projects[0],
            partition_by=mlrun.api.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.api.schemas.SortField.updated,
            partition_order=mlrun.api.schemas.OrderType.desc,
            rows_per_partition=5,
            iter=True,
        )

        # partitioned list, specific project, 5 rows per partition, max of 2 partitions, so 2 names * 5 rows = 10
        runs = _list_and_assert_objects(
            10,
            project=projects[0],
            partition_by=mlrun.api.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.api.schemas.SortField.updated,
            partition_order=mlrun.api.schemas.OrderType.desc,
            rows_per_partition=5,
            max_partitions=2,
            iter=True,
        )

        # partitioned list, specific project, 4 rows per partition, max of 2 partitions, but only iter=0 so each
        # partition has 3 rows, so 2 * 3 = 6
        runs = _list_and_assert_objects(
            6,
            project=projects[0],
            partition_by=mlrun.api.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.api.schemas.SortField.updated,
            partition_order=mlrun.api.schemas.OrderType.desc,
            rows_per_partition=4,
            max_partitions=2,
            iter=False,
        )

        # Some negative testing - no sort by field
        with pytest.raises(mlrun.errors.MLRunBadRequestError):
            _list_and_assert_objects(
                0,
                project=projects[0],
                partition_by=mlrun.api.schemas.RunPartitionByField.name,
            )
        # An invalid partition-by field - will be failed by fastapi due to schema validation.
        with pytest.raises(mlrun.errors.MLRunHTTPError) as excinfo:
            _list_and_assert_objects(
                0,
                project=projects[0],
                partition_by="key",
                partition_sort_by=mlrun.api.schemas.SortField.updated,
            )
        assert (
            excinfo.value.response.status_code
            == http.HTTPStatus.UNPROCESSABLE_ENTITY.value
        )

        # expecting 3 since we're getting back all iterations for that uid
        _list_and_assert_objects(
            3,
            project=projects[0],
            uid=f"{run_names[0]}-uid-{suffixes[0]}",
            iter=True,
        )

        uid_list = [f"{run_names[0]}-uid-{suffix}" for suffix in suffixes]
        runs = _list_and_assert_objects(
            len(uid_list),
            project=projects[0],
            uid=uid_list,
            iter=False,
        )
        uid_list = set(uid_list)
        for run in runs:
            assert run["metadata"]["uid"] in uid_list
            uid_list.remove(run["metadata"]["uid"])


def _list_and_assert_objects(expected_number_of_runs: int, **kwargs):
    runs = mlrun.get_run_db().list_runs(**kwargs)
    assert len(runs) == expected_number_of_runs
    return runs
