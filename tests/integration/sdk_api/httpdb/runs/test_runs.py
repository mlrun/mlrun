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
import datetime
import http
import json

import pytest

import mlrun
import mlrun.common.helpers
import mlrun.common.schemas
import tests.integration.sdk_api.base
from tests.conftest import examples_path


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
        iterations = 3
        for project in projects:
            project_obj = mlrun.new_project(project)
            project_obj.save()
            for name in run_names:
                for suffix in suffixes:
                    uid = f"{name}-uid-{suffix}"
                    for iteration in range(iterations):
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
        _list_and_assert_objects(
            expected_number_of_runs=len(projects)
            * len(run_names)
            * len(suffixes)
            * iterations,
            project="*",
            iter=True,
        )

        # basic list, specific project, all iterations, so 3 names * 3 uids * 3 iterations = 27
        _list_and_assert_objects(
            expected_number_of_runs=len(run_names) * len(suffixes) * iterations,
            project=projects[0],
            iter=True,
        )

        # basic list, specific project, only iteration 0, so 3 names = 3
        _list_and_assert_objects(
            expected_number_of_runs=len(run_names),
            project=projects[0],
            iter=False,
        )

        # basic list, specific project, only iteration 0, so 3 names * 3 uids = 9
        # using start time from to make sure we get all runs (and not just latest)
        _list_and_assert_objects(
            expected_number_of_runs=len(run_names) * len(suffixes),
            start_time_from=datetime.datetime.now() - datetime.timedelta(days=1),
            project=projects[0],
            iter=False,
        )

        # partitioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
        runs = _list_and_assert_objects(
            expected_number_of_runs=len(run_names),
            project=projects[0],
            partition_by=mlrun.common.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.common.schemas.SortField.created,
            partition_order=mlrun.common.schemas.OrderType.asc,
        )
        # sorted by ascending created so only the first ones created
        for run in runs:
            assert "first" in run["metadata"]["uid"]

        # partitioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
        runs = _list_and_assert_objects(
            expected_number_of_runs=len(run_names),
            project=projects[0],
            partition_by=mlrun.common.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
        )
        # sorted by descending updated so only the third ones created
        for run in runs:
            assert "third" in run["metadata"]["uid"]

        # partitioned list, specific project, 5 row per partition, so 3 names * 5 row = 15
        rows_per_partition = 5
        _list_and_assert_objects(
            expected_number_of_runs=len(run_names) * rows_per_partition,
            project=projects[0],
            partition_by=mlrun.common.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
            rows_per_partition=rows_per_partition,
            iter=True,
        )

        # partitioned list, specific project, 5 rows per partition, max of 2 partitions, so 2 names * 5 rows = 10
        _list_and_assert_objects(
            expected_number_of_runs=10,
            project=projects[0],
            partition_by=mlrun.common.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
            rows_per_partition=rows_per_partition,
            max_partitions=2,
            iter=True,
        )

        # partitioned list, specific project, 4 rows per partition, max of 2 partitions, but only iter=0 so each
        # partition has 3 rows, so 2 * 3 = 6
        _list_and_assert_objects(
            expected_number_of_runs=6,
            project=projects[0],
            partition_by=mlrun.common.schemas.RunPartitionByField.name,
            partition_sort_by=mlrun.common.schemas.SortField.updated,
            partition_order=mlrun.common.schemas.OrderType.desc,
            rows_per_partition=4,
            max_partitions=2,
            iter=False,
        )

        # Some negative testing - no sort by field
        with pytest.raises(mlrun.errors.MLRunBadRequestError):
            _list_and_assert_objects(
                expected_number_of_runs=0,
                project=projects[0],
                partition_by=mlrun.common.schemas.RunPartitionByField.name,
            )
        # An invalid partition-by field - will be failed by fastapi due to schema validation.
        with pytest.raises(mlrun.errors.MLRunHTTPError) as excinfo:
            _list_and_assert_objects(
                expected_number_of_runs=0,
                project=projects[0],
                partition_by="key",
                partition_sort_by=mlrun.common.schemas.SortField.updated,
            )
        assert (
            excinfo.value.response.status_code
            == http.HTTPStatus.UNPROCESSABLE_ENTITY.value
        )

        # expecting 3 since we're getting back all iterations for that uid
        _list_and_assert_objects(
            expected_number_of_runs=3,
            project=projects[0],
            uid=f"{run_names[0]}-uid-{suffixes[0]}",
            iter=True,
        )

        uid_list = [f"{run_names[0]}-uid-{suffix}" for suffix in suffixes]
        runs = _list_and_assert_objects(
            expected_number_of_runs=len(uid_list),
            project=projects[0],
            uid=uid_list,
            iter=False,
        )
        uid_list = set(uid_list)
        for run in runs:
            assert run["metadata"]["uid"] in uid_list
            uid_list.remove(run["metadata"]["uid"])

    def test_job_file(self, ensure_default_project):
        filename = f"{examples_path}/training.py"
        fn = mlrun.code_to_function(filename=filename, kind="job")
        assert fn.kind == "job", "kind not set, test failed"
        assert fn.spec.build.functionSourceCode, "code not embedded"
        assert fn.spec.build.origin_filename == filename, "did not record filename"
        assert type(fn.metadata.labels) == dict, "metadata labels were not set"
        run = fn.run(workdir=str(examples_path), local=True)

        project, uri, tag, hash_key = mlrun.common.helpers.parse_versioned_object_uri(
            run.spec.function
        )
        local_fn = mlrun.get_run_db().get_function(
            uri, project, tag=tag, hash_key=hash_key
        )
        assert local_fn["spec"]["command"] == filename, "wrong command path"
        assert (
            local_fn["spec"]["build"]["functionSourceCode"]
            == fn.spec.build.functionSourceCode
        ), "code was not copied to local function"


def _list_and_assert_objects(expected_number_of_runs: int, **kwargs):
    runs = mlrun.get_run_db().list_runs(**kwargs)
    assert len(runs) == expected_number_of_runs
    return runs
