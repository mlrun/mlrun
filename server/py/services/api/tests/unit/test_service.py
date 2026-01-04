# Copyright 2025 Iguazio
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
import asyncio
import datetime
import typing
import unittest.mock
import uuid

from sqlalchemy.orm import Session

import mlrun.common.runtimes
import mlrun.db

from services.api.daemon import daemon
from services.api.tests.unit.conftest import TestAPIBase


class TestService(TestAPIBase):
    @classmethod
    def custom_setup_class(cls):
        cls._project = "test-project"
        cls._service = daemon.service

    async def test_retry_job(self, db: Session):
        mlrun.mlconf.function.spec.retry.backoff.min_base_delay = "0s"
        run_uid = "test-job-uid"
        run = self._generate_retry_job(uid=run_uid)
        run_db = mlrun.db.get_run_db()
        with unittest.mock.patch(
            "framework.api.utils.submit_run_from_body",
            return_value=unittest.mock.Mock(),
        ) as mock_submit_run_from_body:
            run_db.store_run(struct=run, uid=run_uid, project=self._project)
            await self._service._retry_jobs()
            await asyncio.sleep(1)
            mock_submit_run_from_body.assert_called_once()

    async def test_retry_job_retry_exhausted(self, db: Session):
        run_uid = "test-job-uid"
        run = self._generate_retry_job(uid=run_uid, count=2, retry_count=2)
        assert (
            run["status"]["state"]
            == mlrun.common.runtimes.constants.RunStates.pending_retry
        )
        run_db = mlrun.db.get_run_db()
        with unittest.mock.patch(
            "framework.api.utils.submit_run_from_body",
            return_value=unittest.mock.Mock(),
        ) as mock_submit_run_from_body:
            run_db.store_run(struct=run, uid=run_uid, project=self._project)
            await self._service._retry_jobs()
            mock_submit_run_from_body.assert_not_called()

        run = run_db.read_run(uid=run_uid, project=self._project)
        assert run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.error
        assert run["status"]["status_text"] == "Run retries exhausted"

    async def test_retried_jobs_cache(self, db: Session):
        """This test ensures that the retry X is submitted exactly once."""
        mlrun.mlconf.function.spec.retry.backoff.min_base_delay = "0s"
        run_uid = "test-job-uid"
        run = self._generate_retry_job(uid=run_uid)
        run_db = mlrun.db.get_run_db()
        with unittest.mock.patch(
            "framework.api.utils.submit_run_from_body",
            return_value=unittest.mock.Mock(),
        ) as mock_submit_run_from_body:
            run_db.store_run(struct=run, uid=run_uid, project=self._project)
            await self._service._retry_jobs()
            assert run_uid in self._service._retry_in_progress_run_uids
            # Next retry should not submit the job again
            await self._service._retry_jobs()
            await asyncio.sleep(1)
            mock_submit_run_from_body.assert_called_once()
            assert not self._service._retry_in_progress_run_uids

    async def test_retry_job_paginated_list_runs(self, db: Session):
        mlrun.mlconf.monitoring.runs.retry.fetch_runs_limit = 3
        mlrun.mlconf.function.spec.retry.backoff.min_base_delay = "0s"
        run_uids = [str(uuid.uuid4()) for _ in range(10)]
        for run_uid in run_uids:
            run = self._generate_retry_job(uid=run_uid)
            mlrun.db.get_run_db().store_run(
                struct=run, uid=run_uid, project=self._project
            )

        # Create a running job - should be filtered out
        running_run_uid = "running_run"
        run = self._generate_retry_job(
            uid=running_run_uid, state=mlrun.common.runtimes.constants.RunStates.running
        )
        mlrun.db.get_run_db().store_run(
            struct=run, uid=running_run_uid, project=self._project
        )

        # There seems to be some OS level caching that prevents the new db session from seeing the new runs immediately.
        # Also, waiting less than a second does not work.
        await asyncio.sleep(1)
        with unittest.mock.patch(
            "framework.api.utils.submit_run_from_body",
            return_value=unittest.mock.Mock(),
        ) as mock_submit_run_from_body:
            await self._service._retry_jobs()
            assert mock_submit_run_from_body.call_count == 10

    async def test_retry_stale_job(self, db: Session):
        staleness_threshold = 60 * 24 * 1
        mlrun.mlconf.monitoring.runs.retry.staleness_threshold = staleness_threshold
        run_uid = "test-stale-job-uid"
        run = self._generate_retry_job(uid=run_uid)

        run_db = mlrun.db.get_run_db()
        run_db.store_run(struct=run, uid=run_uid, project=self._project)

        # manually add the run to the retry in progress dictionary to simulate a stale run
        self._service._retry_in_progress_run_uids[run_uid] = datetime.datetime.now(
            datetime.UTC
        ) - datetime.timedelta(days=2)

        await self._service._retry_jobs()

        run = run_db.read_run(uid=run_uid, project=self._project)
        assert (
            run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.aborted
        )
        assert (
            f"Retry aborted: run was pending retry for more than {staleness_threshold} minutes"
            in run["status"]["status_text"]
        )

    def _generate_retry_job(
        self,
        uid: str = "test-job-uid",
        project: typing.Optional[str] = None,
        state: typing.Optional[str] = None,
        count: int = 3,
        retry_count: int = 0,
        base_delay: str = "1s",
    ):
        return {
            "metadata": {
                "name": "test-job",
                "project": project or self._project,
                "uid": uid or str(uuid.uuid4()),
                "labels": {
                    "kind": "job",
                },
            },
            "spec": {
                "function": f"{self._project}/test@c37401e5c6bf55b826bafa336a2c6e796280292a",
                "retry": {
                    "count": count,
                    "backoff": {
                        "base_delay": base_delay,
                    },
                },
            },
            "status": {
                "state": state
                or mlrun.common.runtimes.constants.RunStates.pending_retry,
                "error": "some error",
                "retry_count": retry_count,
                "end_time": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        }
