# Copyright 2026 Iguazio
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

import datetime

import mlrun
import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects

from framework.db.sqldb.db import SQLDB
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestResourceCountersDB(TestDatabaseBase):
    """Direct DB-level coverage for the per-(project, kind)/per-project counters
    for functions, alert configs, and workflows."""

    def test_calculate_functions_counters_groups_by_project_and_kind(self):
        """Distinct ``name``s tally per (project, kind); versions/tags collapse."""
        self._store_function("proj-a", "fn-train", kind="job")
        # Same function name+kind, different uid → still one logical function.
        self._store_function("proj-a", "fn-train", kind="job", versioned=True)
        self._store_function("proj-a", "fn-serve", kind="serving")
        # ``new_function(kind="nuclio")`` is rewritten by the SDK to the
        # ``remote`` runtime — assert against what actually lands in the DB.
        self._store_function("proj-b", "fn-nuclio", kind="nuclio")
        self._store_function("proj-b", "fn-nuclio-2", kind="nuclio")

        counts = SQLDB._calculate_functions_counters(self._db_session)
        # The helper returns lists in the SQL GROUP BY order, which is not
        # guaranteed across backends — sort per project before comparing.
        normalized = {project: sorted(pairs) for project, pairs in counts.items()}
        assert normalized == {
            "proj-a": [("job", 1), ("serving", 1)],
            "proj-b": [("remote", 2)],
        }

    def test_calculate_alert_configs_counters_groups_by_project(self):
        """Each stored alert config contributes once to its project's tally."""
        self._create_alert(project="proj-a", name="alert-1")
        self._create_alert(project="proj-a", name="alert-2")
        self._create_alert(project="proj-b", name="alert-1")

        counts = SQLDB._calculate_alert_configs_counters(self._db_session)
        assert counts == {"proj-a": 2, "proj-b": 1}

    def test_calculate_workflow_counters_reads_project_spec(self):
        """Workflow definitions live in the pickled project ``spec.workflows``."""
        self._create_project_with_workflows(
            "proj-a",
            workflows=[{"name": "main"}, {"name": "nightly"}],
        )
        self._create_project_with_workflows("proj-b", workflows=[{"name": "cron"}])
        self._create_project_with_workflows("proj-empty", workflows=[])

        counts = SQLDB._calculate_workflow_counters(self._db_session)
        assert counts == {"proj-a": 2, "proj-b": 1, "proj-empty": 0}

    def _store_function(
        self,
        project: str,
        name: str,
        kind: str,
        versioned: bool = False,
    ) -> None:
        function = mlrun.new_function(
            name=name,
            project=project,
            kind=kind,
            tag="latest",
            command="run.py",
            image="test/test",
        )
        self._db.store_function(
            self._db_session,
            function=function.to_dict(),
            name=name,
            project=project,
            versioned=versioned,
        )

    def _create_alert(self, project: str, name: str) -> None:
        alert = alert_objects.AlertConfig(
            project=project,
            name=name,
            summary="test",
            severity=alert_objects.AlertSeverity.LOW,
            entities=alert_objects.EventEntities(
                kind=alert_objects.EventEntityKind.JOB,
                project=project,
                ids=["1"],
            ),
            trigger=alert_objects.AlertTrigger(events=[alert_objects.EventKind.FAILED]),
            notifications=[
                alert_objects.AlertNotification(
                    notification=mlrun.common.schemas.Notification(
                        kind="slack",
                        name="slack",
                        secret_params={"webhook": "https://slack.com/api/api.test"},
                    )
                )
            ],
            reset_policy=alert_objects.ResetPolicy.AUTO,
        )
        self._db.store_alert(self._db_session, alert)

    def _create_project_with_workflows(self, name: str, workflows: list[dict]) -> None:
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(
                name=name,
                created=datetime.datetime.now() - datetime.timedelta(seconds=1),
            ),
            spec=mlrun.common.schemas.ProjectSpec(
                description="test", workflows=workflows
            ),
        )
        self._db.create_project(self._db_session, project)
